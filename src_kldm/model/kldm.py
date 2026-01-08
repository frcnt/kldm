from typing import Literal, Optional, Union

import torch
from torch import nn
from torch_geometric.data import Batch, Data

from src_kldm.model.continuous import (
    AnalogBitsContinuousDiffusion,
    ContinuousDiffusion,
)
from src_kldm.model.discrete import DiscreteDiffusion
from src_kldm.model.tdm import TDM
from src_kldm.nn.arch import CSPVNet


class KLDM(nn.Module):
    def __init__(
            self,
            net: CSPVNet,
            diffusion_v: Optional[TDM] = None,
            diffusion_l: Optional[ContinuousDiffusion] = None,
            diffusion_h: Optional[
                Union[
                    DiscreteDiffusion,
                    ContinuousDiffusion,
                    AnalogBitsContinuousDiffusion,
                ]
            ] = None,
            **kwargs,
    ):
        super().__init__()

        self.net = net

        self.diffusions = nn.ModuleDict(
            {"v": diffusion_v, "l": diffusion_l, "h": diffusion_h}
        )

    def loss_diffusion(self, t: torch.Tensor, batch: Batch | Data):
        latents, targets = self.training_targets(t=t, batch=batch)

        preds = self.net.forward(
            t=t,
            **latents,
            node_index=batch.batch,
            edge_node_index=batch.edge_node_index,
        )

        losses = {}

        for key in targets:
            loss = self.diffusions[key].loss_diffusion(
                preds[key],
                targets[key],
                (
                    t if key == "l" else t[batch.batch]
                ),  # cast time, when computing node-level property
                latents[key],
            )
            losses[key] = loss

        return losses

    def training_targets(
            self, t: torch.Tensor, batch: Batch | Data
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        targets = {}

        index = batch.batch

        # velocity/position part
        if self.diffusion_v is None:
            v_t, pos_t = torch.zeros_like(batch.pos), batch.pos
        else:
            (v_t, pos_t), target_v_t = self.diffusion_v.training_targets(
                t[index], batch.pos, index=batch.batch
            )
            targets["v"] = target_v_t

        # lattice part
        if self.diffusion_l is None:
            l_t = batch.l
        else:
            l_t, target_l_t = self.diffusion_l.training_targets(t=t, x=batch.l)
            targets["l"] = target_l_t

        # atomic species part
        if self.diffusion_h is None:  # CSP
            h_t = batch.h
        else:  # DNG
            h_t, target_h_t = self.diffusion_h.training_targets(t=t[index], x=batch.h)
            targets["h"] = target_h_t

        latents = {"pos": pos_t, "v": v_t, "h": h_t, "l": l_t}

        return latents, targets

    @torch.inference_mode()
    def sample_prior(
            self,
            batch: Batch | Data,
    ):
        index = batch.batch

        if self.diffusion_v is None:
            assert batch.pos is not None
            pos = batch.pos
            v = torch.zeros_like(pos)
        else:
            pos, v = self.diffusion_v.sample_prior(index)

        if self.diffusion_h is None:
            assert batch.h is not None
            h = batch.h
        else:
            num_nodes = len(index)
            h = self.diffusions["h"].sample_prior(n=num_nodes)

        if self.diffusion_l is None:
            assert batch.l is not None
            l = batch.l
        else:
            num_graphs = torch.amax(index) + 1
            l = self.diffusions["l"].sample_prior(n=num_graphs)

        return pos, v, h, l

    @torch.no_grad()
    def sample(
            self,
            batch: Batch | Data,
            method: Literal["em", "pc"],
            return_traj: bool = False,
            n_steps: int = 1000,
            ts: float = 1.0,
            tf: float = 1e-3,
            **kwargs,
    ) -> Union[
        dict[str, torch.Tensor],
        tuple[dict[str, torch.Tensor], dict[str, list[torch.Tensor]]],
    ]:

        node_index, edge_node_index = batch.batch, batch.edge_node_index
        num_graphs = batch.num_graphs
        device = node_index.device

        ts = torch.linspace(ts, tf, n_steps + 1, device=device)
        pos_t, v_t, h_t, l_t = self.sample_prior(batch=batch)

        if return_traj:
            traj = {"pos": [pos_t], "v": [v_t], "h": [h_t], "l": [l_t]}

        for i in range(n_steps):
            t = ts[i]
            dt = ts[i + 1] - t

            t = torch.full((num_graphs, 1), t, device=device)

            if method == "em":
                pos_t, v_t, h_t, l_t = self.reverse_step_em(
                    t=t,
                    dt=dt,
                    pos_t=pos_t,
                    v_t=v_t,
                    h_t=h_t,
                    l_t=l_t,
                    node_index=node_index,
                    edge_node_index=edge_node_index,
                    **kwargs,
                )
            elif method == "pc":
                pos_t, v_t, h_t, l_t = self.reverse_step_pc(
                    t=t,
                    dt=dt,
                    pos_t=pos_t,
                    v_t=v_t,
                    h_t=h_t,
                    l_t=l_t,
                    node_index=node_index,
                    edge_node_index=edge_node_index,
                    **kwargs,
                )

            if return_traj:
                traj["pos"].append(pos_t)
                traj["v"].append(v_t)
                traj["h"].append(h_t)
                traj["l"].append(l_t)

        samples = self.final_step(
            t=t + dt,
            pos_t=pos_t,
            v_t=v_t,
            h_t=h_t,
            l_t=l_t,
            node_index=node_index,
            edge_node_index=edge_node_index,
        )

        if return_traj:
            return samples, traj

        else:
            return samples

    def reverse_step_em(
            self,
            t: torch.Tensor,
            dt: torch.Tensor,
            pos_t: torch.Tensor,
            v_t: torch.Tensor,
            h_t: torch.Tensor,
            l_t: torch.Tensor,
            node_index: torch.Tensor,
            edge_node_index: torch.Tensor,
            exp: bool = False,
            **_,
    ):

        preds = self.net.forward(
            t=t,
            pos=pos_t,
            v=v_t,
            h=h_t,
            l=l_t,
            node_index=node_index,
            edge_node_index=edge_node_index,
        )
        # reverse step on each modality
        if self.diffusion_v:
            pos_t, v_t = self.diffusion_v.reverse_step_em(
                t=t,
                v_t=v_t,
                pos_t=pos_t,
                pred_v_t=preds["v"],
                dt=dt,
                node_index=node_index,
            )

        if self.diffusion_l:
            l_t = self.diffusion_l.reverse_step(
                t=t, x_t=l_t, pred=preds["l"], dt=dt, exp=exp
            )

        if self.diffusion_h:
            h_t = self.diffusion_h.reverse_step(
                t=t[node_index], x_t=h_t, pred=preds["h"], dt=dt, exp=exp
            )

        return pos_t, v_t, h_t, l_t

    def reverse_step_pc(
            self,
            t,
            dt,
            pos_t,
            v_t,
            h_t,
            l_t,
            node_index,
            edge_node_index,
            exp: bool = False,
            tau: float = 0.25,
            n_correction_steps: int = 1,
            correct_pos: bool = True,
            **_,
    ):
        assert (
                not correct_pos or n_correction_steps == 1
        ), "Only 1 correction step supported currently."
        assert (
            self.diffusion_v
        ), "PC sampler can only be used with 'diffusion_v' enabled."

        # Corrector steps
        for _ in range(n_correction_steps):
            preds = self.net.forward(
                t=t,
                pos=pos_t,
                v=v_t,
                h=h_t,
                l=l_t,
                node_index=node_index,
                edge_node_index=edge_node_index,
            )

            pos_t, v_t = self.diffusion_v.reverse_step_corrector(
                t=t,
                v_t=v_t,
                pos_t=pos_t,
                pred_v_t=preds["v"],
                dt=dt,
                node_index=node_index,
                tau=tau,
                correct_pos=correct_pos,
            )

            if self.diffusion_l:
                l_t = self.diffusion_l.reverse_step_corrector(
                    t=t,
                    x_t=l_t,
                    pred=preds["l"],
                    dt=dt,
                    tau=tau,
                )

            if self.diffusion_h and hasattr(self.diffusion_h, "reverse_step_corrector"):
                h_t = self.diffusion_h.reverse_step_corrector(
                    t=t,
                    x_t=h_t,
                    pred=preds["h"],
                    dt=dt,
                    exp=exp,
                    tau=tau,
                    index=node_index,
                )

        # Predictor step
        preds = self.net.forward(
            t=t,
            pos=pos_t,
            v=v_t,
            h=h_t,
            l=l_t,
            node_index=node_index,
            edge_node_index=edge_node_index,
        )

        pos_t, v_t = self.diffusion_v.reverse_step_predictor(
            t=t,
            v_t=v_t,
            pos_t=pos_t,
            pred_v_t=preds["v"],
            dt=dt,
            node_index=node_index,
        )

        if self.diffusion_l:
            l_t = self.diffusion_l.reverse_step_predictor(
                t=t,
                x_t=l_t,
                pred=preds["l"],
                dt=dt,
            )

        if self.diffusion_h:
            if hasattr(self.diffusion_h, "reverse_step_predictor"):
                h_t = self.diffusion_h.reverse_step_predictor(
                    t=t[node_index], x_t=h_t, pred=preds["h"], dt=dt
                )
            else:
                h_t = self.diffusion_h.reverse_step(
                    t=t[node_index], x_t=h_t, pred=preds["h"], dt=dt, exp=exp
                )

        return pos_t, v_t, h_t, l_t

    def final_step(
            self,
            t: torch.Tensor,
            pos_t: torch.Tensor,
            v_t: torch.Tensor,
            h_t: torch.Tensor,
            l_t: torch.Tensor,
            node_index: torch.Tensor,
            edge_node_index: torch.Tensor,
    ):

        if self.diffusion_h:
            if isinstance(self.diffusion_h, AnalogBitsContinuousDiffusion):
                h_t = self.diffusion_h.embedding.bit2int(h_t)

        return dict(pos=pos_t, h=h_t, l=l_t)

    @property
    def diffusion_v(self) -> TDM:
        return self.diffusions["v"]

    @property
    def diffusion_l(self) -> ContinuousDiffusion:
        return self.diffusions["l"]

    @property
    def diffusion_h(
            self,
    ) -> Optional[
        Union[
            ContinuousDiffusion,
            DiscreteDiffusion,
            AnalogBitsContinuousDiffusion,
        ]
    ]:
        return self.diffusions["h"]
