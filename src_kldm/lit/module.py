import warnings
from typing import Literal, Optional, Sequence

import torch
from ase.data import chemical_symbols
from pymatgen.core import Lattice, Structure
from pytorch_lightning import LightningModule
from torch_geometric.data import Batch

from src_kldm.data.transforms import (
    ContinuousIntervalAngles,
    ContinuousIntervalLengths,
)
from src_kldm.metrics.csp import CSPMetrics
from src_kldm.metrics.dng import DNGMetrics
from src_kldm.model.kldm import KLDM


def sample_uniform(
        size: Sequence[int],
        device: torch.device,
        lb: float = 0.0,
        ub: float = 1.0,
):
    """
    NOTE: invert bounds to get sample in (lb, ub], instead of [lb, ub)
    """
    return (lb - ub) * torch.rand(size, device=device) + ub


def structures_from_tensors(
        tensors: dict[str, torch.Tensor],
        ptr: torch.Tensor,
        decoder: dict[int, str] | list[str],
        transform_lengths: ContinuousIntervalLengths,
        transform_angles: ContinuousIntervalAngles,
        pos_range: float = 1.0,
) -> list[Structure]:
    h = tensors["h"].to("cpu")
    if h.ndim > 1 and h.shape[1] > 1:
        h = torch.argmax(h, dim=1)

    h = h.numpy()
    pos = tensors["pos"].to("cpu").numpy() / pos_range
    l = tensors["l"].to("cpu").numpy()
    ptr = ptr.to("cpu").numpy()

    structures_list = []

    for i, (start_idx, end_idx) in enumerate(zip(ptr[:-1], ptr[1:])):
        coords = pos[start_idx:end_idx, :]
        symbols = [decoder[idx.item()] for idx in h[start_idx:end_idx]]
        n = len(symbols)

        li = l[i]
        log_abc, tan_angles = li[:3], li[3:]  # FIXME: make more general
        a, b, c = transform_lengths.invert_one(log_abc, n)
        alpha, beta, gamma = transform_angles.invert_one(tan_angles)
        lattice = Lattice.from_parameters(
            a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma
        )
        coords = coords % 1.0
        struc = Structure(
            lattice=lattice, species=symbols, coords=coords, coords_are_cartesian=False
        )
        struc = struc.get_sorted_structure()

        structures_list.append(struc)

    return structures_list


def structures_from_batch(
        batch: Batch,
        decoder: dict[int, str] | list[str],
        transform_lengths: ContinuousIntervalLengths,
        transform_angles: ContinuousIntervalAngles,
        pos_range: float = 1.0,
):
    tensors = {"h": batch.h, "pos": batch.pos, "l": batch.l}
    ptr = batch.ptr

    return structures_from_tensors(
        tensors,
        ptr,
        decoder,
        transform_lengths,
        transform_angles,
        pos_range=pos_range,
    )


class LitKLDM(LightningModule):
    def __init__(
            self,
            model: KLDM,
            task: Literal["csp", "dng"],
            transform_lengths: ContinuousIntervalLengths,
            transform_angles: ContinuousIntervalAngles,
            decoder: list[str] = chemical_symbols,
            lr: float = 1e-4,
            with_ema: bool = True,
            ema_decay: float = 0.999,
            ema_start: int = 100,
            loss_weights: Optional[dict] = None,
            metrics: Optional[CSPMetrics | DNGMetrics] = None,
            sampling_kwargs: Optional[dict[str, dict]] = None,
    ):
        super().__init__()
        self.model = model
        if with_ema:
            self.ema_model = torch.optim.swa_utils.AveragedModel(
                model,
                multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(ema_decay),
            )
        else:
            self.ema_model = None

        self.decoder = decoder
        self.transform_lengths = transform_lengths
        self.transform_angles = transform_angles

        self.metrics = metrics

        if task == "csp":
            default_weights = {"v": 1.0, "l": 1.0}
        else:
            default_weights = {"v": 1.0, "l": 1.0, "h": 1.0}

        self.loss_weights = default_weights if loss_weights is None else loss_weights

        self.save_hyperparameters(
            ignore=["model", "transform_lengths", "transform_angles", "metrics"]
        )

    def basic_step(self, batch):
        t = sample_uniform(lb=1e-3, size=(batch.num_graphs, 1), device=self.device)
        losses = self.model.loss_diffusion(t=t, batch=batch)  # per data-point
        loss = torch.mean(
            torch.stack(
                [self.loss_weights[key] * losses[key] for key in losses], dim=-1
            )
        )
        losses["weighted"] = loss

        return loss, losses

    def training_step(
            self, batch, batch_idx: int, dataloader_idx: Optional[int] = None
    ):

        loss, losses = self.basic_step(batch)
        self.log_dict({f"train/loss_{key}": losses[key] for key in losses})

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if (
                self.ema_model is not None
                and self.trainer.current_epoch > self.hparams.ema_start
        ):
            self.ema_model.update_parameters(self.model)

    def validation_step(
            self, batch, batch_idx: int, dataloader_idx: Optional[int] = None
    ):
        loss, losses = self.basic_step(batch)
        self.log_dict({f"val/loss_{key}": losses[key] for key in losses}, on_epoch=True)

        return self.sampling_step(batch=batch, **self.hparams.sampling_kwargs["val"])

    def test_step(self, batch, batch_idx):
        return self.sampling_step(batch=batch, **self.hparams.sampling_kwargs["test"])

    def sampling_step(
            self,
            batch,
            **kwargs,
    ) -> list[Structure]:
        structures = self.sample(batch, **kwargs)

        gt_structures = self.structures_from_batch(batch)

        self.metrics.update(structures, gt_structures)
        return structures

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> list[Structure]:
        return self.sample(batch, **self.hparams.sampling_kwargs["predict"])

    def compute_and_log_metrics(self, stage: str):
        summary = self.metrics.summarize()
        self.log_dict(
            {f"{stage}/{key}": summary[key] for key in summary}, on_epoch=True
        )

    def on_validation_epoch_start(self) -> None:
        self.metrics.reset()

    def on_validation_epoch_end(self):
        self.compute_and_log_metrics(stage="val")

    def on_test_epoch_start(self) -> None:
        self.metrics.reset()

    def on_test_epoch_end(self):
        self.compute_and_log_metrics(stage="test")

    @torch.no_grad()
    def sample(
            self,
            batch,
            force_ema: bool = True,
            **kwargs,
    ) -> list[Structure]:
        model = self.get_model(ema=force_ema)
        samples = model.sample(batch, **kwargs)
        try:
            ptr = batch.ptr
            structures = self.structures_from_tensors(samples, ptr)
        except Exception as e:
            warnings.warn(f"In the conversion the following error occurred: {e}")
            return []
        return structures

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            amsgrad=True,
            foreach=True,
            weight_decay=1e-12,
        )
        return opt

    def get_model(self, ema: bool = False) -> KLDM:
        if self.ema_model and (ema or self.current_epoch > self.hparams.ema_start):
            model: KLDM = self.ema_model.module
            print("Loading EMA model.")
        else:
            model: KLDM = self.model
            print("Loading current model.")
        return model

    def structures_from_batch(self, batch: Batch):
        return structures_from_batch(
            batch,
            decoder=self.decoder,
            transform_lengths=self.transform_lengths,
            transform_angles=self.transform_angles,
        )

    def structures_from_tensors(
            self, samples: dict[str, torch.Tensor], ptr: torch.Tensor
    ):
        return structures_from_tensors(
            samples,
            ptr=ptr,
            decoder=self.decoder,
            transform_lengths=self.transform_lengths,
            transform_angles=self.transform_angles,
        )
