import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

from src_kldm.model.distributions import (
    DistributionGaussian,
    d_log_p_wrapped_normal,
    sigma_norm,
)
from src_kldm.nn.utils import scatter_center, wrap


class TDM(nn.Module):
    def __init__(
            self,
            scale_pos: float = 2.0 * math.pi,
            k_wn_score: int = 13,
            zero_cog_v: bool = True,
            tf: float = 2.0,
            v0_init_scale: float = 0.0,
            simplified_parameterization: bool = True,
            n_sigmas: int = 2000,
            **kwargs,
    ):
        super().__init__()

        self.velocity_distribution = DistributionGaussian(zero_cog=zero_cog_v)
        self.init_velocity_distribution = DistributionGaussian(
            zero_cog=zero_cog_v, scale=v0_init_scale
        )

        self.scale_pos = scale_pos
        self.k_wn_score = k_wn_score

        self.zero_cog_v = zero_cog_v
        self.tf = tf

        self.simplified_parameterization = simplified_parameterization

        if simplified_parameterization:
            assert (
                    float(v0_init_scale) == 0.0
            ), "Simplified parameterization can only be used with zero initial velocities."
            _sigmas = self._sigma_r_t(torch.linspace(0.0, tf, n_sigmas))
            _sigma_norms = sigma_norm(_sigmas, T=self.scale_pos, N=self.k_wn_score)
            self.register_buffer("_sigma_norms", _sigma_norms)

    def training_targets(
            self, t01: torch.Tensor, pos01: torch.Tensor, index: torch.LongTensor
    ):

        # map t01 to internal time
        t = self.tf * t01

        # 1) sample noisy velocity for positions
        # 2) sample noisy positions accordingly
        # 3) the target is the sum of scores
        # v is used just for the argument

        v = self.init_velocity_distribution.sample(index)
        v_t, target_v_t = self._sample_v_t(t=t, v=v, index=index)
        pos_t, target_pos_t = self._sample_pos_t(
            t=t, pos=self.scale_pos * wrap(pos01), v=v, v_t=v_t, index=index
        )

        if self.simplified_parameterization:
            prefactor_t = self._prefactor_t(t)
            sigma_norm_t = torch.sqrt(self._sigma_norm_t(t))

            target = target_pos_t / prefactor_t / sigma_norm_t
        else:
            target = target_v_t + target_pos_t

        latents = (v_t / self.scale_pos, pos_t / self.scale_pos)
        return latents, target

    def loss_diffusion(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            *args: Optional[torch.Tensor],
    ):
        assert pred.shape == target.shape
        return F.mse_loss(pred, target)

    def _sample_v_t(self, t: torch.Tensor, v: torch.Tensor, index: torch.Tensor):
        # Sample xi_t from N(xi_t | mu(xi), std(xi))
        eps_v = self.velocity_distribution.sample(
            index=index
        )  # include zero_cog, already
        mu_v_t = self._mu_v_t_coeff(t) * v
        sigma_v_t = self._sigma_v_t(t)
        v_t = mu_v_t + eps_v * sigma_v_t

        target_v_t = -eps_v / sigma_v_t
        return v_t, target_v_t

    def _mu_v_t_coeff(self, t: torch.Tensor):
        return torch.exp(-t)

    def _sigma_v_t(self, t: torch.Tensor, eps: float = 1e-6):
        return torch.sqrt(1.0 - torch.exp(-2.0 * t) + eps)

    def _prefactor_t(self, t: torch.Tensor):
        return (1.0 - torch.exp(-t)) / (1.0 + torch.exp(-t))

    def _sample_pos_t(
            self,
            t: torch.Tensor,
            pos: torch.Tensor,
            v: torch.Tensor,
            v_t: torch.Tensor,
            index: torch.LongTensor,
    ):

        mu_r = self._mu_r_t(t, v, v_t)
        sigma_r = self._sigma_r_t(t)
        eps_r = self.velocity_distribution.sample(
            index=index
        )  # include zero_cog, already

        r = mu_r + sigma_r * eps_r

        r = self._wrap(r)
        mu_r = self._wrap(mu_r)
        # r = logm(inv(g_0^{-1})g_t) => g_t = g_0expm(r)

        pos_t = self._wrap(pos + r)

        prefactor = self._prefactor_t(t)
        target_pos_t = d_log_p_wrapped_normal(
            r, mu_r, sigma_r, N=self.k_wn_score, T=self.scale_pos
        )

        target_pos_t = prefactor * target_pos_t

        if self.zero_cog_v:
            target_pos_t = scatter_center(target_pos_t, index=index)

        return pos_t, target_pos_t

    def _mu_r_t(self, t: torch.Tensor, v: torch.Tensor, v_t: torch.Tensor):
        prefactor = self._prefactor_t(t)
        return prefactor * (v_t + v)

    def _sigma_r_t(self, t: torch.Tensor, eps: float = 1e-6):
        return torch.sqrt(2.0 * t + 8.0 / (1.0 + torch.exp(t)) - 4.0 + eps)

    def _construct_score_v_t(
            self,
            t: torch.Tensor,
            v_t: torch.Tensor,
            pred_v_t: torch.Tensor,
            index: torch.LongTensor,
    ):

        if self.simplified_parameterization:
            # velocity part
            term_v = -v_t / self._sigma_v_t(t)[index] ** 2

            # WN part
            prefactor = self._prefactor_t(t)[index]
            sigma_norm_t = torch.sqrt(self._sigma_norm_t(t))[index]
            term_wn = pred_v_t * prefactor * sigma_norm_t

            score_v_t = term_v + term_wn
        else:
            score_v_t = pred_v_t

        return score_v_t

    @torch.no_grad()
    def sample_prior(self, index: torch.Tensor):
        num_nodes = len(index)
        # NOTE: we wrap the positions in [-.5, .5]
        #       we scale the velocity
        pos_scaled = wrap(torch.rand([num_nodes, 3], device=index.device))
        v_scaled = self.velocity_distribution.sample(index) / self.scale_pos

        return pos_scaled, v_scaled

    @torch.no_grad()
    def reverse_step_em(
            self,
            t: torch.Tensor,
            v_t: torch.Tensor,
            pos_t: torch.Tensor,
            pred_v_t: torch.Tensor,
            dt: torch.Tensor,
            node_index: torch.LongTensor,
            probability_flow: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Exponential Integrator.
        """
        pos_t = pos_t * self.scale_pos
        v_t = v_t * self.scale_pos
        t = t * self.tf
        dt = dt * self.tf

        dt_abs = abs(dt)

        score_v_t = self._construct_score_v_t(
            t, v_t=v_t, pred_v_t=pred_v_t, index=node_index
        )

        v_t_1 = math.exp(dt_abs) * v_t
        v_t_2 = 2.0 * (math.expm1(dt_abs)) * score_v_t

        if probability_flow:
            v_t = v_t_1 + v_t_2 / 2
        else:
            v_t_3 = math.sqrt(
                math.expm1(2.0 * dt_abs)
            ) * self.velocity_distribution.sample(node_index)
            v_t = v_t_1 + v_t_2 + v_t_3

        pos_t = self._wrap(pos_t - dt_abs * v_t)

        return pos_t / self.scale_pos, v_t / self.scale_pos

    @torch.no_grad()
    def reverse_step_predictor(
            self,
            t: torch.Tensor,
            v_t: torch.Tensor,
            pos_t: torch.Tensor,
            pred_v_t: torch.Tensor,
            dt: torch.Tensor,
            node_index: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        pos_t = pos_t * self.scale_pos
        v_t = v_t * self.scale_pos
        t = t * self.tf
        dt = dt * self.tf

        score_v_t = self._construct_score_v_t(
            t, v_t=v_t, pred_v_t=pred_v_t, index=node_index
        )

        r = self._mu_v_t_coeff(t + dt) / self._mu_v_t_coeff(t)
        sigma_v_t = self._sigma_v_t(t)
        prefactor = (r * sigma_v_t - self._sigma_v_t(t + dt)) * sigma_v_t

        v_t = r[node_index] * v_t + prefactor[node_index] * score_v_t
        pos_t = self._wrap(pos_t + dt * v_t)

        return pos_t / self.scale_pos, v_t / self.scale_pos

    @torch.no_grad()
    def reverse_step_corrector(
            self,
            t: torch.Tensor,
            v_t: torch.Tensor,
            pos_t: torch.Tensor,
            pred_v_t: torch.Tensor,
            dt: torch.Tensor,
            node_index: torch.LongTensor,
            tau: float,
            correct_pos: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos_t = pos_t * self.scale_pos
        v_t = v_t * self.scale_pos
        t = t * self.tf
        dt = dt * self.tf

        score_v_t = self._construct_score_v_t(
            t, v_t=v_t, pred_v_t=pred_v_t, index=node_index
        )

        denominator = scatter_mean(
            score_v_t.square().mean(dim=-1, keepdim=True), dim=0, index=node_index
        )

        delta = tau / denominator[node_index]
        eps = self.velocity_distribution.sample(node_index)

        v_t = v_t + delta * score_v_t + torch.sqrt(2 * delta) * eps
        if correct_pos:
            pos_t = self._wrap(pos_t + dt * v_t)

        return pos_t / self.scale_pos, v_t / self.scale_pos

    def _wrap(self, x: torch.Tensor):
        return wrap(x, x_range=(2.0 * math.pi) / self.scale_pos)

    def _sigma_norm_t(self, t: torch.Tensor):
        idx = torch.round(t / self.tf * (len(self._sigma_norms))).long() - 1
        return self._sigma_norms[idx]


if __name__ == "__main__":
    tdm = TDM(n_sigmas=1000)
    # print(tdm._sigma_norm_t(t))
    print(torch.sqrt(tdm._sigma_norms))
