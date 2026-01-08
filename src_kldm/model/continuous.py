from abc import ABC
from typing import Any, Literal, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

from src_kldm.model.base import BaseDiffusion
from src_kldm.nn.embedding import AnalogBitsEmbedding


class Schedule(ABC, nn.Module):
    def beta(self, t: torch.Tensor):
        raise NotImplementedError

    def integral_beta(self, t: torch.Tensor):
        raise NotImplementedError


class LinearSchedule(Schedule):
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0) -> None:
        super().__init__()
        self.register_buffer("_beta_min", torch.as_tensor(beta_min))
        self.register_buffer("_beta_max", torch.as_tensor(beta_max))

    def beta(self, t):
        return self._beta_min + (self._beta_max - self._beta_min) * t

    def integral_beta(self, t):
        return self._beta_min * t + 0.5 * (self._beta_max - self._beta_min) * t ** 2


class SDE(ABC, nn.Module):

    def diffusion(self, t: torch.Tensor):
        raise NotImplementedError

    def forward_drift(self, t: torch.Tensor, zt: torch.Tensor):
        raise NotImplementedError

    def loc_scale(self, t: torch.Tensor):
        raise NotImplementedError

    def reverse_drift(
            self,
            t: torch.Tensor,
            zt: torch.Tensor,
            score: torch.Tensor,
            eta: Optional[torch.Tensor | float] = None,
    ):
        f = self.forward_drift(t, zt)
        g = self.diffusion(t)

        if eta is None:
            eta = g

        return f - 0.5 * (g ** 2 + eta ** 2) * score


class VPSDE(SDE):
    def __init__(self, schedule: Schedule):
        super().__init__()
        self.schedule = schedule

    def diffusion(self, t: torch.Tensor):
        beta = self.schedule.beta(t)
        return torch.sqrt(beta)

    def forward_drift(
            self,
            t: torch.Tensor,
            zt: torch.Tensor,
    ):
        beta = self.schedule.beta(t)
        return -0.5 * beta * zt

    def loc_scale(self, t: torch.Tensor):
        beta_integral = self.schedule.integral_beta(t)
        loc = torch.exp(-0.5 * beta_integral)
        scale = torch.sqrt(1.0 - loc ** 2)

        return loc, scale


class BaseContinuousDiffusion(BaseDiffusion):
    def __init__(
            self,
            sde: SDE,
            dim: Sequence[int] | int,
    ):
        super().__init__()
        self.sde = sde
        if isinstance(dim, int):
            dim = [dim]
        self.register_buffer("dim", torch.as_tensor(dim, dtype=torch.long))

    def loss_diffusion(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            t: torch.Tensor,
            **kwargs: Optional[Any],
    ):
        raise NotImplementedError

    def training_targets(
            self, t: torch.Tensor, x: torch.Tensor, index: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        a, b = self.sde.loc_scale(t)
        eps = torch.randn_like(x)
        x_t = a * x + b * eps

        if self.parameterization == "eps":
            target = eps
        elif self.parameterization == "x0":
            target = x
        else:
            raise NotImplementedError

        return x_t, target

    def construct_score(
            self,
            t: torch.Tensor,
            x_t: torch.Tensor,
            pred: torch.Tensor,
    ):

        loc, scale = self.sde.loc_scale(t)

        if self.parameterization == "eps":
            score = -pred / scale
        elif self.parameterization == "x0":
            score = (loc * pred - x_t) / scale ** 2
        else:
            raise NotImplementedError

        return score

    @torch.inference_mode()
    def reverse_step(
            self,
            t: torch.Tensor,
            x_t: torch.Tensor,
            pred: torch.Tensor,
            dt: torch.Tensor,
            index: Optional[torch.Tensor] = None,
            exp: bool = False,
            eta: Optional[float] = 1.0,
            **_,
    ):
        score = self.construct_score(t=t, x_t=x_t, pred=pred)

        if exp:  # exploit semi-linearity
            # linear part
            alpha_curr, sigma_curr = self.sde.loc_scale(t)
            alpha_next, sigma_next = self.sde.loc_scale(t + dt)
            linear_term = alpha_next / alpha_curr * x_t

            # linear part
            gamma_curr = torch.log(alpha_curr) - torch.log(sigma_curr)
            gamma_next = torch.log(alpha_next) - torch.log(sigma_next)
            gamma_diff = gamma_next - gamma_curr
            nonlinear_term = (
                    (1 + eta ** 2)
                    * sigma_curr
                    * sigma_next
                    * (torch.expm1(gamma_diff))
                    * score
            )

            # noise part
            noise_std = (
                    eta
                    * sigma_next
                    * torch.sqrt(torch.clamp_min(-torch.expm1(-2.0 * gamma_diff), min=0.0))
            )
            noise_term = noise_std * torch.randn_like(x_t)

            return linear_term + nonlinear_term + noise_term

        else:
            drift_dt = self.sde.reverse_drift(t=t, zt=x_t, score=score) * dt
            diff_dt = (
                    self.sde.diffusion(t)
                    * torch.randn_like(x_t)
                    * torch.sqrt(torch.abs(dt))
            )
            return x_t + drift_dt + diff_dt

    @torch.no_grad()
    def reverse_step_predictor(
            self,
            t: torch.Tensor,
            x_t: torch.Tensor,
            pred: torch.Tensor,
            dt: torch.Tensor,
            **_,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        score = self.construct_score(t=t, x_t=x_t, pred=pred)

        alpha_curr, sigma_curr = self.sde.loc_scale(t)
        alpha_next, sigma_next = self.sde.loc_scale(t + dt)

        alpha_ratio = alpha_next / alpha_curr
        score_coeff = (alpha_ratio * sigma_curr - sigma_next) * sigma_curr

        x_t = alpha_ratio * x_t + score_coeff * score

        return x_t

    @torch.no_grad()
    def reverse_step_corrector(
            self,
            t: torch.Tensor,
            x_t: torch.Tensor,
            pred: torch.Tensor,
            tau: float,
            index: Optional[torch.Tensor] = None,
            **_,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        score = self.construct_score(t=t, x_t=x_t, pred=pred)

        if index is None:
            denominator = score.square().mean(dim=-1, keepdim=True)
            delta = tau / denominator
        else:
            denominator = scatter_mean(
                score.square().mean(dim=-1, keepdim=True), dim=0, index=index
            )

            delta = tau / denominator[index]

        eps = torch.randn_like(x_t)
        x_t = x_t + delta * score + torch.sqrt(2 * delta) * eps

        return x_t

    @torch.inference_mode()
    def sample_prior(self, n: int):
        return torch.randn((n, *self.dim), device=self.dim.device)


class ContinuousDiffusion(BaseContinuousDiffusion):
    def __init__(
            self,
            sde: SDE,
            parameterization: Literal["eps", "x0"],
            dim: Sequence[int] | int,
    ):
        super(ContinuousDiffusion, self).__init__(sde=sde, dim=dim)
        self.parameterization = parameterization

    def loss_diffusion(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            t: torch.Tensor,
            *args: Optional[Any],
    ):
        assert pred.shape == target.shape
        return F.mse_loss(pred, target)


class AnalogBitsContinuousDiffusion(BaseContinuousDiffusion):

    def __init__(
            self,
            sde: SDE,
            embedding: AnalogBitsEmbedding,
            clamp_pred_in_reverse: Optional[tuple[float, float]] = None,
    ):
        super().__init__(sde=sde, dim=embedding.embedding_dim)
        self.embedding = embedding
        self.clamp_pred_in_reverse = clamp_pred_in_reverse

    def loss_diffusion(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            t: torch.Tensor,
            **kwargs: Optional[Any],
    ):
        return F.mse_loss(pred, target)  # logits

    def training_targets(self, t: torch.Tensor, x: torch.Tensor, index: torch.Tensor):
        x_t, x_bits = super().training_targets(
            t=t, x=self.embedding.forward(x), index=index
        )
        return x_t, x_bits  # return bit version as target

    def construct_score(
            self,
            t: torch.Tensor,
            x_t: torch.Tensor,
            pred: torch.Tensor,
    ):
        if self.clamp_pred_in_reverse:
            assert self.parameterization == "x0"
            pred = torch.clamp(pred, *self.clamp_pred_in_reverse)
        return super().construct_score(t=t, x_t=x_t, pred=pred)

    @property
    def parameterization(self):
        return "x0"
