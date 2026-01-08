from abc import ABC
from typing import Any, Optional

import torch
import torch.nn as nn


class BaseDiffusion(ABC, nn.Module):

    def loss_diffusion(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            t: torch.Tensor,
            **kwargs: Optional[Any],
    ):
        raise NotImplementedError

    def training_targets(
            self, t: torch.Tensor, x: torch.Tensor, index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @torch.inference_mode()
    def reverse_step(
            self,
            t: torch.Tensor,
            x_t: torch.Tensor,
            pred: torch.Tensor,
            dt: torch.Tensor,
            **_,
    ):
        raise NotImplementedError

    @torch.inference_mode()
    def sample_prior(self, index: torch.Tensor):
        raise NotImplementedError
