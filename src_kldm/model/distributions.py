import math

import torch
import torch.nn as nn

from ..nn.utils import scatter_center, wrap


class DistributionGaussian(nn.Module):
    def __init__(self, dim: int = 3, scale: float = 1.0, zero_cog: bool = True):
        super(DistributionGaussian, self).__init__()
        self.dim = dim
        self.scale = scale
        self.zero_cog = zero_cog

    def sample(self, index: torch.Tensor):
        sample = torch.randn((len(index), self.dim), device=index.device) * self.scale
        if self.zero_cog:
            sample = scatter_center(sample, index=index)

        return sample


def p_wrapped_normal(
        x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, N: int = 10, T: float = 1.0
):
    p_ = 0
    for i in range(-N, N + 1):
        p_ += torch.exp(-((x - mu + T * i) ** 2) / 2 / sigma ** 2)
    return p_


def d_log_p_wrapped_normal(
        x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, N: int = 10, T: float = 1.0
):
    p_ = 0
    for i in range(-N, N + 1):
        p_ += (
                (x - mu + T * i)
                / sigma ** 2
                * torch.exp(-((x - mu + T * i) ** 2) / 2 / sigma ** 2)
        )
    denom = p_wrapped_normal(x, mu, sigma, N, T)

    return p_ / denom


def sigma_norm(sigma: torch.Tensor, T: float = 1.0, N: int = 10, sn: int = 20000):
    sigmas = sigma[None, :].repeat(sn, 1)
    x_sample = sigma * torch.randn_like(sigmas)
    x_sample = wrap(x_sample, x_range=(2.0 * math.pi) / T)
    normal_ = d_log_p_wrapped_normal(
        x_sample, mu=torch.zeros_like(x_sample), sigma=sigma, T=T, N=N
    )
    return (normal_ ** 2).mean(dim=0)
