import torch
from torch_scatter import scatter_mean


def scatter_center(pos, index):
    return pos - scatter_mean(pos, index=index, dim=0)[index]


def wrap(x, x_range: float = (2.0 * torch.pi)):
    return torch.arctan2(torch.sin(x_range * x), torch.cos(x_range * x)) / x_range
