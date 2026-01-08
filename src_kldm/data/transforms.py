import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import dense_to_sparse, one_hot

from src_kldm.data.utils import read_json


@functional_transform("fully_connected_graph")
class FullyConnectedGraph(BaseTransform):
    def __init__(self, key: str = "edge_node_index", len_from: str = "pos") -> None:
        self.key = key
        self.len_from = len_from

    def forward(self, data: Data) -> Data:
        n = len(getattr(data, self.len_from))
        fc_graph = torch.ones(n, n) - torch.eye(n)
        fc_edges, _ = dense_to_sparse(fc_graph)

        setattr(data, self.key, fc_edges)

        return data


@functional_transform("continuous_interval_lengths")
class ContinuousIntervalLengths(BaseTransform):
    """
    This transform is used to transform the lengths of the lattice into a continuous interval
    This is done by taking l --> log(l)


    NOTE: loc and scale should be given in transformed space.
    """

    def __init__(
            self,
            in_key: str = "lengths",
            out_key: Optional[str] = None,
            normalize_by_num_atoms: bool = False,
            lengths_loc_scale: Optional[
                dict[int, tuple[list[float], list[float]]] | str | Path
                ] = None,
    ) -> None:

        if normalize_by_num_atoms:
            assert (
                    lengths_loc_scale is None
            ), "'normalize_by_num_atoms' and 'lengths_loc_scale' cannot be combined"

        self.in_key = in_key
        self.out_key = out_key
        self.normalize_by_num_atoms = normalize_by_num_atoms

        self.lengths_loc_scale = self.maybe_read_from_json(lengths_loc_scale)

    def forward(self, data: Data) -> Data:
        if not hasattr(data, self.in_key):
            raise ValueError(f"Data object must have '{self.in_key}' attribute!")

        value = getattr(data, self.in_key)

        if self.normalize_by_num_atoms:
            n = len(data.pos)
            value = value / (n ** (1 / 3))

        log_value = torch.log(value)

        if self.lengths_loc_scale:
            n = len(data.pos)
            loc, scale = self.lengths_loc_scale[
                n
            ]  # loc, scale are 3-dimensional for a, b, c
            log_value = (log_value - torch.as_tensor(loc)) / torch.as_tensor(scale)

        if self.out_key is None:
            setattr(data, self.in_key, log_value)
        else:
            setattr(data, self.out_key, log_value)

        return data

    def invert_one(self, log_abc: np.ndarray, n: int):
        if self.lengths_loc_scale:
            loc, scale = self.lengths_loc_scale[n]
            log_abc = log_abc * scale + loc

        a, b, c = np.exp(log_abc)

        if self.normalize_by_num_atoms:
            scale = n ** (1 / 3)
            a, b, c = scale * a, scale * b, scale * c

        return a, b, c

    @staticmethod
    def maybe_read_from_json(
            lengths_loc_scale: Optional[
                dict[int, tuple[list[float], list[float]]] | str | Path
                ],
    ):
        if isinstance(lengths_loc_scale, str) or isinstance(lengths_loc_scale, Path):
            print(f"Reading 'lengths_loc_scale' from '{lengths_loc_scale}'...")
            json_dict = read_json(lengths_loc_scale)
            json_dict = {int(k): json_dict[k] for k in json_dict}
            return json_dict
        else:
            return lengths_loc_scale


@functional_transform("continuous_interval_angles")
class ContinuousIntervalAngles(BaseTransform):
    """
    This transform is used to transform the angles of the lattice into a continuous interval
    This is done by taking \phi --> \tan(\phi + \pi/2)

    NOTE: loc and scale should be given in transformed space.
    """

    def __init__(
            self,
            in_key: str = "angles",
            out_key: Optional[str] = None,
            is_deg: bool = True,
            angles_loc_scale: Optional[tuple[float, float]] = None,
    ) -> None:
        self.in_key = in_key
        self.out_key = out_key
        self.is_deg = is_deg
        self.angles_loc_scale = angles_loc_scale

    def forward(self, data: Data) -> Data:
        if not hasattr(data, self.in_key):
            raise ValueError(f"Data object must have '{self.in_key}' attribute!")

        value = getattr(data, self.in_key)

        if self.is_deg:
            value = torch.deg2rad(value)

        new_value = torch.tan(value - torch.pi / 2.0)

        if self.angles_loc_scale:
            loc, scale = self.angles_loc_scale
            new_value = (new_value - loc) / scale

        if self.out_key is None:
            setattr(data, self.in_key, new_value)
        else:
            setattr(data, self.out_key, new_value)

        return data

    def invert_one(self, tan_angles: np.ndarray):

        if self.angles_loc_scale:
            loc, scale = self.angles_loc_scale
            tan_angles = tan_angles * scale + loc

        angles = np.arctan(tan_angles) + (math.pi / 2.0)
        if self.is_deg:
            angles = np.rad2deg(angles)

        alpha, beta, gamma = angles

        return alpha, beta, gamma


@functional_transform("one_hot")
class OneHot(BaseTransform):
    def __init__(
            self,
            values: list[int],
            key: str = "h",
            scale: float = 1.0,
            noise_std: float = 0.0,
            dtype: torch.dtype = torch.get_default_dtype(),
            expand_as_vector: bool = True,
    ) -> None:
        self.mapping = {v: i for (i, v) in enumerate(values)}
        self.key = key
        self.dtype = dtype
        self.noise_std = noise_std
        self.scale = scale
        self.expand_as_vector = expand_as_vector

    def forward(self, data: Data) -> Data:
        data_key = getattr(data, self.key)
        assert data_key.ndim == 1

        x = torch.as_tensor([self.mapping[xi.item()] for xi in data_key])
        if self.expand_as_vector:
            x = self.scale * one_hot(x, num_classes=len(self.mapping)).to(self.dtype)

            if self.noise_std > 0.0:
                x = x + torch.randn_like(x) * self.noise_std

        setattr(data, self.key, x)

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.mapping})"


@functional_transform("concat_features")
class ConcatFeatures(BaseTransform):
    def __init__(self, in_keys: list[str], out_key: str, dim: int = -1) -> None:
        self.in_keys = in_keys
        self.out_key = out_key
        self.dim = dim

    def forward(self, data: Data) -> Data:
        features = [getattr(data, key) for key in self.in_keys]
        concat_features = torch.cat(features, dim=self.dim)
        setattr(data, self.out_key, concat_features)

        return data
