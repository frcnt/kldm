from typing import Callable, Optional, Sequence

import chemparse as chemparse
import numpy as np
import torch
import torch.utils.data as torchdata
from ase.data import atomic_numbers
from torch_geometric.data import Data
from torch_geometric.io import fs

from src_kldm.metrics.constants import EMPIRICAL_LEN_DISTRIBUTIONS


class Dataset(torchdata.Dataset):
    def __init__(
            self,
            path: str,
            transform: Optional[Callable] = None,
    ) -> None:
        self.transform = transform
        self.data = self.load(path)

    @staticmethod
    def load(path):
        return fs.torch_load(path)

    def __getitem__(self, idx):
        data = self.data[idx]

        if not isinstance(data, Data):  # the data was saved as a dict of numpy arrays
            data = Data(
                pos=torch.Tensor(data["pos"]),
                h=torch.LongTensor(data["h"]),
                lengths=torch.Tensor(data["lengths"]).view(1, -1),
                angles=torch.Tensor(data["angles"]).view(1, -1),
            )

        data = data if self.transform is None else self.transform(data)

        return data

    def __len__(self):
        return len(self.data)


class SampleDatasetDNG(torchdata.Dataset):
    def __init__(
            self,
            empirical_distribution: np.ndarray,
            n_samples: int = 10_000,
            transform: Optional[Callable] = None,
            seed: int = 42,
    ) -> None:
        self.empirical_distribution = empirical_distribution

        rng = np.random.RandomState(seed)
        self.num_atoms = rng.choice(
            len(empirical_distribution), n_samples, p=empirical_distribution
        )

        self.transform = transform

    def __getitem__(self, idx):
        n = self.num_atoms[idx]
        data = Data(
            pos=torch.randn(n, 3),
            h=torch.LongTensor([6] * n),  # NOTE: default to carbon
        )
        data = data if self.transform is None else self.transform(data)

        return data

    def __len__(self):
        return len(self.num_atoms)

    @classmethod
    def from_cmd_args(
            cls,
            data_name: str,
            len_range: Optional[str] = None,
            n_samples: int = 10_000,
            transform: Optional[Callable] = None,
            seed: int = 42,
    ) -> "SampleDatasetDNG":
        if len_range is None:
            empirical_distribution = EMPIRICAL_LEN_DISTRIBUTIONS[data_name]
        else:
            empirical_distribution = cls.uniform_from_range(len_range)

        return cls(
            empirical_distribution=empirical_distribution,
            n_samples=n_samples,
            transform=transform,
            seed=seed,
        )

    @staticmethod
    def uniform_from_range(len_range: str) -> np.ndarray:
        lb, ub = len_range.split("-")
        lb, ub = int(lb), int(ub)

        empirical_distribution = np.zeros(ub + 1)
        empirical_distribution[lb:ub] = 1.0
        empirical_distribution /= np.sum(empirical_distribution)

        return empirical_distribution


class SampleDatasetCSP(torchdata.Dataset):
    def __init__(
            self,
            formulas: Sequence[str],
            n_samples_per_formula: int = 5,
            transform: Optional[Callable] = None,
    ) -> None:
        self.transform = transform
        self.parsed_formulas = [
            chemparse.parse_formula(f)
            for f in formulas
            for _ in range(n_samples_per_formula)
        ]

    def __getitem__(self, idx):
        formula: dict[str, float] = self.parsed_formulas[idx]
        h = torch.LongTensor(
            [atomic_numbers[el] for el in formula for _ in range(int(formula[el]))]
        )

        data = Data(
            pos=torch.randn(len(h), 3),
            h=h,
        )
        data = data if self.transform is None else self.transform(data)

        return data

    def __len__(self):
        return len(self.parsed_formulas)
