import os
from pathlib import Path
from typing import Iterable, Literal

import fire
import numpy as np
import pandas as pd
import torch
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from torch_geometric.data import Data
from tqdm import tqdm

from src_kldm.data.utils import save_json


def lattice_params_to_matrix(a, b, c, alpha, beta, gamma):
    """Converts lattice from abc, angles to matrix.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/core/lattice.py#L311
    """

    def abs_cap(val: float, max_abs_val: float = 1.0):
        return max(min(val, max_abs_val), -max_abs_val)

    angles_r = np.radians([alpha, beta, gamma])
    cos_alpha, cos_beta, cos_gamma = np.cos(angles_r)
    sin_alpha, sin_beta, sin_gamma = np.sin(angles_r)

    val = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta)
    # Sometimes rounding errors result in values slightly > 1.
    val = abs_cap(val)
    gamma_star = np.arccos(val)

    vector_a = [a * sin_beta, 0.0, a * cos_beta]
    vector_b = [
        -b * sin_alpha * np.cos(gamma_star),
        b * sin_alpha * np.sin(gamma_star),
        b * cos_alpha,
    ]
    vector_c = [0.0, 0.0, float(c)]
    return np.array([vector_a, vector_b, vector_c])


def process_cif(crystal_str: str) -> dict[str, np.ndarray]:
    crystal = Structure.from_str(crystal_str, fmt="cif")
    crystal = crystal.get_reduced_structure()

    canonical_crystal = Structure(
        lattice=Lattice.from_parameters(*crystal.lattice.parameters),
        species=crystal.species,
        coords=crystal.frac_coords,
        coords_are_cartesian=False,
    )

    frac_coords = canonical_crystal.frac_coords
    atom_types = canonical_crystal.atomic_numbers
    lattice_parameters = canonical_crystal.lattice.parameters
    lengths = lattice_parameters[:3]
    angles = lattice_parameters[3:]

    assert np.allclose(
        canonical_crystal.lattice.matrix, lattice_params_to_matrix(*lengths, *angles)
    )

    atom_types = np.array(atom_types)
    lengths, angles = np.array(lengths), np.array(angles)

    return {"pos": frac_coords, "h": atom_types, "lengths": lengths, "angles": angles}


def preprocess_csv(
        csv_folder: str | Path,
        splits: Iterable[str] = ("train", "val", "test"),
        fmt: Literal["pyg", "numpy"] = "pyg",
        max_atoms: int = -1,
):
    for split in tqdm(splits):
        csv_path = os.path.join(csv_folder, f"{split}.csv")

        if not os.path.exists(csv_path):
            print(f"Did not find {csv_path}, skipping...")
            continue

        df = pd.read_csv(csv_path)
        n_structures = len(df)

        data_lst = []

        loc_scale_dct = dict()

        for i in tqdm(range(n_structures), desc=f"Preprocessing {split}"):
            cif = df.iloc[i]["cif"]
            data = process_cif(cif)

            n_atoms = data["pos"].shape[0]
            if 0 < max_atoms < n_atoms:  # skip
                continue

            # length stats
            if n_atoms not in loc_scale_dct:
                loc_scale_dct[n_atoms] = []

            loc_scale_dct[n_atoms].append(data["lengths"])

            if fmt == "pyg":
                data = Data(
                    pos=torch.Tensor(data["pos"]),
                    h=torch.LongTensor(data["h"]),
                    lengths=torch.Tensor(data["lengths"]).view(1, -1),  # Å
                    angles=torch.Tensor(data["angles"]).view(1, -1),  # degrees
                )
            data_lst.append(data)

        # save data
        save_path = os.path.join(csv_folder, f"{split}.pt")
        torch.save(data_lst, save_path)
        print(f"Preprocessed '{split}' split saved in {save_path}")

        # compute stats
        loc_scale_dct = {
            n: np.log(np.array(loc_scale_dct[n])) for n in loc_scale_dct
        }  # OBS: log

        loc_scale = {}
        for n in loc_scale_dct:
            log_abc = np.sort(loc_scale_dct[n], axis=0)
            idx = int(len(loc_scale_dct[n]) * 0.025)  # compute mean and std, of 95%
            log_abc_idx = log_abc[idx:-idx, :]  # leave extreme values out
            loc, scale = np.mean(log_abc_idx, axis=0), np.std(log_abc_idx, axis=0)

            loc_scale[n] = (loc.tolist(), scale.tolist())

        save_path = os.path.join(csv_folder, f"{split}_loc_scale.json")
        save_json(json_dict=loc_scale, json_path=save_path, sort_keys=True)

        print(f"Precomputed stats for '{split}' split saved in {save_path}")


def main():
    fire.Fire(preprocess_csv)


if __name__ == "__main__":
    main()
