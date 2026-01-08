import itertools
import warnings
from collections import Counter

import numpy as np
import smact
from ase.data import chemical_symbols
from matminer.featurizers.composition.composite import ElementProperty
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from pymatgen.core import Structure
from scipy.stats import wasserstein_distance
from smact.screening import pauling_test

CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CompFP = ElementProperty.from_preset("magpie")

Percentiles = {
    "mp20": np.array([-3.17562208, -2.82196882, -2.52814761]),
    "carbon": np.array([-154.527093, -154.45865733, -154.44206825]),
    "perovskite": np.array([0.43924842, 0.61202443, 0.7364607]),
}


def validity_structure(structure: Structure, cutoff: float = 0.5):
    try:
        distance_matrix = structure.distance_matrix
    except Exception as e:
        warnings.warn(f"In the structure validity the following error occurred: {e}")
        return False

    # Pad diagonal with a large number
    distance_matrix = distance_matrix + np.diag(
        np.ones(distance_matrix.shape[0]) * (cutoff + 10.0)
    )
    if (
            distance_matrix.min() < cutoff
            or structure.volume < 0.1
            or max(structure.lattice.abc) > 40
    ):
        return False
    else:
        return True


def _validity_composition(
        unique_symbols, count, use_pauling_test=True, include_alloys=True
):
    """
    Taken from DiffCSP.
    """
    space = smact.element_dictionary(unique_symbols)
    smact_elems = [e[1] for e in space.items()]
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = [e.oxidation_states for e in smact_elems]
    if len(set(unique_symbols)) == 1:
        return True
    if include_alloys:
        is_metal_list = [elem_s in smact.metals for elem_s in unique_symbols]
        if all(is_metal_list):
            return True

    threshold = np.max(count)
    compositions = []
    # if len(list(itertools.product(*ox_combos))) > 1e5:
    #     return False
    oxn = 1
    for oxc in ox_combos:
        oxn *= len(oxc)
    if oxn > 1e7:
        return False
    for ox_states in itertools.product(*ox_combos):
        stoichs = [(c,) for c in count]
        # Test for charge balance
        cn_e, cn_r = smact.neutral_ratios(
            ox_states, stoichs=stoichs, threshold=threshold
        )
        # Electronegativity test
        if cn_e:
            if use_pauling_test:
                try:
                    electroneg_OK = pauling_test(ox_states, electronegs)
                except TypeError:
                    # if no electronegativity data, assume it is okay
                    electroneg_OK = True
            else:
                electroneg_OK = True
            if electroneg_OK:
                return True
    return False


def validity_composition(structure: Structure):
    # need to extract composition and count for each
    counter = Counter(structure.atomic_numbers)
    unique_symbols, count = [], []
    try:
        for n, c in counter.items():
            unique_symbols.append(chemical_symbols[n])
            count.append(c)
        return _validity_composition(unique_symbols, count)
    except:
        return False


def fp_composition(structure: Structure):
    comp = structure.composition
    fp_comp = CompFP.featurize(comp)

    return fp_comp


def fp_structural(structure: Structure):
    try:
        fp_struct = np.array(
            [CrystalNNFP.featurize(structure, i) for i in range(len(structure))]
        )
        fp_struct = fp_struct.mean(axis=0)
    except Exception:
        fp_struct = None

    return fp_struct


def wdist_density(input_s: list[Structure], target_s: list[Structure]):
    input_densities = []
    for s in input_s:
        try:
            d = s.density
            input_densities.append(d)
        except Exception:
            warnings.warn(f"Exception while calling density of '{s}'.")
            pass

    gt_densities = [s.density for s in target_s]
    wdist_density = wasserstein_distance(input_densities, gt_densities)
    return wdist_density


def wdist_num_elements(input_s: list[Structure], target_s: list[Structure]):
    pred_num_elements = [len(set(c.structure.species)) for c in input_s]
    gt_num_elements = [len(set(c.structure.species)) for c in target_s]
    wdist_num_elements = wasserstein_distance(pred_num_elements, gt_num_elements)

    return wdist_num_elements
