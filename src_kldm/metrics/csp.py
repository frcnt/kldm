from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure

from .common import validity_structure
from ..utils import safe_divide


class CSPMetrics:
    def __init__(self, stol: float = 0.5, angle_tol: float = 10.0, ltol: float = 0.3):
        self.matcher = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol)

        self.valid = ...
        self.match = ...
        self.rmse = ...

        self.reset()

    def __call__(self, input_s: list[Structure], target_s: list[Structure]):
        return self.update(input_s, target_s)

    def update(self, input_s: list[Structure], target_s: list[Structure]):

        assert len(input_s) == len(target_s)

        for si, st in zip(input_s, target_s):
            v, m = 0, 0
            if si is not None:
                v = validity_structure(si)
                if v:
                    rms = self.matcher.get_rms_dist(si, st)
                    m = int(rms is not None)

                    if rms is not None:
                        self.rmse.append(rms[0])  # NOTE: only for valid and matching

            self.match.append(m)
            self.valid.append(v)

    def summarize(self) -> dict:
        summary = {}

        summary["valid"] = safe_divide(sum(self.valid), len(self.valid))
        summary["match_rate"] = safe_divide(sum(self.match), len(self.match))
        summary["rmse"] = safe_divide(sum(self.rmse), len(self.rmse))

        return summary

    def reset(self):
        self.valid = []
        self.match = []
        self.rmse = []

    @property
    def details(self) -> dict:
        return {"valid": self.valid, "match": self.match, "rmse": self.rmse}
