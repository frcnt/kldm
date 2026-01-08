import json
from pathlib import Path
from typing import Sequence, Union

import ase.io
import numpy as np


def read_json(json_path: str | Path):
    with open(json_path, encoding="utf-8", mode="r") as fp:
        return json.load(fp)


def save_json(json_dict: dict, json_path: str, sort_keys: bool = False):
    def _fix_dict():
        for key in json_dict:
            if isinstance(json_dict[key], np.ndarray):
                json_dict[key] = json_dict[key].tolist()

    _fix_dict()
    with open(json_path, encoding="utf-8", mode="w") as fp:
        json.dump(json_dict, fp, sort_keys=sort_keys)


def save_images(
        images: Sequence[ase.Atoms], filename: Union[str, Path], fmt: str = "extxyz"
):
    ase.io.write(filename=filename, images=images, format=fmt)
