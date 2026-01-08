import os
from pathlib import Path
from typing import Optional, Sequence

import fire
import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from torch_geometric.data import DataLoader

from src_kldm import utils
from src_kldm.data.dataset import SampleDatasetCSP
from src_kldm.data.transforms import FullyConnectedGraph
from src_kldm.data.utils import save_json
from src_kldm.lit.module import LitKLDM
from src_kldm.utils import get_next_version, load_cfg
from src_kldm.utils.callback import LogSampledAtomsCallback

log = utils.get_pylogger(__name__)


def get_sample_loader(
        formulas: Sequence[str], n_samples_per_formula: int, batch_size: int
):
    dataset = SampleDatasetCSP(
        formulas=formulas,
        n_samples_per_formula=n_samples_per_formula,
        transform=FullyConnectedGraph(),
    )
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )


def instantiate_model(cfg: DictConfig, checkpoint_path: str) -> LitKLDM:
    if cfg.get("seed"):
        pl.seed_everything(cfg.get("seed"), workers=True)

    print(f"Instantiating lit_module <{cfg.lit_module.get('_target_')}>")
    lit_module: LitKLDM = hydra.utils.instantiate(cfg.lit_module)
    print(f"Loading checkpoint from <{cfg.lit_module.get('_target_')}>")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    lit_module.load_state_dict(state_dict=ckpt["state_dict"])
    return lit_module


def generate(
        checkpoint_path: str | Path,
        formulas: Sequence[str],
        n_samples_per_formula: int = 1,
        seed: int = 42,
        sampling_kwargs: Optional[dict] = None,
):
    job_dir, cfg = load_cfg(checkpoint=checkpoint_path)
    eval_root_dir = os.path.join(job_dir, "gen")
    v = get_next_version(eval_root_dir)
    gen_dir = os.path.join(eval_root_dir, f"version_{v}")
    os.makedirs(gen_dir, exist_ok=True)

    save_json(
        json_dict=dict(
            checkpoint_path=checkpoint_path,
            seed=seed,
            formulas=formulas,
            n_samples_per_formula=n_samples_per_formula,
            sampling_kwargs=sampling_kwargs,
        ),
        json_path=os.path.join(gen_dir, "cmd_args.json"),
    )

    print("Instantiating callbacks...")
    callbacks = [
        LogSampledAtomsCallback(
            dirpath=gen_dir,
            save_atoms=True,
            num_log_wandb=0,
            log_wandb_gt=False,
            prefix_with_epoch=False,
        ),
    ]

    lit_module = instantiate_model(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
    )

    print("Overriding sampling parameters with provided ones.")
    print(sampling_kwargs)
    print("--------")
    if lit_module.hparams.sampling_kwargs is None:
        lit_module.hparams.sampling_kwargs = {"val": {}, "test": {}, "predict": {}}
    if "predict" not in lit_module.hparams.sampling_kwargs:
        lit_module.hparams.sampling_kwargs["predict"] = {}
    lit_module.hparams.sampling_kwargs["predict"].update(**sampling_kwargs)

    print(f"Instantiating trainer <{cfg.trainer.get('_target_')}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks)

    print("Creating data for provided formulas...")
    loader = get_sample_loader(
        formulas=formulas,
        n_samples_per_formula=n_samples_per_formula,
        batch_size=cfg.datamodule.test_batch_size,
    )

    print(f"Starting generation for seed: {seed}")
    pl.seed_everything(seed, workers=True)

    trainer.predict(model=lit_module, dataloaders=loader, return_predictions=False)

    print(f"Done. Check '{gen_dir}' for the generated structures.")


def main():
    fire.Fire(generate)


if __name__ == "__main__":
    main()
