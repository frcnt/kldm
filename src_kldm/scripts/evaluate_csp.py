import os
from pathlib import Path
from typing import Optional

import fire
import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from src_kldm import utils
from src_kldm.data.utils import read_json, save_json
from src_kldm.lit.module import LitKLDM
from src_kldm.utils.callback import LogMetricsDetails, LogSampledAtomsCallback
from src_kldm.utils.utils import get_next_version, load_cfg

log = utils.get_pylogger(__name__)

SEED_DIRNAME = "seed={}"
METRICS_FNAME = "metrics.json"
METRICS_DETAILS_FNAME = "metrics_details.json"


def instantiate_model_and_dm(
        cfg: DictConfig, checkpoint_path: str, num_test_subset: int = -1
):
    if cfg.get("seed"):
        pl.seed_everything(cfg.get("seed"), workers=True)

    print(f"Instantiating datamodule <{cfg.datamodule.get('_target_')}>")
    cfg.datamodule.num_test_subset = (
        num_test_subset  # Make sure we load everything from the test set
    )
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    print(f"Instantiating lit_module <{cfg.lit_module.get('_target_')}>")
    lit_module: LitKLDM = hydra.utils.instantiate(cfg.lit_module)
    print(f"Loading checkpoint from <{cfg.lit_module.get('_target_')}>")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    lit_module.load_state_dict(state_dict=ckpt["state_dict"])
    return lit_module, datamodule


def evaluate(
        cfg: DictConfig,
        lit_module: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        save_dir: str,
        seed: int,
):
    print("Instantiating callbacks...")
    callbacks = [
        LogSampledAtomsCallback(
            dirpath=save_dir,
            save_atoms=True,
            num_log_wandb=0,
            log_wandb_gt=False,
            prefix_with_epoch=False,
        ),
        LogMetricsDetails(dirpath=save_dir, fname=METRICS_DETAILS_FNAME),
    ]

    print(f"Instantiating trainer <{cfg.trainer.get('_target_')}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks)

    print(f"Starting evaluating for seed: {seed}")
    pl.seed_everything(seed, workers=True)
    metrics_lst = trainer.test(model=lit_module, datamodule=datamodule)
    assert len(metrics_lst) == 1
    metrics = metrics_lst[0]
    metrics_path = os.path.join(save_dir, METRICS_FNAME)

    # average metrics
    save_json(json_dict=metrics, json_path=metrics_path)
    print(f"Metrics saved in '{metrics_path}'.")

    # metrics
    details_metrics = read_json(json_path=os.path.join(save_dir, METRICS_DETAILS_FNAME))

    del trainer
    return metrics, details_metrics


def matching_rate_at_n(details_lst):
    mr = np.array([d["match"] for d in details_lst])
    return np.mean(np.sum(mr, axis=0) > 0)


def rmse_at_n(details_lst):
    rmse = [[] for _ in range(len(details_lst[0]["match"]))]
    for d in details_lst:
        i = 0
        for j, m in enumerate(d["match"]):
            if m:
                rmse[j].append(d["rmse"][i])
                i += 1
    min_rmse = [min(l) for l in rmse if l]

    return np.mean(min_rmse)


def evaluate_csp(
        checkpoint_path: str | Path,
        n: int = 1,
        seed: int = None,
        sampling_kwargs: Optional[dict] = None,
        num_test_subset: int = -1,
):
    if seed:
        assert n == 1, "seed can only be specified for 'n=1'."

    job_dir, cfg = load_cfg(checkpoint=checkpoint_path)
    eval_root_dir = os.path.join(job_dir, "eval")
    v = get_next_version(eval_root_dir)
    eval_dir = os.path.join(eval_root_dir, f"version_{v}")
    os.makedirs(eval_dir, exist_ok=True)

    save_json(
        json_dict=dict(
            checkpoint_path=checkpoint_path,
            n=n,
            seed=seed,
            sampling_kwargs=sampling_kwargs,
        ),
        json_path=os.path.join(eval_dir, "cmd_args.json"),
    )

    lit_module, dm = instantiate_model_and_dm(
        cfg=cfg, checkpoint_path=checkpoint_path, num_test_subset=num_test_subset
    )

    print("Overriding sampling parameters with provided ones.")
    print(sampling_kwargs)
    print("--------")
    if lit_module.hparams.sampling_kwargs is None:
        lit_module.hparams.sampling_kwargs = {"val": {}, "test": {}}
    lit_module.hparams.sampling_kwargs["test"].update(**sampling_kwargs)

    metrics, details = None, []

    if seed and n == 1:
        seeds = [seed]
    else:
        seeds = range(n)

    for seed in seeds:
        save_dir = os.path.join(eval_dir, SEED_DIRNAME.format(seed))
        m, details_m = evaluate(
            cfg=cfg, lit_module=lit_module, datamodule=dm, save_dir=save_dir, seed=seed
        )

        # Gather for mean and std
        if metrics is None:
            metrics = {k: [m[k]] for k in m}
        else:
            for k in m:
                metrics[k].append(m[k])

        # details
        details.append(details_m)

    # Summary
    summary = {}

    for k in metrics:
        summary[f"{k}@1"] = {"mean": np.mean(metrics[k]), "std": np.std(metrics[k])}

    summary[f"match_rate@{n}"] = matching_rate_at_n(details_lst=details)
    summary[f"rmse@{n}"] = rmse_at_n(details_lst=details)

    save_json(summary, json_path=os.path.join(eval_dir, "summary_metrics.json"))

    print(summary)


def main():
    fire.Fire(evaluate_csp)


if __name__ == "__main__":
    main()
