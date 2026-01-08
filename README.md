# Kinetic Langevin Diffusion for Crystalline Materials Generation

This repository contains the implementation accompanying "Kinetic Langevin Diffusion for Crystalline Materials
Generation" (ICML
2025).

## Installation

```
# clone this repo
git clone https://github.com/frcnt/kldm.git

# move to the root directory
cd kldm/

# create an environment
pip install uv
uv venv .venv --python 3.11
source .venv/bin/activate

# install requirements
uv pip install -e .
```

## Getting started with CSP on MP-20

### Pre-processing the data

To preprocess `MP-20` with the usual data splits, the following command can be run.

It should take a couple of minutes approximately.

```
export CSV_FOLDER="data/mp_20" 

kldm-preprocess --csv_folder $CSV_FOLDER
```

### Training KLDM

By default, the example config expects the env variables `DATA_PATH` and `LOG_PATH` to be defined.

```
export DATA_PATH="data/"
export LOG_PATH="path/to/where/to/save/logs-and-checkpoints"

export CONFIG_NAME="train_csp_mp_20"

kldm-train -cn $CONFIG_NAME
```

### Evaluating KLDM

A trained CSP model can be evaluated by running a command similar to,

```
export CKPT_PATH="path/to/file.ckpt"

kldm-evaluate-csp \
--ckpt_path $CKPT_PATH \
--n 10 \
--sampling_kwargs "{'force_ema': True, 'method': 'pc', 'n_steps': 1000, 'tf': 0.0, 'correct_pos': True}"
```

In addition to evaluating metrics, the script saves samples in a directory called `eval`
at the root of `CKPT_PATH`.

### Performing CSP on user-specified formula

```
export CKPT_PATH="path/to/file.ckpt"

kldm-generate-csp \
--ckpt_path $CKPT_PATH \
--formulas "[LiFePO4, Li3Co3O6]" \
--n_samples_per_formula 5 \
--sampling_kwargs "{'force_ema': True, 'method': 'pc', 'n_steps': 1000, 'correct_pos': True}"
```

The script saves samples in a directory called `gen` at the root of `CKPT_PATH`.

## Citation

If you find this work useful, please consider citing our paper:

```

@inproceedings{
cornet2025kinetic,
title={Kinetic Langevin Diffusion for Crystalline Materials Generation},
author={François Cornet and Federico Bergamin and Arghya Bhowmik and Juan Maria Garcia-Lastra and Jes Frellsen and
Mikkel N. Schmidt},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=7J1kwZY72h}
}

```