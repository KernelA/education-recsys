# Comparison of different RecSys methods

## Description

## Requirements

1. Python 3.10 or higher.

## How to run

Install dependencies:
```
pip install -r ./requirements.txt
```

For neural network training:
```
pip install -r ./requirements.torch.txt
```

For development:
```
pip install -r ./requirements.txt -r ./requirements.dev.txt
```

[Setup Kaggle secrets](https://github.com/Kaggle/kaggle-api#api-credentials)

See DAG of all stages:
```
dvc dag
```

Run all stages or select some stages (you can skip stages with hyperparameters optimization):
```
dvc repro -R .
```

**Some scripts use wandb. If you do not have account on WandDB then set up [WANDB_MODE="disabled"](https://docs.wandb.ai/guides/track/environment-variables#optional-environment-variables)**

Open [l2_coll_filtering.ipynb](./l2_coll_filtering.ipynb)

