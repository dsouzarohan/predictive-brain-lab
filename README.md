# Predictive Brain Lab

Predictive Brain Lab is a small Python research-code project for exploring predictive processing-inspired toy models before moving to more realistic neural time-series data.

The first milestone is a synthetic signal demo:

- generate a time-series signal
- train simple next-step predictors
- compute prediction error
- visualize a surprise proxy when the signal regime changes

This project is intentionally concrete and educational. It does not claim to model consciousness or biological attention. In this repo, terms like `prediction error`, `surprise proxy`, and `attention-like signal` are used as practical modeling labels for simple experiments.

## Core Idea

Train a model on one signal regime, then expose it to a changed regime. If the model learned the earlier pattern, prediction error should increase after the shift. That error can be used as a simple surprise proxy.

## Project Layout

```text
predictive-brain-lab/
  README.md
  requirements.txt
  notebooks/
  scripts/
  src/
  outputs/
  tests/
```

## Setup

Python `3.10+` is supported.

Conda is the default workflow for this project:

```bash
conda env create -f environment.yml
conda activate predictive-brain-lab
```

If you prefer to install from `requirements.txt` inside an existing Conda env:

```bash
conda create -n predictive-brain-lab python=3.10
conda activate predictive-brain-lab
pip install -r requirements.txt
```

You can also use `venv`, but the repo is now set up with Conda first.

## Run The First Demo

From the repository root:

```bash
conda activate predictive-brain-lab
python scripts/run_surprise_demo.py
```

The script saves the main figure to:

`outputs/figures/surprise_demo.png`

It also saves a JSON summary to:

`outputs/logs/surprise_demo_results.json`

It also prints:

- mean squared error
- mean absolute error
- number of detected surprise events
- the true change point

## Example Commands

Run the default MLP demo:

```bash
python scripts/run_surprise_demo.py
```

Run the GRU version:

```bash
python scripts/run_surprise_demo.py --model gru --epochs 30
```

Run the default baseline:

```bash
python scripts/run_surprise_demo.py --model baseline
```

In single-model mode, `baseline` uses the persistence baseline.

Change the synthetic regime and output figure path:

```bash
python scripts/run_surprise_demo.py \
  --model mlp \
  --window-size 40 \
  --change-point 700 \
  --frequency-before 2.0 \
  --frequency-after 8.0 \
  --noise-std 0.08 \
  --output-path outputs/figures/custom_surprise_demo.png
```

Compare all models on the same signal:

```bash
python scripts/run_surprise_demo.py --compare --epochs 20 --output-path outputs/figures/model_comparison.png
```

## Run Tests

```bash
pytest
```

## Interpreting The Figure

The surprise demo figure has two panels:

- the top panel shows the signal and the model's aligned next-step predictions
- the bottom panel shows absolute prediction error over time

When training is restricted to the pre-change regime, prediction error should usually rise after the true frequency shift. High-error points above the configured percentile threshold are marked as surprise events. This is a simple surprise proxy, not a claim about biological attention.

## Notebooks

The notebooks are lightweight starting points:

- `01_signal_generation.ipynb`
- `02_baseline_predictor.ipynb`
- `03_gru_predictor.ipynb`
- `04_prediction_error_attention.ipynb`

They are meant to be readable stepping stones rather than a heavy workflow layer.

## Roadmap

- add richer synthetic regime changes
- compare baseline, MLP, and GRU predictors more systematically
- improve evaluation and plotting utilities
- add saved checkpoints and experiment logs
- later introduce neural time-series data handling in a clearly separate phase
