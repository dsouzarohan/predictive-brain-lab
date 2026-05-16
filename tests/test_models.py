from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from predictive_brain_lab.models import GRUPredictor, MLPPredictor, MovingAverageBaseline, PersistenceBaseline


def test_baseline_predictors_return_expected_shapes() -> None:
    window = np.arange(10, dtype=np.float32)

    persistence_prediction = PersistenceBaseline().predict(window)
    moving_average_prediction = MovingAverageBaseline().predict(window)

    assert persistence_prediction.shape == (1,)
    assert moving_average_prediction.shape == (1,)


def test_mlp_predictor_forward_shape() -> None:
    model = MLPPredictor(input_size=50, hidden_size=32)
    batch = torch.randn(8, 50)
    output = model(batch)
    assert output.shape == (8, 1)


def test_gru_predictor_forward_shape() -> None:
    model = GRUPredictor(hidden_size=16)
    batch = torch.randn(8, 50, 1)
    output = model(batch)
    assert output.shape == (8, 1)
