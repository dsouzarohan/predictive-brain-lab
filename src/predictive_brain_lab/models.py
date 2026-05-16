"""Baseline and neural network predictors."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn


class PersistenceBaseline:
    """Predict the next value as the last observed value in the window."""

    def predict(self, window: np.ndarray | Tensor) -> np.ndarray:
        """Return persistence predictions for one or more windows."""

        window_array = np.asarray(window, dtype=np.float32)
        if window_array.ndim == 1:
            return np.array([window_array[-1]], dtype=np.float32)
        if window_array.ndim == 2:
            return window_array[:, -1:].astype(np.float32)
        raise ValueError("window must have shape [window_size] or [batch, window_size].")


class MovingAverageBaseline:
    """Predict the next value as the mean of the input window."""

    def predict(self, window: np.ndarray | Tensor) -> np.ndarray:
        """Return moving-average predictions for one or more windows."""

        window_array = np.asarray(window, dtype=np.float32)
        if window_array.ndim == 1:
            return np.array([window_array.mean()], dtype=np.float32)
        if window_array.ndim == 2:
            return window_array.mean(axis=1, keepdims=True).astype(np.float32)
        raise ValueError("window must have shape [window_size] or [batch, window_size].")


class MLPPredictor(nn.Module):
    """Small feed-forward next-step predictor."""

    expects_sequence_input = False

    def __init__(self, input_size: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Predict the next value from a batch of windows."""

        return self.network(inputs)


class GRUPredictor(nn.Module):
    """Small GRU next-step predictor."""

    expects_sequence_input = True

    def __init__(self, input_size: int = 1, hidden_size: int = 32, num_layers: int = 1) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.readout = nn.Linear(hidden_size, 1)

    def forward(self, inputs: Tensor) -> Tensor:
        """Predict the next value from a batch of sequences."""

        if inputs.ndim != 3:
            raise ValueError("GRUPredictor expects inputs with shape [batch, window_size, 1].")
        outputs, _ = self.gru(inputs)
        final_hidden = outputs[:, -1, :]
        return self.readout(final_hidden)
