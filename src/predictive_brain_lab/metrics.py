"""Evaluation helpers for prediction and surprise analysis."""

from __future__ import annotations

import numpy as np


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean squared error."""

    true_array = np.asarray(y_true, dtype=float)
    pred_array = np.asarray(y_pred, dtype=float)
    return float(np.mean((true_array - pred_array) ** 2))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean absolute error."""

    true_array = np.asarray(y_true, dtype=float)
    pred_array = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(true_array - pred_array)))


def absolute_prediction_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute elementwise absolute prediction error."""

    true_array = np.asarray(y_true, dtype=float)
    pred_array = np.asarray(y_pred, dtype=float)
    return np.abs(true_array - pred_array)


def detect_surprise_events(errors: np.ndarray, percentile: float = 95) -> np.ndarray:
    """Return a boolean mask for high-error surprise events."""

    error_array = np.asarray(errors, dtype=float)
    if error_array.size == 0:
        return np.asarray([], dtype=bool)
    threshold = np.percentile(error_array, percentile)
    return error_array >= threshold
