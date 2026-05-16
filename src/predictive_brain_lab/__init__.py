"""Utilities for predictive processing-inspired toy signal experiments."""

from .data import generate_frequency_shift_signal, generate_sine_wave
from .datasets import SlidingWindowDataset
from .metrics import (
    absolute_prediction_error,
    detect_surprise_events,
    mean_absolute_error,
    mean_squared_error,
)
from .models import GRUPredictor, MLPPredictor, MovingAverageBaseline, PersistenceBaseline
from .train import predict_baseline, predict_torch_model, train_torch_model
from .viz import plot_error_comparison, plot_predictions_and_errors, plot_signal_with_change_point

__all__ = [
    "GRUPredictor",
    "MLPPredictor",
    "MovingAverageBaseline",
    "PersistenceBaseline",
    "SlidingWindowDataset",
    "absolute_prediction_error",
    "detect_surprise_events",
    "generate_frequency_shift_signal",
    "generate_sine_wave",
    "mean_absolute_error",
    "mean_squared_error",
    "plot_error_comparison",
    "plot_predictions_and_errors",
    "plot_signal_with_change_point",
    "predict_baseline",
    "predict_torch_model",
    "train_torch_model",
]
