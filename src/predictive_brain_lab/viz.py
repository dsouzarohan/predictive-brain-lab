"""Visualization helpers for signals, predictions, and surprise."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _save_figure(fig: plt.Figure, save_path: str | Path | None) -> None:
    if save_path is None:
        return
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def plot_signal_with_change_point(
    signal: np.ndarray,
    change_point: int,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot a signal with a marked change point."""

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(signal, label="Signal", linewidth=1.5)
    ax.axvline(change_point, color="crimson", linestyle="--", label="Change point")
    ax.set_title("Synthetic Signal With Frequency Shift")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(alpha=0.3)
    _save_figure(fig, save_path)
    return fig


def plot_predictions_and_errors(
    signal: np.ndarray,
    target_indices: np.ndarray,
    predictions: np.ndarray,
    errors: np.ndarray,
    surprise_events: np.ndarray | None = None,
    change_point: int | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot the signal, aligned predictions, and absolute prediction error."""

    target_indices = np.asarray(target_indices)
    predictions = np.asarray(predictions)
    errors = np.asarray(errors)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    signal_ax, error_ax = axes

    signal_ax.plot(signal, label="Actual signal", linewidth=1.5)
    signal_ax.plot(target_indices, predictions, label="Predicted next value", linewidth=1.2, alpha=0.85)
    if change_point is not None:
        signal_ax.axvline(change_point, color="crimson", linestyle="--", label="Change point")
    signal_ax.set_title("Predictions Across A Frequency Shift")
    signal_ax.set_ylabel("Amplitude")
    signal_ax.legend()
    signal_ax.grid(alpha=0.3)

    error_ax.plot(target_indices, errors, color="darkorange", label="Absolute prediction error", linewidth=1.3)
    if surprise_events is not None:
        surprise_events = np.asarray(surprise_events, dtype=bool)
        error_ax.scatter(
            target_indices[surprise_events],
            errors[surprise_events],
            color="crimson",
            s=18,
            label="Surprise events",
            zorder=3,
        )
    if change_point is not None:
        error_ax.axvline(change_point, color="crimson", linestyle="--")
    error_ax.set_xlabel("Time Step")
    error_ax.set_ylabel("Abs. error")
    error_ax.legend()
    error_ax.grid(alpha=0.3)

    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_error_comparison(
    target_indices: np.ndarray,
    error_by_model: dict[str, np.ndarray],
    change_point: int | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot prediction errors from multiple models on the same axes."""

    fig, ax = plt.subplots(figsize=(12, 5))
    target_indices = np.asarray(target_indices)

    for model_name, errors in error_by_model.items():
        ax.plot(target_indices, np.asarray(errors), linewidth=1.3, label=model_name)

    if change_point is not None:
        ax.axvline(change_point, color="black", linestyle="--", linewidth=1.1, label="Change point")

    ax.set_title("Prediction Error Comparison Across Models")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Absolute prediction error")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig
