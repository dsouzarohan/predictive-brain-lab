"""Synthetic signal generation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def _validate_n_steps(n_steps: int) -> None:
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")


def generate_sine_wave(
    n_steps: int,
    frequency: float,
    amplitude: float,
    noise_std: float,
    sampling_rate: float,
    phase: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a noisy sine wave."""

    _validate_n_steps(n_steps)
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive.")

    rng = np.random.default_rng(seed)
    time = np.arange(n_steps, dtype=float) / sampling_rate
    signal = amplitude * np.sin((2.0 * np.pi * frequency * time) + phase)
    noise = rng.normal(loc=0.0, scale=noise_std, size=n_steps)
    return signal + noise


def generate_frequency_shift_signal(
    n_steps: int,
    frequency_before: float,
    frequency_after: float,
    change_point: int,
    amplitude: float = 1.0,
    noise_std: float = 0.05,
    sampling_rate: float = 100.0,
    seed: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate a sine signal with a frequency shift at a known change point."""

    _validate_n_steps(n_steps)
    if not 0 < change_point < n_steps:
        raise ValueError("change_point must be between 0 and n_steps.")
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive.")

    rng = np.random.default_rng(seed)
    time = np.arange(n_steps, dtype=float) / sampling_rate

    signal = np.empty(n_steps, dtype=float)

    before_time = time[:change_point]
    signal[:change_point] = amplitude * np.sin(2.0 * np.pi * frequency_before * before_time)

    after_time = time[change_point:] - time[change_point]
    phase_offset = 2.0 * np.pi * frequency_before * time[change_point]
    signal[change_point:] = amplitude * np.sin(phase_offset + (2.0 * np.pi * frequency_after * after_time))

    noise = rng.normal(loc=0.0, scale=noise_std, size=n_steps)
    signal = signal + noise

    metadata = {
        "change_point": change_point,
        "description": (
            f"Frequency shift from {frequency_before:.3f} Hz to {frequency_after:.3f} Hz "
            f"at sample {change_point}."
        ),
    }
    return signal, metadata
