from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from predictive_brain_lab.data import generate_frequency_shift_signal, generate_sine_wave
from predictive_brain_lab.datasets import SlidingWindowDataset


def test_generate_sine_wave_shape() -> None:
    signal = generate_sine_wave(
        n_steps=200,
        frequency=2.0,
        amplitude=1.0,
        noise_std=0.0,
        sampling_rate=100.0,
        seed=3,
    )
    assert signal.shape == (200,)


def test_frequency_shift_metadata_contains_change_point() -> None:
    signal, metadata = generate_frequency_shift_signal(
        n_steps=300,
        frequency_before=2.0,
        frequency_after=5.0,
        change_point=180,
        seed=11,
    )
    assert signal.shape == (300,)
    assert metadata["change_point"] == 180
    assert "description" in metadata


def test_sliding_window_dataset_length_is_correct() -> None:
    signal = np.arange(20, dtype=np.float32)
    dataset = SlidingWindowDataset(signal=signal, window_size=5)
    assert len(dataset) == 15
