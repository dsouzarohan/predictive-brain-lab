"""Dataset utilities for time-series prediction."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class SlidingWindowDataset(Dataset):
    """Create fixed-size windows and next-step targets from a 1D signal."""

    def __init__(self, signal: np.ndarray, window_size: int) -> None:
        signal_array = np.asarray(signal, dtype=np.float32)
        if signal_array.ndim != 1:
            raise ValueError("signal must be one-dimensional.")
        if window_size <= 0:
            raise ValueError("window_size must be positive.")
        if window_size >= len(signal_array):
            raise ValueError("window_size must be smaller than the signal length.")

        self.signal = signal_array
        self.window_size = window_size
        self.target_indices = np.arange(window_size, len(signal_array))

    def __len__(self) -> int:
        return len(self.signal) - self.window_size

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        start = idx
        end = idx + self.window_size
        window = torch.from_numpy(self.signal[start:end])
        target = torch.tensor([self.signal[end]], dtype=torch.float32)
        return window, target
