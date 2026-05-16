"""Training and inference utilities."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Adam
from tqdm import tqdm


def _prepare_inputs(inputs: Tensor, model: nn.Module, device: torch.device) -> Tensor:
    inputs = inputs.to(device)
    if getattr(model, "expects_sequence_input", False):
        inputs = inputs.unsqueeze(-1)
    return inputs


def _average_loss(total_loss: float, dataset_size: int) -> float:
    return total_loss / max(dataset_size, 1)


def train_torch_model(
    model: nn.Module,
    train_loader: Iterable,
    val_loader: Iterable,
    epochs: int,
    lr: float,
    device: str | torch.device,
) -> dict[str, list[float]]:
    """Train a PyTorch model with MSE loss and Adam."""

    device_obj = torch.device(device)
    model.to(device_obj)

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": []}

    for _ in tqdm(range(epochs), desc="Training", leave=False):
        model.train()
        train_loss_total = 0.0

        for inputs, targets in train_loader:
            prepared_inputs = _prepare_inputs(inputs, model, device_obj)
            targets = targets.to(device_obj)

            optimizer.zero_grad()
            predictions = model(prepared_inputs)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item() * inputs.size(0)

        history["train_loss"].append(_average_loss(train_loss_total, len(train_loader.dataset)))

        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                prepared_inputs = _prepare_inputs(inputs, model, device_obj)
                targets = targets.to(device_obj)
                predictions = model(prepared_inputs)
                loss = criterion(predictions, targets)
                val_loss_total += loss.item() * inputs.size(0)

        history["val_loss"].append(_average_loss(val_loss_total, len(val_loader.dataset)))

    return history


def predict_torch_model(
    model: nn.Module,
    data_loader: Iterable,
    device: str | torch.device,
    model_type: str = "mlp",
) -> tuple[np.ndarray, np.ndarray]:
    """Run batched predictions for a trained PyTorch model."""

    device_obj = torch.device(device)
    model.to(device_obj)
    model.eval()

    use_sequence_input = model_type.lower() == "gru" or getattr(model, "expects_sequence_input", False)
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device_obj)
            if use_sequence_input:
                inputs = inputs.unsqueeze(-1)

            predictions = model(inputs)
            y_true.append(targets.cpu().numpy())
            y_pred.append(predictions.cpu().numpy())

    true_array = np.concatenate(y_true, axis=0).reshape(-1)
    pred_array = np.concatenate(y_pred, axis=0).reshape(-1)
    return true_array, pred_array


def predict_baseline(model: object, dataset: Iterable) -> tuple[np.ndarray, np.ndarray]:
    """Run predictions for a non-PyTorch baseline model over a dataset."""

    y_true: list[float] = []
    y_pred: list[float] = []

    for inputs, targets in dataset:
        prediction = model.predict(inputs.numpy()).reshape(-1)
        y_true.append(float(targets.item()))
        y_pred.append(float(prediction[0]))

    return np.asarray(y_true, dtype=np.float32), np.asarray(y_pred, dtype=np.float32)
