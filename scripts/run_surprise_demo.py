"""Run the first surprise proxy demo from the command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predictive_brain_lab.data import generate_frequency_shift_signal
from predictive_brain_lab.datasets import SlidingWindowDataset
from predictive_brain_lab.metrics import (
    absolute_prediction_error,
    detect_surprise_events,
    mean_absolute_error,
    mean_squared_error,
)
from predictive_brain_lab.models import (
    GRUPredictor,
    MLPPredictor,
    MovingAverageBaseline,
    PersistenceBaseline,
)
from predictive_brain_lab.train import predict_baseline, predict_torch_model, train_torch_model
from predictive_brain_lab.viz import plot_error_comparison, plot_predictions_and_errors

DEFAULT_FIGURE_PATH = PROJECT_ROOT / "outputs" / "figures" / "surprise_demo.png"
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "outputs" / "logs" / "surprise_demo_results.json"


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the demo."""

    parser = argparse.ArgumentParser(description="Run a frequency-shift surprise demo.")
    parser.add_argument("--model", "--model-type", dest="model", choices=["mlp", "gru", "baseline"], default="mlp")
    parser.add_argument("--compare", action="store_true", help="Run all models and compare prediction errors.")
    parser.add_argument("--n-steps", type=int, default=1200)
    parser.add_argument("--sampling-rate", type=float, default=100.0)
    parser.add_argument("--frequency-before", type=float, default=2.0)
    parser.add_argument("--frequency-after", type=float, default=6.0)
    parser.add_argument("--change-point", type=int, default=800)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--surprise-percentile", type=float, default=95.0)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_FIGURE_PATH)
    return parser


def split_pre_change_indices(dataset: SlidingWindowDataset, change_point: int) -> tuple[list[int], list[int]]:
    """Split pre-change windows into train and validation subsets."""

    pre_change_mask = dataset.target_indices < change_point
    pre_change_indices = dataset.target_indices[pre_change_mask]
    dataset_indices = pre_change_mask.nonzero()[0].tolist()

    if len(pre_change_indices) < 10:
        raise ValueError("Not enough pre-change samples for a train/validation split.")

    split_index = max(1, int(0.8 * len(dataset_indices)))
    split_index = min(split_index, len(dataset_indices) - 1)
    train_indices = dataset_indices[:split_index]
    val_indices = dataset_indices[split_index:]
    return train_indices, val_indices


def build_model(model_type: str, window_size: int, hidden_size: int) -> torch.nn.Module:
    """Construct the requested predictor."""

    if model_type == "gru":
        return GRUPredictor(hidden_size=hidden_size)
    return MLPPredictor(input_size=window_size, hidden_size=hidden_size)


def save_results_summary(results: dict[str, Any], output_path: Path = DEFAULT_RESULTS_PATH) -> None:
    """Save a JSON summary for the demo run."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


def build_run_config(args: argparse.Namespace, output_path: Path) -> dict[str, Any]:
    """Build the config block stored in the results JSON."""

    return {
        "model": args.model,
        "n_steps": args.n_steps,
        "sampling_rate": args.sampling_rate,
        "frequency_before": args.frequency_before,
        "frequency_after": args.frequency_after,
        "change_point": args.change_point,
        "window_size": args.window_size,
        "noise_std": args.noise_std,
        "epochs": args.epochs,
        "output_path": str(output_path),
    }


def summarize_run(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    surprise_percentile: float,
    change_point: int,
    history: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    """Build a compact metric summary for one model run."""

    errors = absolute_prediction_error(y_true, y_pred)
    surprise_events = detect_surprise_events(errors, percentile=surprise_percentile)

    summary: dict[str, Any] = {
        "model": model_name,
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "num_surprise_events": int(surprise_events.sum()),
        "true_change_point": int(change_point),
    }

    if history is not None:
        summary["final_train_loss"] = float(history["train_loss"][-1])
        summary["final_val_loss"] = float(history["val_loss"][-1])

    return {
        "summary": summary,
        "errors": errors,
        "surprise_events": surprise_events,
    }


def run_baseline(dataset: SlidingWindowDataset, surprise_percentile: float, change_point: int) -> dict[str, Any]:
    """Run the default baseline for single-model mode."""

    y_true, y_pred = predict_baseline(PersistenceBaseline(), dataset)
    run_data = summarize_run(
        model_name="persistence_baseline",
        y_true=y_true,
        y_pred=y_pred,
        surprise_percentile=surprise_percentile,
        change_point=change_point,
    )
    run_data["predictions"] = y_pred
    return run_data


def run_torch_predictor(
    model_name: str,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    full_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> tuple[dict[str, list[float]], Any, Any]:
    """Train a torch predictor and return history and predictions."""

    history = train_torch_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        device=device,
    )
    y_true, y_pred = predict_torch_model(
        model=model,
        data_loader=full_loader,
        device=device,
        model_type=model_name,
    )
    return history, y_true, y_pred


def run_single_model(
    args: argparse.Namespace,
    dataset: SlidingWindowDataset,
    train_loader: DataLoader,
    val_loader: DataLoader,
    full_loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """Run one model and return results needed for plotting and logging."""

    if args.model == "baseline":
        return run_baseline(dataset, args.surprise_percentile, args.change_point)

    model = build_model(args.model, args.window_size, args.hidden_size)
    history, y_true, y_pred = run_torch_predictor(
        model_name=args.model,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        full_loader=full_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
    )
    run_data = summarize_run(
        model_name=args.model,
        y_true=y_true,
        y_pred=y_pred,
        surprise_percentile=args.surprise_percentile,
        change_point=args.change_point,
        history=history,
    )
    run_data["predictions"] = y_pred
    return run_data


def run_comparison_mode(
    args: argparse.Namespace,
    dataset: SlidingWindowDataset,
    train_loader: DataLoader,
    val_loader: DataLoader,
    full_loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """Run all models and compare absolute prediction errors."""

    results: dict[str, Any] = {}

    baseline_models = {
        "persistence_baseline": PersistenceBaseline(),
        "moving_average_baseline": MovingAverageBaseline(),
    }
    for name, baseline_model in baseline_models.items():
        y_true, y_pred = predict_baseline(baseline_model, dataset)
        results[name] = summarize_run(
            model_name=name,
            y_true=y_true,
            y_pred=y_pred,
            surprise_percentile=args.surprise_percentile,
            change_point=args.change_point,
        )

    for model_name in ("mlp", "gru"):
        model = build_model(model_name, args.window_size, args.hidden_size)
        history, y_true, y_pred = run_torch_predictor(
            model_name=model_name,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            full_loader=full_loader,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
        )
        results[model_name] = summarize_run(
            model_name=model_name,
            y_true=y_true,
            y_pred=y_pred,
            surprise_percentile=args.surprise_percentile,
            change_point=args.change_point,
            history=history,
        )

    return results


def print_single_model_summary(summary: dict[str, Any], output_path: Path) -> None:
    """Print a concise summary for a single-model run."""

    print(f"Model: {summary['model']}")
    if "final_train_loss" in summary:
        print(f"Final train loss: {summary['final_train_loss']:.6f}")
        print(f"Final val loss: {summary['final_val_loss']:.6f}")
    print(f"MSE: {summary['mse']:.6f}")
    print(f"MAE: {summary['mae']:.6f}")
    print(f"Number of surprise events: {summary['num_surprise_events']}")
    print(f"True change point: {summary['true_change_point']}")
    print(f"Saved figure: {output_path}")
    print(f"Saved results: {DEFAULT_RESULTS_PATH}")


def print_comparison_summary(results: dict[str, Any], output_path: Path) -> None:
    """Print a concise comparison summary."""

    print("Comparison mode:")
    for model_name, model_results in results.items():
        summary = model_results["summary"]
        print(
            f"{model_name}: MSE={summary['mse']:.6f}, "
            f"MAE={summary['mae']:.6f}, "
            f"surprise_events={summary['num_surprise_events']}"
        )
    print(f"True change point: {next(iter(results.values()))['summary']['true_change_point']}")
    print(f"Saved figure: {output_path}")
    print(f"Saved results: {DEFAULT_RESULTS_PATH}")


def main() -> None:
    """Run the synthetic surprise demo."""

    args = build_parser().parse_args()

    signal, metadata = generate_frequency_shift_signal(
        n_steps=args.n_steps,
        frequency_before=args.frequency_before,
        frequency_after=args.frequency_after,
        change_point=args.change_point,
        amplitude=1.0,
        noise_std=args.noise_std,
        sampling_rate=args.sampling_rate,
        seed=args.seed,
    )

    dataset = SlidingWindowDataset(signal=signal, window_size=args.window_size)
    train_indices, val_indices = split_pre_change_indices(dataset, args.change_point)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    full_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = args.output_path

    if args.compare:
        comparison_results = run_comparison_mode(
            args=args,
            dataset=dataset,
            train_loader=train_loader,
            val_loader=val_loader,
            full_loader=full_loader,
            device=device,
        )
        figure = plot_error_comparison(
            target_indices=dataset.target_indices,
            error_by_model={name: data["errors"] for name, data in comparison_results.items()},
            change_point=metadata["change_point"],
            save_path=output_path,
        )
        results_summary = {
            "mode": "comparison",
            "config": build_run_config(args, output_path),
            "models": {name: data["summary"] for name, data in comparison_results.items()},
        }
        save_results_summary(results_summary)
        print_comparison_summary(comparison_results, output_path)
        figure.clf()
        return

    run_results = run_single_model(
        args=args,
        dataset=dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        full_loader=full_loader,
        device=device,
    )
    figure = plot_predictions_and_errors(
        signal=signal,
        target_indices=dataset.target_indices,
        predictions=run_results["predictions"],
        errors=run_results["errors"],
        surprise_events=run_results["surprise_events"],
        change_point=metadata["change_point"],
        save_path=output_path,
    )

    results_summary = {
        "mode": "single_model",
        "config": build_run_config(args, output_path),
        "results": run_results["summary"],
    }
    save_results_summary(results_summary)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print_single_model_summary(run_results["summary"], output_path)
    figure.clf()


if __name__ == "__main__":
    main()
