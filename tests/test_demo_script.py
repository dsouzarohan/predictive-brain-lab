from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_demo_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "scripts" / "run_surprise_demo.py"
    spec = importlib.util.spec_from_file_location("run_surprise_demo", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_demo_parser_accepts_requested_arguments() -> None:
    demo = _load_demo_module()
    parser = demo.build_parser()

    args = parser.parse_args(
        [
            "--model",
            "gru",
            "--epochs",
            "5",
            "--window-size",
            "30",
            "--change-point",
            "400",
            "--frequency-before",
            "2.5",
            "--frequency-after",
            "7.0",
            "--noise-std",
            "0.1",
            "--output-path",
            "outputs/figures/test.png",
        ]
    )

    assert args.model == "gru"
    assert args.epochs == 5
    assert args.window_size == 30
    assert args.change_point == 400
    assert args.frequency_before == 2.5
    assert args.frequency_after == 7.0
    assert args.noise_std == 0.1
    assert args.output_path == Path("outputs/figures/test.png")


def test_save_results_summary_writes_json(tmp_path: Path) -> None:
    demo = _load_demo_module()
    output_path = tmp_path / "surprise_demo_results.json"

    demo.save_results_summary({"mode": "single_model", "results": {"mse": 0.1}}, output_path=output_path)

    assert output_path.exists()
    assert '"mode": "single_model"' in output_path.read_text(encoding="utf-8")
