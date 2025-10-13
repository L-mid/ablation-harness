import json
from pathlib import Path

from ablation_harness.plot_variance import compute_group_stats, plot_variance


def _write_jsonl(path: Path, rows):
    """Writes jsonl. (does not append)."""
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _row(cfg, seed, acc):
    """Defines row with a cfg + out schema."""
    # Matches your schema: metrics live under 'out' with key 'val/acc'
    return {
        "cfg": cfg,
        "out": {"seed": seed, "val/acc": acc},
    }


def test_plot_variance_creates_png(tmp_path: Path):
    """Smokes plot_variance.py and ensures there's a png."""
    results = tmp_path / "results.jsonl"

    # compose mini josnl mock.
    rows = [
        _row({"optimizer": "adam", "lr": 0.001, "ema": False}, 0, 0.42),
        _row({"optimizer": "adam", "lr": 0.001, "ema": False}, 1, 0.44),
        _row({"optimizer": "adam", "lr": 0.001, "ema": False}, 2, 0.41),
        _row({"optimizer": "sgd", "lr": 0.01, "ema": False}, 0, 0.35),
        _row({"optimizer": "sgd", "lr": 0.01, "ema": False}, 1, 0.36),
        _row({"optimizer": "sgd", "lr": 0.01, "ema": False}, 2, 0.34),
    ]
    _write_jsonl(results, rows)

    stats = compute_group_stats(
        rows=list(json.loads(line) for line in results.read_text().splitlines()),
        metric_path="val/acc",  # <- the key inside 'out'
        label_fields=["optimizer", "lr", "ema"],
    )

    out_png = tmp_path / "variance.png"
    plot_variance(
        stats=stats,
        out_path=out_png,
        title="Test variance plot",
        metric_name="val/acc",
    )

    assert out_png.exists(), "PNG was not created"
    assert out_png.stat().st_size > 0, "PNG is empty"


def test_plot_variance_single_seed_no_crash(tmp_path: Path):
    """Ensures plot variance does not fail on n=1. (defaults to std = 0.0)"""
    results = tmp_path / "results_single.jsonl"
    rows = [
        _row({"optimizer": "adam", "lr": 0.001, "ema": True}, 0, 0.43),  # n=1
        _row({"optimizer": "sgd", "lr": 0.003, "ema": False}, 0, 0.37),  # n=1
    ]
    _write_jsonl(results, rows)

    stats = compute_group_stats(
        rows=list(json.loads(line) for line in results.read_text().splitlines()),
        metric_path="val/acc",
        label_fields=["optimizer", "lr", "ema"],
    )

    out_png = tmp_path / "variance_single.png"
    plot_variance(
        stats=stats,
        out_path=out_png,
        title="Single-seed variance (no crash)",
        metric_name="val/acc",
    )

    assert out_png.exists(), "PNG was not created for single-seed case"
    assert out_png.stat().st_size > 0, "PNG is empty in single-seed case"

    # std must be 0.0 when n == 1
    for _, _, std, n in stats:
        if n == 1:
            assert std == 0.0
