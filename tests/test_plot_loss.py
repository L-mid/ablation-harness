"""
Tests for scripts/plot_loss.py

Covers:
- JSONL plotting via subprocess
- CSV plotting via subprocess (directory output handling)
- Multi-input overlay with labels, EMA, downsampling
- --ylim enforcement + summary printing (import module, no subprocess)
- Auto-zoom behavior for nearly-flat series (import module)

"""

import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from matplotlib import pyplot as plt


def _write_jsonl(p: Path, *, rows=None, ys=None):
    """Write JSONL file. Either pass full rows or just a sequence of ys."""
    p.parent.mkdir(parents=True, exist_ok=True)
    if rows is None:
        assert ys is not None, "Provide rows=... or ys=..."
        rows = [{"cfg": {"seed": 1}, "out": {"val/loss": float(y)}, "_i": i} for i, y in enumerate(ys)]
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _write_csv(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["_i", "val/loss"])
        w.writeheader()
        w.writerow({"_i": 0, "val/loss": 2.0})
        w.writerow({"_i": 1, "val/loss": 1.5})
        w.writerow({"_i": 2, "val/loss": 1.2})


def _run_plot(cmd, cwd=None):
    env = os.environ.copy()
    # Ensure headless matplotlib
    env.setdefault("MPLBACKEND", "Agg")
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, env=env)
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    print("[TEST]:", res)
    return res


def _assert_png(path: Path):
    assert path.exists(), f"Missing output: {path}"
    data = path.read_bytes()
    assert len(data) > 50, "PNG too small?"
    assert data.startswith(b"\x89PNG\r\n\x1a\n"), "Not a PNG signature"


def _import_plotter_from_module(dotted: str):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return importlib.import_module(dotted)


def test_plot_loss_jsonl(tmp_path: Path):
    logs = tmp_path / "metrics.jsonl"
    _write_jsonl(logs, ys=[2.0, 1.5, 1.2])
    out_png = tmp_path / "loss_jsonl.png"

    cmd = [
        sys.executable,
        "-m",
        "ablation_harness.plot_loss",
        str(logs),
        "--metrics",
        "val/loss",
        "--out",
        str(out_png),
    ]
    _run_plot(cmd)
    _assert_png(out_png)


def test_plot_loss_csv(tmp_path: Path):
    logs_csv = tmp_path / "metrics.csv"
    _write_csv(logs_csv)
    out_dir = tmp_path / "plots_dir"  # test --out as a directory
    cmd = [
        sys.executable,
        "-m",
        "ablation_harness.plot_loss",  # should save to plots_dir/loss.png
        str(logs_csv),
        "--out",
        str(out_dir),
    ]
    _run_plot(cmd)
    _assert_png(out_dir / "loss.png")


def test_plot_loss_multiple_inputs_with_labels(tmp_path: Path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    _write_jsonl(a, ys=[2.0, 1.9, 1.8])
    _write_jsonl(b, ys=[2.1, 2.0, 1.95])

    out_png = tmp_path / "overlay.png"

    cmd = [
        sys.executable,
        "-m",
        "ablation_harness.plot_loss",
        str(a),
        str(b),
        "--metrics",
        "val/loss",
        "--labels",
        "runA",
        "runB",
        "--ema",
        "0.2",
        "--every",
        "1",
        "--out",
        str(out_png),
    ]
    _run_plot(cmd)
    _assert_png(out_png)


# ------------------------
# Import based tests (check --ylim and auto-zoom)
# ------------------------
def test_ylim_enforced_and_summary_printed_import(tmp_path: Path, monkeypatch, capsys):
    # Headless + clean figs
    monkeypatch.setenv("MPLBACKEND", "Agg")
    plt.close("all")

    logs = tmp_path / "m.jsonl"
    _write_jsonl(logs, ys=[2.0, 1.5, 1.2])

    txt = logs.read_text(encoding="utf-8")
    lines = [ln for ln in txt.splitlines() if ln.strip()]
    assert len(lines) == 3

    out_png = tmp_path / "ylims.png"

    mod = _import_plotter_from_module("ablation_harness.plot_loss")

    # Simulate CLI
    monkeypatch.setenv("PYTHONWARNINGS", "ignore")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_loss.py",
            str(logs),
            "--metrics",
            "val/loss",
            "--ylim",
            "1.0",
            "3.0",
            "--out",
            str(out_png),
        ],
    )

    rc = mod.main()
    assert rc == 0, "plot_loss.py exited non-zero"
    _assert_png(out_png)

    # Grab last figure's y-lims
    figs = plt.get_fignums()
    assert figs, "No figures found"
    ax = plt.figure(figs[-1]).gca()
    ymin, ymax = ax.get_ylim()
    assert pytest.approx(1.0) == ymin
    assert pytest.approx(3.0) == ymax

    # Summary printed (default on)
    out = capsys.readouterr().out
    assert "m:val/loss" in out
    assert "points=3" in out

    plt.close("all")


def test_auto_zoom_for_flat_series_import(tmp_path: Path, monkeypatch):
    # Headless + clean figs
    monkeypatch.setenv("MPLBACKEND", "Agg")
    plt.close("all")

    ys = [2.3052, 2.3056, 2.3055, 2.3054, 2.3057, 2.3053, 2.3056, 2.3055]
    logs = tmp_path / "flat.jsonl"
    _write_jsonl(logs, ys=ys)
    out_png = tmp_path / "flat.png"

    mod = _import_plotter_from_module("ablation_harness.plot_loss")
    # No --ylim here; rely on auto-zoom.
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_loss.py",
            str(logs),
            "--metrics",
            "val/loss",
            "--out",
            str(out_png),
        ],
    )

    rc = mod.main()
    assert rc == 0
    _assert_png(out_png)

    figs = plt.get_fignums()
    assert figs, "No figures found"
    ax = plt.figure(figs[-1]).gca()
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    assert span < 0.02, f"Auto-zoom not tight enough, got span={span:.6f}"

    plt.close("all")
