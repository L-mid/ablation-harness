#!/usr/bin/env python3
"""
Plot loss curves from JSONL/CSV logs produced by scripts/ablate.py.

Supports:
- JSONL lines shaped like: {"cfg": {...}, "out": {...}, "_i": <step>}
- CSV rows with columns like: _i, val/loss, train/loss, ...

Usage:
    python -m ablation_harness.plot_loss runs/cifar_tiny/results.jsonl --metrics val/loss --out runs/cifar_tiny/loss

"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot loss curves from logs.")
    p.add_argument("inputs", nargs="+", help="One or more JSONL/CSV log paths.")
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output PNG path or directory. Default: next to first input as loss.png",
    )
    p.add_argument("--xkey", type=str, default="_i", help="Key/column for x-axis (default: _i).")
    p.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Metrics to plot (e.g., val/loss train/loss). Auto-detects if omitted.",
    )
    p.add_argument("--labels", nargs="*", default=None, help="Legend labels (must match number of inputs).")
    p.add_argument(
        "--ema",
        type=float,
        default=0.0,
        help="EMA smoothing factor in [0,1]. 0.0 disables smoothing.",
    )
    p.add_argument(
        "--every",
        type=int,
        default=1,
        help="Downsample: take every Nth point (default: 1 = no downsample).",
    )
    p.add_argument("--title", type=str, default="Loss curves", help="Plot title.")
    p.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        metavar=("YMIN", "YMAX"),
        help="Fix y-axis limits. If omitted, the plot auto-zooms when the y-range is tiny.",
    )
    p.add_argument(
        "--summary",
        action="store_true",
        default=True,
        help="Print a short data summary per plotted series (default: on).",
    )
    p.add_argument("--no-summary", dest="summary", action="store_false")

    return p.parse_args()


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[plot_loss.py: WARN] {path}:{ln} JSON decode error: {e}", file=sys.stderr)
    return rows


def load_csv(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def collect_series_from_jsonl(rows: List[Dict], xkey: str, metrics=None) -> Tuple[List[float], Dict[str, List[float]]]:
    # Find available metrics if not provided
    available = set()
    for r in rows:
        out = r.get("out", {})
        if isinstance(out, dict):
            available.update(out.keys())

    if metrics is None:
        # Auto-pick anything ending with "/loss" or named "loss"
        metrics = sorted([k for k in available if k.endswith("/loss") or k == "loss"])
        if not metrics:
            print(
                "[plot_loss.py: WARN] No loss-like metrics found; specify --metrics.",
                file=sys.stderr,
            )

    # Build x and series
    x = []
    series = {m: [] for m in metrics}
    for r in rows:
        if xkey not in r:
            # Some logs might put xkey at top-level only; skip if missing.
            continue
        x.append(float(r[xkey]))
        out = r.get("out", {})
        for m in metrics:
            v = out.get(m, None)
            series[m].append(float(v) if v is not None else float("nan"))

    return x, series


def collect_series_from_csv(rows: List[Dict], xkey: str, metrics=None) -> Tuple[List[float], Dict[str, List[float]]]:
    # Determine columns
    columns = set(rows[0].keys()) if rows else set()
    if xkey not in columns:
        raise KeyError(f"xkey '{xkey}' not found in CSV columns: {sorted(columns)}")

    if metrics is None:
        metrics = sorted([c for c in columns if c.endswith("/loss") or c == "loss"])
        if not metrics:
            print(
                "[plot_loss.py: WARN] No loss-like columns found; specify --metrics.",
                file=sys.stderr,
            )

    x = []
    series = {m: [] for m in metrics}
    for r in rows:
        x.append(float(r[xkey]))
        for m in metrics:
            val = r.get(m, None)
            series[m].append(float(val) if val not in (None, "", "nan") else float("nan"))

    return x, series


def ema_smooth(values: List[float], alpha: float) -> List[float]:
    if not values or alpha <= 0.0:
        return values
    smoothed = []
    prev = None
    for v in values:
        if prev is None:
            prev = v
        else:
            prev = alpha * v + (1.0 - alpha) * prev
        smoothed.append(prev)
    return smoothed


def downsample(x: List[float], ys: Dict[str, List[float]], every: int) -> Tuple[List[float], Dict[str, List[float]]]:
    if every <= 1:
        return x, ys
    x2 = x[::every]
    ys2 = {k: v[::every] for k, v in ys.items()}
    return x2, ys2


def main():  # noqa: C901
    args = parse_args()

    inputs = [Path(p) for p in args.inputs]
    for p in inputs:
        if not p.exists():
            print(f"[plot_loss.py: ERROR] Not found: {p}", file=sys.stderr)
            return 2

    if args.labels and len(args.labels) != len(inputs):
        print("[plot_loss.py: ERROR] --labels count must match number of inputs.", file=sys.stderr)
        return 2

    # Prepare output path
    if args.out is None:
        # default next to first input
        default = inputs[0].with_name("loss.png")
        out_path = default
    else:
        out_base = Path(args.out)
        known_file_suffixes = {".png", ".jpg", ".jpeg", ".pdf"}
        if str(args.out).endswith(("/", "\\")):
            out_path = out_base / "loss.png"
        elif out_base.suffix.lower() in known_file_suffixes:
            out_path = out_base
        else:
            # no suffix → assume directory
            out_path = out_base / "loss.png"

    plt.figure()
    plt.title(args.title)
    plt.xlabel(args.xkey)
    plt.ylabel("value")

    any_series = False
    all_y_vals = []  # <-- accumulate all finite y for auto-zoom
    summaries = []  # <-- per-series summaries for printing

    for idx, path in enumerate(inputs):
        label_base = args.labels[idx] if args.labels else path.stem

        if path.suffix.lower() == ".jsonl":
            rows = load_jsonl(path)
            x, series = collect_series_from_jsonl(rows, xkey=args.xkey, metrics=args.metrics)
        elif path.suffix.lower() == ".csv":
            rows = load_csv(path)
            x, series = collect_series_from_csv(rows, xkey=args.xkey, metrics=args.metrics)
        else:
            print(f"[plot_loss.py: WARN] Skipping unsupported file type: {path}", file=sys.stderr)
            continue

        if not x or not series:
            print(f"[plot_loss.py: WARN] No data extracted from {path}", file=sys.stderr)
            raise RuntimeError("No data extracted: not usually intended (remove error if was).")
            continue

        # Downsample and smooth
        x, series = downsample(x, series, args.every)
        for m, y in series.items():
            y_plot = ema_smooth(y, args.ema) if args.ema > 0 else y

            finite_y = [v for v in y_plot if isinstance(v, (int, float)) and math.isfinite(v)]
            all_y_vals.extend(finite_y)

            if finite_y:
                y_min, y_max = min(finite_y), max(finite_y)
                summaries.append(
                    {
                        "label": f"{label_base}:{m}",
                        "count": len(finite_y),
                        "y_min": y_min,
                        "y_max": y_max,
                        "y_range": y_max - y_min,
                    }
                )
            else:
                summaries.append(
                    {
                        "label": f"{label_base}:{m}",
                        "count": 0,
                        "y_min": None,
                        "y_max": None,
                        "y_range": None,
                    }
                )

            plt.plot(x, y_plot, label=f"{label_base}:{m}")
            any_series = True

        if not any_series:
            print(
                "[plot_loss.py: ERROR] Nothing to plot. " "Check --metrics and input format.",
                file=sys.stderr,
            )
            return 2

        if args.ylim:
            plt.ylim(args.ylim[0], args.ylim[1])
        else:
            # Auto-zoom if everything is nearly flat
            finite_all = [v for v in all_y_vals if isinstance(v, (int, float)) and math.isfinite(v)]
            if len(finite_all) >= 2:
                gmin, gmax = min(finite_all), max(finite_all)
                if (gmax - gmin) < 1e-2:  # very tight range
                    span = gmax - gmin
                    pad = max(1e-4, 0.1 * span)  # 10% of span, not 10% of the level
                    plt.ylim(gmin - pad, gmax + pad)

        # Optional summary
        if args.summary and summaries:
            for s in summaries:
                if s["count"] > 0:
                    print(f"[plot_loss.py: INFO] {s['label']} points={s['count']}, " f"y=[{s['y_min']:.6f}..{s['y_max']:.6f}], delta={s['y_range']:.6f}")
                else:
                    print(f"[plot_loss.py: INFO] {s['label']} points=0 (no finite values)")

    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[plot_loss.py: OK] Saved plot -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
