#!/usr/bin/env python3
"""
Plot loss curves from JSONL/CSV logs produced by scripts/ablate.py.

Supports:
- JSONL lines shaped like: {"cfg": {...}, "out": {...}, "_i": <step>}
- CSV rows with columns like: _i, val/loss, train/loss, ...

Usage:
    python -m ablation_harness.plot_loss runs/.../results.jsonl \
        --metrics train/loss val/loss val/acc --y2 val/acc --out runs/.../loss

    # auto-detect y2 (metrics that look like accuracy):
        python -m ablation_harness.plot_loss runs/.../loss.jsonl --metrics train/loss val/loss val/acc



Current:
    python -m ablation_harness.plot_loss
    runs/tinycnn_tester/with_dropout/loss.jsonl
    runs/tinycnn_tester/without_dropout/loss.jsonl
    --metrics train/loss val/loss val/acc
    --out runs/tinycnn_tester/loss_plots
    --labels with_dropout without_dropout
    --xkey epoch --title "Loss & Acc (100 epochs)"

"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, SupportsFloat, Tuple, Union

import matplotlib.pyplot as plt

FloatLike = Union[int, float, str, SupportsFloat]


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

    # New: second axis controls
    p.add_argument("--y2", nargs="*", default=None, help="Metrics to send to right y-axis (accuracy etc.).")
    p.add_argument("--auto-y2", action="store_true", default=True, help="Auto-detect accuracy-like metrics (default on).")
    p.add_argument("--no-auto-y2", dest="auto_y2", action="store_false")

    return p.parse_args()


# --------------------- IO ----------------------
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


# ---------------------- Extraction + Coalescing ---------------------
def _auto_detect_metrics_from_jsonl(rows: List[Dict]) -> List[str]:
    # pick loss-like by default when --metrics omitted
    available = set()
    for r in rows:
        out = r.get("out", {})
        if isinstance(out, dict):
            available.update(out.keys())
    cand = sorted([k for k in available if k.endswith("/loss") or k == "loss"])
    return cand


def _auto_detect_metrics_from_csv(rows: List[Dict]) -> List[str]:
    cols = set(rows[0].keys()) if rows else set()
    cand = sorted([c for c in cols if c.endswith("/loss") or c == "loss"])
    return cand


def _is_accuracy_like(metric: str) -> bool:
    base = metric.split("/")[-1].lower()
    return metric.endswith("/acc") or base in {"acc", "top1", "top5"}


def _ema(values: List[float], alpha: float) -> List[float]:
    if not values or alpha <= 0.0:
        return values
    out, prev = [], None
    for v in values:
        if prev is None:
            prev = v
        else:
            prev = alpha * v + (1.0 - alpha) * prev
        out.append(prev)
    return out


def _downsample(x: List[float], ys: Dict[str, List[float]], every: int) -> Tuple[List[float], Dict[str, List[float]]]:
    if every <= 1:
        return x, ys
    return x[::every], {k: v[::every] for k, v in ys.items()}


def _get_x(r: dict[str, Any], xkey: str) -> Optional[FloatLike | Any]:
    # direct
    if xkey in r:
        return r[xkey]
    # common nested: out[<xkey>]
    if isinstance(r.get("out"), dict) and xkey in r["out"]:
        return r["out"][xkey]
    # dot-path like: out.epoch (so you can pass --xkey out.epoch)
    if "." in xkey:
        cur = r
        for part in xkey.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return None
            cur = cur[part]
        return cur
    return None


def _coalesce_series_jsonl(rows: List[Dict], xkey: str, metrics: Iterable[str]) -> Dict[str, List[Tuple[float, float]]]:
    """
    Build per-metric series with de-dup by step: last value for (metric, step) wins.
    Looks under r['out'][metric] and uses r[xkey] as the step.
    """
    bucket = {m: {} for m in metrics}
    for r in rows:
        x = _get_x(r, xkey)
        if x is None:
            continue
        try:
            step = float(x)
        except Exception:
            continue

        out = r.get("out", {})
        if not isinstance(out, dict):
            continue
        for m in metrics:
            if m in out and out[m] is not None:
                bucket[m][step] = float(out[m])
    return {m: sorted(d.items(), key=lambda t: t[0]) for m, d in bucket.items()}


def _coalesce_series_csv(rows: List[Dict], xkey: str, metrics: Iterable[str]) -> Dict[str, List[Tuple[float, float]]]:
    bucket = {m: {} for m in metrics}
    for r in rows:
        x = _get_x(r, xkey)
        if x is None:
            continue
        try:
            step = float(x)
        except Exception:
            continue

        out = r.get("out", {})
        if not isinstance(out, dict):
            continue
        for m in metrics:
            if m in out and out[m] is not None:
                bucket[m][step] = float(out[m])
    return {m: sorted(d.items(), key=lambda t: t[0]) for m, d in bucket.items()}


# ------------------- Main ---------------------
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
        out_path = inputs[0].with_name("loss.png")
    else:
        out_base = Path(args.out)
        if str(args.out).endswith(("/", "\\")):
            out_path = out_base / "loss.png"
        elif out_base.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf"}:
            out_path = out_base
        else:
            out_path = out_base / "loss.png"

    fig, ax_left = plt.subplots()
    ax_left.set_title(args.title)
    ax_left.set_xlabel(args.xkey)
    ax_left.set_ylabel("loss / value")

    ax_right = None  # will create lazily if needed

    any_series = False
    all_left_vals: List[float] = []
    summaries = []

    # Determine y2 set
    y2_metrics_user = set(args.y2 or [])
    auto_y2_enabled = args.auto_y2

    # colour
    from itertools import cycle

    # before the file loop
    DEFAULT_COLORS = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"])
    color_cycle = cycle(DEFAULT_COLORS)
    color_for_run = {}  # e.g., {'with_dropout': (r,g,b,a)}

    for idx, path in enumerate(inputs):
        label_base = args.labels[idx] if args.labels else path.stem

        # Load & detect metrics
        if path.suffix.lower() == ".jsonl":
            rows = load_jsonl(path)
            metrics = args.metrics or _auto_detect_metrics_from_jsonl(rows)
            series = _coalesce_series_jsonl(rows, xkey=args.xkey, metrics=metrics)
        elif path.suffix.lower() == ".csv":
            rows = load_csv(path)
            metrics = args.metrics or _auto_detect_metrics_from_csv(rows)
            series = _coalesce_series_csv(rows, xkey=args.xkey, metrics=metrics)
        else:
            print(f"[plot_loss.py: WARN] Skipping unsupported file type: {path}", file=sys.stderr)
            continue

        if not metrics:
            print(f"[plot_loss.py: WARN] No metrics to plot for {path}. Try --metrics.", file=sys.stderr)
            continue

        # Draw each metric series
        for m in metrics:
            pts = series.get(m, [])
            if not pts:
                # still print a summary line to help debugging
                summaries.append({"label": f"{label_base}:{m}", "count": 0, "y_min": None, "y_max": None, "y_range": None})
                continue

            xs, ys = zip(*pts)

            # smoothing / downsample
            xs2, ys2 = list(xs), list(ys)
            if args.every > 1:
                xs2 = xs2[:: args.every]
                ys2 = ys2[:: args.every]
            if args.ema > 0:
                ys2 = _ema(ys2, args.ema)

            # markers for sparse series (e.g., epoch-end val points)
            use_markers = len(xs2) <= 20

            # ---- choose axis & style ----
            send_to_y2 = (m in y2_metrics_user) or (auto_y2_enabled and _is_accuracy_like(m))
            target_ax = ax_left
            if send_to_y2:
                if ax_right is None:
                    ax_right = ax_left.twinx()
                    ax_right.set_ylabel("accuracy / alt-scale")
                target_ax = ax_right

            # stable per-run color
            run_name = (args.labels[idx] if args.labels else str(path.parent.name)).split(":")[0]
            if run_name not in color_for_run:
                color_for_run[run_name] = next(color_cycle)
            color = color_for_run[run_name]

            # train vs val linestyle; keep acc on right axis but same run color
            linestyle = "--" if (m.startswith("val/") and not _is_accuracy_like(m)) else "-"
            marker = "o" if use_markers else None

            # ---- single plot call (no duplicates) ----
            this_label = f"{label_base}:{m}"
            (ln,) = target_ax.plot(
                xs2,
                ys2,
                color=color,
                linestyle=linestyle,
                marker=marker,
                linewidth=(2.0 if marker else 1.6),
                label=this_label,
            )
            any_series = True

    if not any_series:
        print("[plot_loss.py: ERROR] Nothing to plot. Check --metrics and input format.", file=sys.stderr)
        return 2

    # Y-limits
    if args.ylim:
        ax_left.set_ylim(args.ylim[0], args.ylim[1])
    else:
        finite_all = [v for v in all_left_vals if isinstance(v, (int, float)) and math.isfinite(v)]
        if len(finite_all) >= 2:
            gmin, gmax = min(finite_all), max(finite_all)
            if (gmax - gmin) < 1e-2:
                span = max(1e-6, gmax - gmin)
                pad = max(1e-4, 0.1 * span)
                ax_left.set_ylim(gmin - pad, gmax + pad)

    # Legends (merge both axes if present)
    lines1, labels1 = ax_left.get_legend_handles_labels()
    lines2, labels2 = ax_right.get_legend_handles_labels() if ax_right else ([], [])
    ax_left.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=6)

    # Light padding for right axis if present (keeps ~0.10 off x-axis)
    if ax_right:
        ymin2, ymax2 = ax_right.get_ylim()
        ax_right.set_ylim(max(0.0, ymin2 - 0.05), min(1.0, ymax2 + 0.05))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[plot_loss.py: OK] Saved plot -> {out_path}")

    # Optional series summaries
    if args.summary:
        for s in summaries:
            if s["count"] > 0:
                print(f"[plot_loss.py: INFO] {s['label']} points={s['count']}, " f"y=[{s['y_min']:.6f}..{s['y_max']:.6f}], delta={s['y_range']:.6f}")
            else:
                print(f"[plot_loss.py: INFO] {s['label']} points=0 (no finite values)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
