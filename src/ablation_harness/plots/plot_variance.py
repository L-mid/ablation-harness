"""
## What the variance plot is meant to show

    It's a seed variance view: for each single config, compute mean ± std of a metric across multiple seeds.
    One bar = one config (as defined by --label-fields).
    Height = mean of the metric over seeds.
    Error bar = standard deviation over seeds.
    The “n=” annotation is how many rows got grouped into that bar (ideally, number of seeds).


Useage:

    python -m ablation_harness.plot_variance runs/wk2_tinycnn/results.jsonl \
    --metric val/acc \
    --label-fields optimizer,lr,ema \
    --out runs/wk2_tinycnn/plots/variance_val-acc.png \
    --metric-name "val/acc" \
    --title "Seed variance: TinyCNN CIFAR-subset"


# Current:
    python -m ablation_harness.plot_variance runs/tinycnn_tester/results.jsonl
    --metric val/acc  --label-fields dropout  --out runs/tinycnn_tester/plots/variance_val-acc.png --metric-name "val/acc"  --title "Seed variance: TinyCNN CIFAR-subset"


"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Reads jsonl. (non-compact)."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _get_path(d: Dict[str, Any], path: str) -> Any:
    """
    Get a nested value via dotted or slash-separated path.
    Example: "metrics.val.acc" or "metrics/val/acc".
    """
    keys = path.replace("/", ".").split(".")
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _label_from_cfg(cfg: Dict[str, Any], label_fields: List[str]) -> str:
    """
    Build a human-readable label from selected cfg fields.
    Missing fields are shown as '-'.
    """
    parts = []
    for f in label_fields:
        v = _get_path(cfg, f)
        if isinstance(v, float):
            # keep concise formatting
            v = f"{v:g}"
        parts.append(f"{f.split('.')[-1]}={v if v is not None else '-'}")
    return ", ".join(parts) if parts else "config"


def _get_metric_value(row: Dict[str, Any], metric_path: str) -> Any:
    """
    Robust metric lookup supporting your schema:
      - row['out']['val/acc']  (primary)
      - others can be added.
    """
    mp = metric_path

    # 1) Direct key in out: 'val/acc'
    out = row.get("out", {})
    if isinstance(out, dict) and mp in out:
        return out[mp]


def compute_group_stats(
    rows: Iterable[Dict[str, Any]],
    metric_path: str,
    label_fields: List[str],
) -> List[Tuple[str, float, float, int]]:
    """
    Returns: list of (label, mean, std, n).
    std=0.0 when n<2 so plotting never crashes.
    """
    grouped: Dict[str, List[float]] = {}

    for r in rows:
        cfg = r.get("cfg", {})
        val = _get_metric_value(r, metric_path)
        if val is None:
            continue  # skip rows without the metric

        label = _label_from_cfg(cfg, label_fields)
        grouped.setdefault(label, []).append(float(val))

    stats: List[Tuple[str, float, float, int]] = []
    for label, vals in grouped.items():
        n = len(vals)
        mean = sum(vals) / n if n else float("nan")
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n >= 2 else 0.0
        stats.append((label, mean, std, n))

    # Stable ordering by mean desc then label
    stats.sort(key=lambda t: (-float("-inf") if math.isnan(t[1]) else -t[1], t[0]))
    return stats


def plot_variance(
    stats: List[Tuple[str, float, float, int]],
    out_path: Path,
    title: str,
    metric_name: str,
) -> None:
    """
    Bar chart with error bars = std. Annotates n for each bar.
    Never crashes on n=1 (std=0).

    Works like a mini-main.
    """
    if not stats:
        raise ValueError("No stats to plot (empty dataset or metric missing).")

    labels = [s[0] for s in stats]
    means = [s[1] for s in stats]
    stds = [s[2] for s in stats]
    ns = [s[3] for s in stats]

    # Plot
    plt.figure(figsize=(max(6, len(labels) * 0.9), 4.5))
    x = range(len(labels))
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel(metric_name)
    plt.title(title)

    # Annotate n above each bar
    for i, (m, n) in enumerate(zip(means, ns)):
        if not math.isnan(m):
            plt.text(i, m, f"n={n}", ha="center", va="bottom", fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def main():
    """
    Central Orchestrator.
    """

    p = argparse.ArgumentParser(description="Plot mean±std across seeds per config.")
    p.add_argument("results_jsonl", type=Path, help="Path to results.jsonl")
    p.add_argument("--metric", required=True, help='Metric path, e.g. "metrics.val.acc" or "val/acc"')
    p.add_argument(
        "--label-fields",
        default="optimizer,lr,ema",
        help='Comma-separated cfg fields for grouping label (supports dotted), e.g. "optimizer,lr,ema"',
    )
    p.add_argument("--out", type=Path, required=True, help="Output PNG path")
    p.add_argument("--title", default="Seed variance", help="Plot title")
    p.add_argument("--metric-name", default="val/acc", help="Y-axis label")
    args = p.parse_args()

    rows = list(_read_jsonl(args.results_jsonl))
    label_fields = [s.strip() for s in args.label_fields.split(",") if s.strip()]
    stats = compute_group_stats(rows, args.metric, label_fields)
    plot_variance(stats, args.out, args.title, args.metric_name)

    print(f"[plot_variance] OK, plotted to: {args.out}")


if __name__ == "__main__":
    main()
