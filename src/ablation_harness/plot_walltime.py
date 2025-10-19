"""
Might want to fix how it shows many runs (text too big/something)


What it does: Bar chart of wall-time per run.


Useage example:
    python -m ablation_harness.plot_walltime runs/wk2_tinycnn/results.jsonl --label-keys optimizer lr ema --out runs/wk2_tinycnn/plots


"""

import argparse
import json
import os
import pathlib
from typing import Any, Dict, List

import matplotlib.pyplot as plt


def _load_jsonl(path):
    """Loads jsonl. Uses yield (not understood)."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                f.seek(0)
                data = json.load(f)
                if isinstance(data, dict):
                    yield data
                elif isinstance(data, list):
                    for d in data:
                        yield d
                return


def _get_cfg(d):
    """Best guess at proviced cfg (d)."""
    for k in ("cfg", "config", "run_cfg"):
        if k in d and isinstance(d[k], dict):
            return d[k]
    return {}


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


def _get_walltime_s(d):  # ignore C901
    """Best guess get time out of cfg."""

    # current method
    t = d.get("out", {})
    if isinstance(t, dict):
        if t["_elapsed_sec"] == 0.0:
            return None
        else:
            try:
                return float((t["_elapsed_sec"]))
            except Exception:
                pass

    return None


def main():
    p = argparse.ArgumentParser(description="Wall-time bar plot from results.jsonl")
    p.add_argument("results", help="PATH to results.jsonl")
    p.add_argument("--label-keys", nargs="*", default=["optimizer", "lr", "ema"], help="Config keys to show in x-labels")
    p.add_argument("--out", required=True, help="Output directory for .png")
    p.add_argument("--filename", default="walltime_bar.png", help="Output filename")
    args = p.parse_args()

    pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)

    rows = []
    for d in _load_jsonl(args.results):
        wt = _get_walltime_s(d)
        if wt is None:
            print("[plot_walltime] WARNING: This row had no walltime.")
            continue
        cfg = _get_cfg(d)
        label = _label_from_cfg(cfg, args.label_keys)
        rows.append((label, float(wt)))

    if not rows:
        raise SystemExit(f"No wall-time fields found in {args.results}. " f"Expected one of timing.total_sec / total_time_s / epoch_times / timestamps.")

    # Sort shortest first so “faster is better” reads left→right
    rows.sort(key=lambda x: x[1])
    labels, values = zip(*rows)

    plt.figure()
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), labels, rotation=30, ha="right")
    plt.ylabel("wall-time (s)")
    plt.title("Wall-time per run")
    plt.tight_layout()

    outpath = os.path.join(args.out, args.filename)
    plt.savefig(outpath, dpi=160)
    plt.close()

    print(f"[plot_walltime] Saved {outpath}")


if __name__ == "__main__":
    main()
