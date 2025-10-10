"""
cfg style assumed:

{"cfg":{"optimizer":"adam","lr":0.001,"wd":0.0,"ema":false,"epochs":25},
 "out":{"val/acc":0.42,"val/loss":2.11,"_elapsed_sec":142.7,"run_id":"..."}}


Useage:
    python -m scripts.aggregate runs/wk2_tinycnn/results.jsonl --metric val/acc --goal max --cols optimizer lr wd ema --timing _elapsed_sec --out reports/wk2_ablation.md

Optional: --watch to auto-refresh while runs append

"""

import argparse
import json
import time
from pathlib import Path


def read_jsonl(path: Path):
    """
    Reads a jsonl file.
        Takes its rows str, and puts them in: rows = [].
    """
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                # tolerate partial line if file being written; skip it
                continue
    return rows


def normalize_row(row, metric_key, timing_key):
    """Fetch essential keys from json/jsonl."""

    cfg = row.get("cfg", {})
    out = row.get("out", {})
    metric = out.get(metric_key)
    timing = out.get(timing_key) if timing_key else None
    return cfg, out, metric, timing


def to_markdown(rows, cfg_cols, metric_key, timing_key):
    """Emits cfgs given to .md."""
    # Header
    hdr = ["config"] + cfg_cols + [metric_key]
    if timing_key:
        hdr += [timing_key]
    lines = []
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "|".join("---" for _ in hdr) + "|")

    # Examines each config to give certain aspects
    # Builds markdown body
    for i, (cfg, out, metric, timing) in enumerate(rows, start=1):
        vals = [str(i)]
        for k in cfg_cols:
            v = cfg.get(k, "")
            if isinstance(v, bool):
                v = "on" if v else "off"  # defaults to off on wrong key
            vals.append(str(v))
        vals.append(f"{metric:.3f}" if isinstance(metric, (int, float)) else str(metric))
        if timing_key:
            v = out.get(timing_key, timing)
            vals.append(f"{v:.1f}" if isinstance(v, (int, float)) else (str(v) if v is not None else ""))
        lines.append("| " + "| ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def aggregate_once(src: Path, out_md: Path, metric_key: str, goal: str, cfg_cols, timing_key: str):
    """Runs like a mini main, aggregates and appends all text to .md using helper fn."""
    raw = read_jsonl(src)  # the rows in: []
    parsed = []
    for r in raw:
        cfg, out, metric, timing = normalize_row(r, metric_key, timing_key)
        if metric is None:  # skip runs that haven't logged the target metric yet
            continue
        parsed.append((cfg, out, metric, timing))

    if not parsed:
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text("_No complete runs found yet._\n", encoding="utf-8")
        return

    reverse = goal == "max"
    parsed.sort(key=lambda t: t[2], reverse=reverse)

    # Build a tiny preamble + table
    pre = ["# Ablation results\n", f"- **Source:** `{src.as_posix()}`", f"- **Metric:** `{metric_key}` ({'maximize' if reverse else 'minimize'})", ""]
    table = to_markdown(parsed, cfg_cols=cfg_cols, metric_key=metric_key, timing_key=timing_key)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(pre) + table, encoding="utf-8")


def main():
    """Works like model forward. + parses args and --watch logic."""
    # parse args:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path, help="Path to results.jsonl")
    ap.add_argument("--metric", required=True, help="Metric key to rank/sort by (e.g., val/acc)")
    ap.add_argument("--goal", choices=["max", "min"], default="max")
    ap.add_argument("--cols", nargs="+", default=["optimizer", "lr", "wd", "ema"], help="cfg keys to show as columns")
    ap.add_argument("--timing", default=None, help="Optional timing key in out (e.g., _elapsed_sec)")
    ap.add_argument("--out", type=Path, required=True, help="Where to write Markdown table")
    ap.add_argument("--watch", action="store_true", help="Continuously rebuild while file changes")
    ap.add_argument("--interval", type=float, default=2.0, help="Watch interval seconds")
    args = ap.parse_args()

    # --watch & --interval logic
    last_mtime = None
    mtime = None

    print(f"[Aggregate.py] Writing Markdown table to: {args.out}.")
    while True:
        try:
            mtime = args.jsonl.stat().st_mtime
        except FileNotFoundError:
            # Write a placeholder if not yet created
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_test("_Waiting for results.jsonl..._\n", encoding="utf-8")
            if not args.watch:  # interesting, before check?
                return

        if (not args.watch) or (mtime != last_mtime):
            aggregate_once(src=args.jsonl, out_md=args.out, metric_key=args.metric, goal=args.goal, cfg_cols=args.cols, timing_key=args.timing)
            last_mtime = mtime

        if not args.watch:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
