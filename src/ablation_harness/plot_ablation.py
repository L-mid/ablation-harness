"""
UNTESTED.


Plots cfgs vs other cfgs.

Useage:
    python -m ablation_harness.plot_ablation runs/wk2_tinycnn/results.jsonl --metric val/acc --goal max --label-keys optimizer lr ema --out runs/wk2_tinycnn/plots


If you do more that 10 runs = non-legible.
< 10 works great!

"""

import argparse
import json
import os
import pathlib

import matplotlib.pyplot as plt


def _load_jsonl(path):
    "Loads jsonl. This one uses yield, which I don't understand."
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # allow a signle JSON object file as a fallback
                f.seek(0)
                data = json.load(f)
                if isinstance(data, dict):
                    yield data
                elif isinstance(data, list):
                    for d in data:
                        yield d
                return


def _get_cfg(d):
    """Best guesser find the cfg."""
    # try common places for the recorded config
    for k in ("cfg", "config", "run_cfg"):
        if k in d and isinstance(d[k], dict):
            return d[k]
    return {}


def _get_metric_value(d, metric_name):
    """
    Return the final value for metric_name.
    Search order (with aliases 'val/acc' <-> 'val_acc'):
      1) flat dicts: d['metrics'], d['final_metrics'], d['out']
      2) history lists: d['history'], d['log'], d['records'], d['out']['history']
      3) top-level fields
    """

    aliases = {metric_name}
    if "/" in metric_name:
        aliases.add(metric_name.replace("/", "_"))
    else:
        aliases.add(metric_name.replace("_", "/"))

    def _find_in_dict(maybe_dict):
        if isinstance(maybe_dict, dict):
            for alias in aliases:
                if alias in maybe_dict:
                    return maybe_dict[alias]
        return None

    # 1) flat dicts (common summary buckets + 'out')
    v = _find_in_dict(d.get("out", {}))
    if v is not None:
        return v

    # 2) top level
    for alias in aliases:
        if alias in d:
            return d[alias]

    return None


def _format_label(cfg, label_keys):
    """Formats bool (on/off), commas (,), or using run_id, run."""
    parts = []
    for k in label_keys:
        v = cfg.get(k, None)
        if isinstance(v, bool):
            v = "on" if v else "off"
        parts.append(f"{k}={v}")
    return ", ".join(parts) if parts else cfg.get("run_id", "run")


def main():
    p = argparse.ArgumentParser(description="Ablation bar plot from results.jsonl")
    p.add_argument("results", help="Path to results.jsonl")
    p.add_argument("--metric", default="val/acc", help="Metric key to plot (default: val/acc)")
    p.add_argument("--goal", choices=["max", "min"], default="max", help="Optimization direction")
    p.add_argument("--label-keys", nargs="*", default=["optimizer", "lr", "ema"], help="Config keys to show in x-labels")
    p.add_argument("--out", required=True, help="Output directory for .png")
    p.add_argument("--filename", default=None, help="Override output filename")
    args = p.parse_args()

    pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)

    rows = []
    for d in _load_jsonl(args.results):
        cfg = _get_cfg(d)
        val = _get_metric_value(d, args.metric)
        # print(val)
        if val is None:
            print("[plot_ablation] Got None for this row.")
            continue
        label = _format_label(cfg, args.label_keys)
        if (label == "run" or label is None) and isinstance(d.get("out"), dict):
            label = d["out"].get("run_id", label)

        rows.append((label, float(val)))

    if not rows:
        raise SystemExit(f"No values found for metric '{args.metric}' in {args.results}")

    # Stable sort: best first if goal=max, worst first if goal=min
    reverse = args.goal == "max"
    rows.sort(key=lambda x: x[1], reverse=reverse)

    labels, values = zip(*rows)

    plt.figure()
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), labels, rotation=30, ha="right")
    plt.ylabel(args.metric)
    title_dir = "higher is better" if args.goal == "max" else "lower is better"
    plt.title(f"Ablation - {args.metric} ({title_dir})")
    plt.tight_layout()

    fname = args.filename or f"ablation_bar_{args.metric.replace('/', '_')}.png"
    outpath = os.path.join(args.out, fname)
    plt.savefig(outpath, dpi=160)
    plt.close()

    print(f"[plot_ablation] Saved {outpath}")


if __name__ == "__main__":
    main()
