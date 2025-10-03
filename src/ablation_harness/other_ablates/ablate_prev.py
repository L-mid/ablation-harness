import argparse
import csv
import json
import pathlib
import time
from dataclasses import dataclass
from math import sqrt
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from ablation_harness.seed_utils import set_seed


@dataclass
class Experiment:
    name: str
    n_samples: int = 1000


def run_one(exp: Experiment, seed: int) -> Dict[str, Any]:
    set_seed(seed)
    x = np.random.randn(exp.n_samples)
    loss = float(np.mean(np.abs(x)))
    return {"config_name": exp.name, "seed": seed, "metrics": {"dummy_loss": loss}}


def load_cfg(path: str):
    cfg = OmegaConf.load(path)
    exps = [Experiment(**e) for e in cfg.experiments]
    seeds = list(cfg.seeds)
    return exps, seeds


def aggregate(rows):
    by = {}
    for r in rows:
        by.setdefault(r["config_name"], []).append(r["metrics"]["dummy_loss"])
    # mean/std/n
    stats = {}
    for name, vals in by.items():
        arr = np.array(vals, float)
        stats[name] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)),
            "n": int(arr.size),
            "vals": arr,
        }
    # effect vs first config (Welch t-test)
    names = sorted(stats.keys())
    base = names[0]
    for name in names:
        if name == base:
            stats[name].update({"delta": 0.0, "pvalue": 1.0})
            continue
        a, b = stats[base], stats[name]
        m1, s1, n1 = a["mean"], a["std"], a["n"]
        m2, s2, n2 = b["mean"], b["std"], b["n"]
        # Welch t
        denom = sqrt((s1**2) / n1 + (s2**2) / n2) if n1 > 1 and n2 > 1 else float("inf")
        t = (m2 - m1) / denom if denom > 0 else 0.0
        # very small-sample approx (no SciPy): two-sided p via normal approx
        # (good enough to rank, not for papers)
        import math

        p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / sqrt(2))))
        stats[name].update({"delta": m2 - m1, "pvalue": float(p)})

    table = [
        (
            name,
            stats[name]["mean"],
            stats[name]["std"],
            stats[name]["n"],
            stats[name]["delta"],
            stats[name]["pvalue"],
        )
        for name in names
    ]
    return table


def write_report(table, out_dir: pathlib.Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    md = [
        "# Ablation Report",
        "",
        "| config | mean | std | n | Delta vs base | p (Welch, ~normal) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, mean, std, n, delta, p in table:
        md.append(f"| {name} | {mean:.6f} | {std:.6f} | {n} | {delta:+.6f} | {p:.3f} |")
    (out_dir / "ablate.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def write_csv(table, out_path: pathlib.Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "mean", "std", "n", "delta_vs_base", "pvalue"])
        for row in table:
            w.writerow(row)


def plot_seed_variance(rows: List[Dict[str, Any]], out_dir: pathlib.Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    by = {}
    for r in rows:
        by.setdefault(r["config_name"], []).append(r["metrics"]["dummy_loss"])
    for name, vals in by.items():
        plt.figure()
        plt.title(f"Seed variance: {name}")
        plt.plot(vals, marker="o")
        plt.xlabel("seed idx")
        plt.ylabel("dummy_loss")
        plt.tight_layout()
        (out_dir / f"seed_variance_{name}.png").parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / f"seed_variance_{name}.png", dpi=150)
        plt.close()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True, help="Path to experiments YAML")
    p.add_argument("--out", default=None, help="Runs dir (default: runs/<ts>)")
    p.add_argument("--append", action="store_true", help="Append instead of overwrite.")
    args = p.parse_args(argv)

    exps, seeds = load_cfg(args.config)
    ts = time.strftime("%Y%m%d-%H%M%S")
    import os

    run_id = f"{ts}-{os.getpid()}"
    runs_dir = pathlib.Path(args.out) if args.out else pathlib.Path("runs") / f"ablate-{ts}"
    reports_dir = pathlib.Path("reports")
    plots_dir = reports_dir / "plots"
    runs_dir.mkdir(parents=True, exist_ok=True)

    results_path = runs_dir / "results.jsonl"
    rows = []
    t0 = time.time()
    for e in exps:
        for s in seeds:
            r = run_one(e, s)
            r["run_id"] = run_id
            r["elapsed_sec"] = round(time.time() - t0, 4)
            r["timestamp"] = ts
            rows.append(r)

    mode = "a" if args.append else "w"  # append is bugged
    with results_path.open(mode, encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    table = aggregate(rows)
    write_report(table, reports_dir)
    write_csv(table, reports_dir / "ablate.csv")
    plot_seed_variance(rows, plots_dir)
    print(f"[abalate.py] Wrote {results_path} and reports/ablate.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
