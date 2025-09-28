import argparse
import json
import pathlib
import time
from dataclasses import dataclass
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


def aggregate(rows: List[Dict[str, Any]]):
    by = {}
    for r in rows:
        by.setdefault(r["config_name"], []).append(r["metrics"]["dummy_loss"])
    table = []
    for k, v in by.items():
        import numpy as np

        arr = np.array(v, float)
        table.append((k, float(arr.mean()), float(arr.std(ddof=1)), len(arr)))
    table.sort()
    return table


def write_report(table, out_dir: pathlib.Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    md = ["# Ablation Report", "", "| config | mean | std | n |", "|---|---:|---:|---:|"]
    for name, mean, std, n in table:
        md.append(f"| {name} | {mean:.6f} | {std:.6f} | {n} |")
    (out_dir / "ablate.md").write_text("\n".join(md) + "\n", encoding="utf-8")


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

    mode = "a" if args.append else "w"
    with results_path.open(mode, encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    table = aggregate(rows)
    write_report(table, reports_dir)
    plot_seed_variance(rows, plots_dir)
    print(f"Wrote {results_path} and reports/ablate.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
