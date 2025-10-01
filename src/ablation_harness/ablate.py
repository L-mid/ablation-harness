import argparse
import csv
import importlib
import itertools
import json
import os
import time
from typing import Any, Callable, Dict, List, Tuple

import yaml

"""
running calls:

Teir 1 (synthetic)
python -m ablation_harness.ablate --config configs/toy_moons.yaml --out_dir runs/moons

Teir 2 (tiny real)
python -m ablation_harness.ablate --config configs/tiny_cifar.yaml --out_dir runs/cifar_tiny

pytest -q tests/test_ablate_smoke.py -k smoke


pip install -e ".[dev,torch-cpu]"

"""


def cartesian_grid(base: Dict[str, Any], grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    runs = []
    for combo in itertools.product(*vals):
        cfg = dict(base)
        for k, v in zip(keys, combo):
            cfg[k] = v
        runs.append(cfg)
    return runs


def load_yaml(path: str) -> Dict[str, Any]:  # here
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():

    import torch

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))

    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="YAML with base/grid/metric/goal")
    p.add_argument(
        "--trainer", default="ablation_harness.trainer", help="Module with run(config_dict)->dict"
    )
    p.add_argument("--out_dir", default="runs/ablation", help="Output directory")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    spec = load_yaml(args.config)  # here (more likely)

    base = spec.get("base", {})
    grid = spec.get("grid", {})
    metric = spec.get("metric", "val/acc")
    goal = spec.get("goal", "max")  # "max" or "min"

    runs = cartesian_grid(base, grid) if grid else [base]
    print(f"[ablate.py] {len(runs)} runs")

    trainer_mod = importlib.import_module(args.trainer)  # cannot find 'trainer' module
    run_fn: Callable[[dict[str, Any]], dict[str, Any]] = getattr(trainer_mod, "run")

    results: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    jsonl_path = os.path.join(args.out_dir, "results.jsonl")
    csv_path = os.path.join(args.out_dir, "summary.csv")

    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for i, cfg in enumerate(runs):
            t0 = time.time()
            out: Dict[str, Any]
            try:
                out = run_fn(cfg)
            except Exception as e:
                out = {"error": str(e)}
            out["_elapsed_sec"] = round(time.time() - t0, 3)
            rec = {"cfg": cfg, "out": out, "_i": i}
            jf.write(json.dumps(rec) + "\n")
            results.append((cfg, out))
            print(
                (
                    f"[{i+1}/{len(runs)}] {cfg} -> {out.get(metric, 'NA')} "
                    f"({out.get('_elapsed_sec', '?')}s)"
                )
            )

    # Rank
    def score_fn(o: Dict[str, Any]) -> float:
        v = o.get(metric, None)
        if v is None:
            return float("-inf") if goal == "max" else float("inf")
        return float(v)

    sorted_rows = sorted(
        results,
        key=lambda co: score_fn(co[1]),
        reverse=(goal == "max"),
    )

    # Write CSV summary
    fieldnames = list(sorted(set(k for cfg, _ in results for k in cfg.keys())))
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        header = ["rank", metric, "_elapsed_sec"] + fieldnames
        w.writerow(header)
        for rank, (cfg, out) in enumerate(sorted_rows, 1):
            row = [rank, out.get(metric), out.get("_elapsed_sec")] + [
                cfg.get(k) for k in fieldnames
            ]
            w.writerow(row)

    print(f"[ablate.py] Wrote {jsonl_path}")
    print(f"[ablate.py] Wrote {csv_path}")


if __name__ == "__main__":
    main()
