import argparse
import csv
import importlib
import itertools
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Tuple

import yaml

"""
running calls:

Teir 1 (synthetic)
python -m ablation_harness.ablate --config configs/toy_moons.yaml --out_dir runs/moons

Teir 2 (tiny real)
python -m ablation_harness.ablate --config configs/tiny_cifar.yaml --out_dir runs/cifar_tiny

pytest -q

python -m ablation_harness.ablate --config configs/toy_moons.yaml --dry-run   # should make no dirs. works with toy_moons, no seeding


for installation: pip install -e ".[dev,torch-cpu]"


seed precedence rule:
    If --seed is provided → use that seed only (remove grid.seed).
    Else if grid.seed exists → sweep those seeds.
    Else use base.seed (default).

    python -m ablation_harness.ablate --config configs/tiny_cifar.yaml --out_dir runs/cifar_tiny --seed 799

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


def eprint(*a, **k):
    print(*a, **k, file=sys.stderr)


def main():  # noqa: C901

    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="YAML with base/grid/metric/goal")
    p.add_argument("--trainer", default="ablation_harness.trainer", help="Module with run(config_dict)->dict")
    p.add_argument("--dry-run", action="store_true", help="Plan and validate only; no writes/training")
    p.add_argument("--out_dir", default="runs/ablation", help="Output directory")
    p.add_argument("--seed", type=int, default=None, help="Override seed for a single run")
    args = p.parse_args()
    spec = load_yaml(args.config)

    base = spec.get("base", {})
    grid = spec.get("grid", {})
    metric = spec.get("metric", "val/acc")
    goal = spec.get("goal", "max")  # "max" or "min"

    if args.seed is not None:
        eprint("seed(s):", args.seed)
        spec["base"]["seed"] = args.seed
        # If user asked for a single seed, don't also sweep seeds from YAML:
        if "grid" in spec and "seed" in spec["grid"]:
            del spec["grid"]["seed"]

    runs = cartesian_grid(base, grid) if grid else [base]

    eprint(f"[ablate.py] {len(runs)} runs")

    trainer_mod = importlib.import_module(args.trainer)
    run_fn: Callable[[dict[str, Any]], dict[str, Any]] = getattr(trainer_mod, "run")
    preflight_fn = getattr(trainer_mod, "preflight")

    plan = {
        "run_count": len(runs),
        "metric": metric,
        "goal": goal,
        "out_dir": args.out_dir,
        "runs": [],
    }

    # dry run exclusives

    def _diff(base, cfg):
        return {k: v for k, v in cfg.items() if base.get(k) != v}

    def _grid_cardinality(runs):
        from collections import defaultdict

        vals = defaultdict(set)
        for r in runs:
            for k, v in r["cfg"].items():
                vals[k].add(v)
        return {k: len(vs) for k, vs in vals.items()}

    def _print_summary(plan, top=10, errors_only=False):
        runs = plan["runs"]
        errs = [r for r in runs if (r.get("preflight") or {}).get("ok") is False]
        oks = [r for r in runs if (r.get("preflight") or {}).get("ok") is not False]
        eprint(f"DRY-RUN 'OK!'  runs={len(runs)}  ok={len(oks)}  errors={len(errs)}")
        eprint(f"metric={plan['metric']}  goal={plan['goal']}")
        eprint(f"out_dir={plan['out_dir']}")
        eprint("grid:", _grid_cardinality(runs))
        base = runs[0]["cfg"] if runs else {}
        to_show = errs if errors_only else runs[:top]
        for r in to_show:
            pf = r.get("preflight") or {}
            eprint(f"[{r['i']:>3}] diff={_diff(base, r['cfg'])}  pf_ok={pf.get('ok', 'NA')}  params={pf.get('params','?')}")

    # If DRY-RUN: build a plan and optionally ask trainer to preflight each cfg
    if args.dry_run:
        for i, cfg in enumerate(runs):
            entry = {
                "i": i,
                "cfg": cfg,
                "planned_artifacts": [
                    os.path.join(args.out_dir, "results.jsonl"),
                    os.path.join(args.out_dir, "summary.csv"),
                ],
                "preflight": None,
            }
            try:
                entry["preflight"] = preflight_fn({**cfg, "dry_run": True})
            except Exception as e:
                entry["preflight"] = {"ok": False, "error": str(e)}
            plan["runs"].append(entry)
        # IMPORTANT: not makedirs, no writes

        print(json.dumps(plan, indent=2))  # prints the full dict
        _print_summary(plan)  # prints dict summary

        # exit nonzero if any error
        bad = any((r.get("preflight") or {}).get("ok") is False for r in plan["runs"])
        raise SystemExit(2 if bad else 0)

    os.makedirs(args.out_dir, exist_ok=True)
    jsonl_path = os.path.join(args.out_dir, "results.jsonl")
    csv_path = os.path.join(args.out_dir, "summary.csv")

    results: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
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
            print((f"[{i+1}/{len(runs)}] {cfg} -> {out.get(metric, 'NA')} " f"({out.get('_elapsed_sec', '?')}s)"))

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
            row = [rank, out.get(metric), out.get("_elapsed_sec")] + [cfg.get(k) for k in fieldnames]
            w.writerow(row)

    print(f"[ablate.py] Wrote {jsonl_path}")
    print(f"[ablate.py] Wrote {csv_path}")


if __name__ == "__main__":
    main()
