# --- helpers: keep types light and local ---

"""
Untested.
"""


import argparse
import csv
import importlib
import itertools
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List

import yaml


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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="YAML with base/grid/metric/goal")
    p.add_argument("--trainer", default="ablation_harness.trainer", help="Module with run(config_dict)->dict")
    p.add_argument("--dry-run", action="store_true", help="Plan and validate only; no writes/training")
    p.add_argument("--out_dir", default="runs/ablation", help="Output directory")
    return p


def parse_spec(spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list[Any]], str, str]:
    base = spec.get("base", {}) or {}
    grid = spec.get("grid", {}) or {}
    metric = spec.get("metric", "val/acc")
    goal = spec.get("goal", "max")  # "max" or "min"
    return base, grid, metric, goal


def resolve_runs(base: dict[str, Any], grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    return cartesian_grid(base, grid) if grid else [base]


def import_trainer(modname: str) -> tuple[Callable[[dict[str, Any]], dict[str, Any]], Callable[[dict[str, Any]], dict[str, Any]]]:
    mod = importlib.import_module(modname)
    run_fn = getattr(mod, "run")
    preflight_fn = getattr(mod, "preflight")
    return run_fn, preflight_fn


def make_plan(runs: list[dict[str, Any]], out_dir: str, metric: str, goal: str) -> dict[str, Any]:
    return {
        "run_count": len(runs),
        "metric": metric,
        "goal": goal,
        "out_dir": out_dir,
        "runs": [
            {
                "i": i,
                "cfg": cfg,
                "planned_artifacts": [
                    os.path.join(out_dir, "results.jsonl"),
                    os.path.join(out_dir, "summary.csv"),
                ],
                "preflight": None,
            }
            for i, cfg in enumerate(runs)
        ],
    }


def _diff(base_cfg: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in cfg.items() if base_cfg.get(k) != v}


def _grid_cardinality(plan_runs: list[dict[str, Any]]) -> dict[str, int]:
    from collections import defaultdict

    vals = defaultdict(set)
    for r in plan_runs:
        for k, v in r["cfg"].items():
            vals[k].add(v)
    return {k: len(vs) for k, vs in vals.items()}


def preflight_plan(plan: dict[str, Any], preflight_fn: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
    for entry in plan["runs"]:
        try:
            entry["preflight"] = preflight_fn({**entry["cfg"], "dry_run": True})
        except Exception as e:
            entry["preflight"] = {"ok": False, "error": str(e)}


def print_plan_summary(plan: dict[str, Any], top: int = 10, errors_only: bool = False) -> None:
    runs = plan["runs"]
    errs = [r for r in runs if (r.get("preflight") or {}).get("ok") is False]
    oks = [r for r in runs if (r.get("preflight") or {}).get("ok") is not False]
    eprint(f"DRY-RUN 'OK!'  runs={len(runs)}  ok={len(oks)}  errors={len(errs)}")
    eprint(f"metric={plan['metric']}  goal={plan['goal']}")
    eprint(f"out_dir={plan['out_dir']}")
    eprint("grid:", _grid_cardinality(runs))
    base_cfg = runs[0]["cfg"] if runs else {}
    to_show = errs if errors_only else runs[:top]
    for r in to_show:
        pf = r.get("preflight") or {}
        eprint(f"[{r['i']:>3}] diff={_diff(base_cfg, r['cfg'])}  " f"pf_ok={pf.get('ok','NA')}  params={pf.get('params','?')}")


def execute_runs(
    runs: list[dict[str, Any]],
    run_fn: Callable[[dict[str, Any]], dict[str, Any]],
    out_dir: str,
    metric: str,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], str]:
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, "results.jsonl")
    results: list[tuple[dict[str, Any], dict[str, Any]]] = []
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
    return results, jsonl_path


def rank_results(results: list[tuple[dict[str, Any], dict[str, Any]]], metric: str, goal: str):
    def score(o: dict[str, Any]) -> float:
        v = o.get(metric, None)
        if v is None:
            return float("-inf") if goal == "max" else float("inf")
        return float(v)

    return sorted(results, key=lambda co: score(co[1]), reverse=(goal == "max"))


def write_csv_summary(sorted_rows, csv_path: str, metric: str) -> None:
    fieldnames = list(sorted(set(k for cfg, _ in sorted_rows for k in cfg.keys())))
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        header = ["rank", metric, "_elapsed_sec"] + fieldnames
        w.writerow(header)
        for rank, (cfg, out) in enumerate(sorted_rows, 1):
            row = [rank, out.get(metric), out.get("_elapsed_sec")] + [cfg.get(k) for k in fieldnames]
            w.writerow(row)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = load_yaml(args.config)
    base, grid, metric, goal = parse_spec(spec)
    runs = resolve_runs(base, grid)
    eprint(f"[ablate.py] {len(runs)} runs")
    run_fn, preflight_fn = import_trainer(args.trainer)

    if args.dry_run:
        plan = make_plan(runs, args.out_dir, metric, goal)
        preflight_plan(plan, preflight_fn)
        print(json.dumps(plan, indent=2))
        print_plan_summary(plan)
        bad = any((r.get("preflight") or {}).get("ok") is False for r in plan["runs"])
        raise SystemExit(2 if bad else 0)

    results, jsonl_path = execute_runs(runs, run_fn, args.out_dir, metric)
    sorted_rows = rank_results(results, metric, goal)
    csv_path = os.path.join(args.out_dir, "summary.csv")
    write_csv_summary(sorted_rows, csv_path, metric)

    print(f"[ablate.py] Wrote {jsonl_path}")
    print(f"[ablate.py] Wrote {csv_path}")
    return 0
