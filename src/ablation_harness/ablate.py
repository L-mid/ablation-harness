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
Example Usage:

(baseline.yaml: study/example set):
    python -m ablation_harness.ablate --config experiments/baseline.yaml --out_dir runs/writeup_baseline --seed 11

(synthetic):
    python -m ablation_harness.ablate --config configs/toy_moons.yaml --out_dir runs/moons

(tiny real):
    python -m ablation_harness.ablate --config configs/tiny_cifar.yaml --out_dir runs/cifar_tiny


Make sure to delete previous runs before starting in same dir for intended behavoiur

Current user:
    python -m ablation_harness.ablate --config experiments/study_test.yaml --out_dir runs/del_test




--dry-run param (explicit command):
    python -m ablation_harness.ablate --config configs/toy_moons.yaml --dry-run
    # makes no dirs. works with toy_moons only. no seeding


--seed precedence rule:
    If --seed is provided → use that seed only (remove grid.seed).
    Else if grid.seed exists → sweep those seeds.
    Else use base.seed (default).

    usage:
        python -m ablation_harness.ablate --config configs/tiny_cifar.yaml --out_dir runs/cifar_tiny --seed 799


Supports both study + sweep schematics (needs serious cleaning).

"""

"""
TODO:
    Needs a better error message when runs fail.

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


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge
        else:
            out[k] = v
    return out


_warned_legacy_keys = False


def translate_legacy_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    """Map legacy study keys onto TrainConfig names without breaking sweeps."""
    global _warned_legacy_keys
    out = dict(d)
    if "n_samples" in out and "subset" not in out:
        out["subset"] = out.pop("n_samples")
        if not _warned_legacy_keys:
            eprint("[ablate.py: WARN] Translated legacy key 'n_samples' -> 'subset' in StudySpec overrides")
            _warned_legacy_keys = True
    return out


def resolve_sweep_spec(spec: Dict[str, Any], cli_seed: int | None) -> List[Dict[str, Any]]:
    base = spec.get("base", {}) or {}
    grid = dict(spec.get("grid", {}) or {})
    # CLI seed overrides any seed in grid or base
    if cli_seed is not None:
        base["seed"] = cli_seed
        grid.pop("seed", None)
        seeds = [cli_seed]
    else:
        seeds = grid.pop("seed", None)
        if seeds is None:
            seeds = [base.get("seed", 0)]
    # Build combos for non-seed params
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    combos = [dict(zip(keys, combo)) for combo in (itertools.product(*vals) if keys else [()])]
    runs: List[Dict[str, Any]] = []
    for combo in combos:
        for s in seeds:
            cfg = dict(base)
            cfg.update(combo)
            cfg["seed"] = s
            runs.append(cfg)
    return runs


def resolve_study_spec(spec: Dict[str, Any], cli_seed: int | None) -> List[Dict[str, Any]]:
    base = translate_legacy_keys(spec.get("baseline", {}) or {})
    seeds = spec.get("seeds", [base.get("seed", 0)])
    if cli_seed is not None:
        seeds = [cli_seed]
        base["seed"] = cli_seed
    variants = spec.get("variants", [])
    study_name = spec.get("study_name") or spec.get("name") or "study"
    runs: List[Dict[str, Any]] = []
    if not variants:
        # If no variants, treat baseline itself as one variant
        variants = [{"name": "baseline", "overrides": {}}]
    for v in variants:
        vname = v.get("name", "variant")
        overrides = translate_legacy_keys(v.get("overrides", {}) or {})
        cfg0 = deep_merge(base, overrides)
        for s in seeds:
            cfg = dict(cfg0)
            cfg["seed"] = s
            # Attach study metadata (trainer can ignore unknown keys)
            cfg["_study"] = study_name
            cfg["_variant"] = vname
            runs.append(cfg)
    return runs


def detect_schema(spec: Dict[str, Any]) -> str:
    if "base" in spec or "grid" in spec:
        return "sweep"
    schema = str(spec.get("schema", ""))
    if schema.startswith("study/") or "baseline" in spec or "variants" in spec:
        return "study"
    return "unknown"


def eprint(*a, **k):
    print(*a, **k, file=sys.stderr)


def main():  # noqa: C901

    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="YAML with either base/grid or a study spec")
    p.add_argument("--trainer", default="ablation_harness.trainer", help="Module with run(config_dict)->dict")
    p.add_argument("--dry-run", action="store_true", help="Plan and validate only; no writes/training")
    p.add_argument("--out_dir", default="runs/ablation", help="Output directory")
    p.add_argument("--seed", type=int, default=None, help="Override seed expansion (disables grid/study seed sweeps)")
    args = p.parse_args()
    spec = load_yaml(args.config)

    schema = detect_schema(spec)
    # metric/goal can live in either spec type
    metric = spec.get("metric", "val/acc")
    goal = spec.get("goal", "max")  # "max" or "min"

    if args.seed is not None:
        eprint("seed(s):", args.seed)

        has_base = isinstance(spec.get("base"), dict)
        has_grid = isinstance(spec.get("grid"), dict)
        has_baseline = isinstance(spec.get("baseline"), dict)
        looks_like_study = spec.get("schema") == schema.startswith("study/") or has_baseline or "variants" in spec or "seeds" in spec

        if has_base:
            spec["base"]["seed"] = args.seed
        if has_grid:
            spec["grid"].pop("seed", None)

        if looks_like_study:
            spec["baseline"]["seed"] = args.seed
            spec["seeds"] = args.seed

    if schema == "sweep":
        runs = resolve_sweep_spec(spec, args.seed)
    elif schema == "study":
        runs = resolve_study_spec(spec, args.seed)
    elif args.dry_run:
        grid = spec.get("grid", {})
        base = spec.get("base", {})
        runs = cartesian_grid(spec, grid) if grid else [base]
    else:
        raise SystemExit("Config must be a SweepSpec (base+grid) or StudySpec (schema: study/v1)")

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
    errors_in_any = None

    results: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for i, cfg in enumerate(runs):
            t0 = time.time()
            out: Dict[str, Any]
            try:
                out = run_fn(cfg)
            except Exception as e:
                out = {"error": str(e)}
                errors_in_any = True

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

    if errors_in_any:
        print("[ablate.py] WARNING: got an error in at least one run")  # consider checking and printing which
    print(f"[ablate.py] Wrote {jsonl_path}")
    print(f"[ablate.py] Wrote {csv_path}")


if __name__ == "__main__":
    main()
