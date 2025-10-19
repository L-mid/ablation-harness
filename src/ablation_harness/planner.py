import copy
import itertools
import sys
from typing import Any, Dict, List

import yaml

CONTROL_KEYS = {
    "schema",
    "study_name",
    "name",
    "sweep_name",
    "variants",
    "seeds",
    "out_dir",
    "grid",
}


def _get(cfg: dict, path: str, default=None):
    """Dot-path getter that supports both nested and flat fallback."""
    cur = cfg
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return cfg.get(path, default)  # flat fallback (legacy)
    return cur


def _assert_unique_run_ids(runs: list[dict], enabled=False) -> None:
    """
    Asserts run_ids are unique and gives debugging print. (does NOT let collisons through, will crash runs).
    Enable with enabled=True.
    """

    if not enabled:
        return None

    seen = {}
    for r in runs:
        rid = r["run_id"]
        if rid in seen:
            a, b = seen[rid], r
            eprint("run_id collision:", rid)
            eprint("A only keys:", {k: a.get(k) for k in a.keys() - b.keys()})
            eprint("B only keys:", {k: b.get(k) for k in b.keys() - a.keys()})
            eprint("Diff on shared keys:", {k: (a[k], b[k]) for k in a.keys() & b.keys() if a[k] != b[k]})
            raise ValueError(f"Duplicate run_id: {rid}")
        seen[rid] = r


def _as_list(x):
    """Returns a list if None and converts to list otherwise."""
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _extract_base(spec: Dict[str, Any]) -> Dict[str, Any]:
    "Take everything at the root except control/private keys"
    return {k: v for k, v in spec.items() if k not in CONTROL_KEYS and not k.startswith("_")}


def eprint(*a, **k):
    """Prints specifically to stderr."""
    print(*a, **k, file=sys.stderr)


def load_yaml(path: str) -> Dict[str, Any]:
    """Loads yaml (no Omegaconf)."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge things into dict."""
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def detect_schema(spec: Dict[str, Any]) -> str:
    """Detects if sweep (from 'grid') or study (from study)."""
    if "grid" in spec:
        return "sweep"
    sch = str(spec.get("schema", ""))
    if sch.startswith("study/"):
        return "study"
    return "unknown"


def plan(spec: Dict[str, Any], cli_seed: int | None) -> List[Dict[str, Any]]:  # legacy
    """Determines which schema plan was provided."""
    schema = detect_schema(spec)
    if schema == "sweep":
        return _resolve_sweep_spec(spec, cli_seed)
    if schema == "study":
        return _resolve_study_spec(spec, cli_seed)
    raise ValueError("Config must be SweepSpec (base+grid) or StudySpec (schema: study/vX)")


# Run id logic:


def make_run_id(study_name: str, cfg: dict) -> str:
    """Makes a run_id for every run."""
    model = _get(cfg, "model.name")
    dataset = _get(cfg, "data.dataset")
    dropout = _get(cfg, "model.dropout")
    opt = _get(cfg, "optim.optimizer")
    lr = _get(cfg, "optim.lr")
    wd = _get(cfg, "optim.wd")
    ema = _get(cfg, "ema.enabled")
    seed = _get(cfg, "seed")

    def fmt_float(v):
        if isinstance(v, float):
            s = f"{v:.0e}" if v != 0.0 and (abs(v) < 1e-3 or abs(v) >= 1e3) else f"{v:g}"
            return s.replace("+", "")
        return str(v)

    parts = [
        study_name,
        model,
        dataset,
        f"dro{fmt_float(dropout)}" if dropout is not None else None,
        opt,
        f"lr{fmt_float(lr)}" if lr is not None else None,
        f"wd{fmt_float(wd)}" if wd is not None else None,
        ("ema1" if ema else "ema0") if ema is not None else None,
        f"seed={seed}" if seed is not None else None,
    ]
    SAFE = str.maketrans({"/": "-", "\\": "-", " ": "_"})
    return "__".join(str(p) for p in parts if p is not None).translate(SAFE)


def _resolve_sweep_spec(spec: Dict[str, Any], cli_seed: int | None) -> List[Dict[str, Any]]:
    """(depreceated) Resolves if sweep spc is valid + seed override rules."""
    base = spec.get("base", {}) or {}
    grid = dict(spec.get("grid", {}) or {})
    if cli_seed is not None:
        # apply override seed
        base["seed"] = cli_seed
        grid.pop("seed", None)
        seeds = [cli_seed]
    else:
        seeds = grid.pop("seed", None) or [base.get("seed", 0)]
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    combos = [dict(zip(keys, combo)) for combo in (itertools.product(*vals) if keys else [()])]

    runs = []
    for combo in combos:
        # normalize each combo too (it might contain flat keys like 'lr')
        cfg0 = deep_merge(base, combo)
        for s in seeds:
            cfg = dict(cfg0)
            cfg["seed"] = s
            # keep a consistent study/study_name key
            sweep_name = spec.get("sweep_name") or spec.get("study_name") or "sweep"
            cfg["study_name"] = sweep_name
            cfg["run_id"] = make_run_id(sweep_name, cfg)
            runs.append(cfg)
            _assert_unique_run_ids(runs, enabled=False)  # enable for hard crashing + debugging print
    return runs


def _resolve_study_spec(spec: Dict[str, Any], cli_seed: int | None) -> List[Dict[str, Any]]:
    """Plan runs from a study spec without using `baseline:`."""
    study_name = spec.get("study_name") or spec.get("name") or "study"

    # Base cfg is just the root (minus control keys)
    base = _extract_base(spec)

    # Seeds: prefer CLI, else `seeds: [...]`, else `seed: int`, else [0]
    seeds = [cli_seed] if cli_seed is not None else (_as_list(spec.get("seeds")) or _as_list(spec.get("seed")) or [0])

    variants = spec.get("variants") or [{"name": "default", "overrides": {}}]

    runs: List[Dict[str, Any]] = []
    for v in variants:
        overrides = v.get("overrides") or {}
        cfg0 = deep_merge(base, overrides)  # keep nested structure
        for s in seeds:
            cfg = copy.deepcopy(cfg0)
            cfg["seed"] = s
            cfg["study_name"] = study_name
            cfg["run_id"] = make_run_id(study_name, cfg)
            runs.append(cfg)

    _assert_unique_run_ids(runs, enabled=False)
    return runs


def print_preview(runs: List[dict], metric="val/acc"):
    """Plans the print returned throughout training."""
    print(f"{len(runs)} planned runs")
    cols = sorted({k for r in runs for k in r.keys() if k in ("optimizer", "lr", "wd", "ema", "seed")})
    header = ["#", *cols]
    print(" | ".join(header))
    for i, r in enumerate(runs, 1):
        row = [str(i), *[str(r.get(c, "")) for c in cols]]
        print(" | ".join(row))
