"""
Helper for cli/planner and later train.py: resolves configs (even if nested) before run.
"""

from dataclasses import fields, is_dataclass
from typing import Any, Dict, Mapping, TypeVar, cast, get_type_hints

from .configs import RuntimeConfig, StudySpec

D = TypeVar("D")


ALLOW_UNKNOWN_KEYS = {"goal", "metric", "schema"}  # allow anywhere


def strict_merge(dc_obj: D, updates: Mapping[str, Any], path: str = "cfg") -> D:
    """
    Recursively merge Mapping `updates` into dataclass instance `dc_obj`.
    - Keeps nested dataclasses as instances (no dict leakage)
    - Raises on unknown keys
    - If a field is None but its annotated type is a dataclass and the update is a dict,
      it will be constructed and merged.
    - 'metric', 'goal', and '_<for_private_key>' keys specifically allowlisted, even if not in config.
    """
    if not is_dataclass(dc_obj) or isinstance(dc_obj, type):
        raise TypeError(f"{path} is not a dataclass instance (got {type(dc_obj)!r})")

    out = {f.name: getattr(dc_obj, f.name) for f in fields(dc_obj)}
    allowed = set(out)

    # Unknowns at this level
    unknown = set(updates) - allowed
    hard_unknown = {k for k in unknown if k not in ALLOW_UNKNOWN_KEYS and not str(k).startswith("_")}
    if hard_unknown:
        raise KeyError(f"Unknown keys under {path}: {sorted(hard_unknown)}")

    # Only merge allowed fields; drop allowlisted unknowns and privates
    updates = {k: v for k, v in updates.items() if k in allowed}

    hints = get_type_hints(type(dc_obj))
    for k, v in updates.items():
        cur = out[k]
        hint = hints.get(k)
        if is_dataclass(cur) and isinstance(v, Mapping):
            out[k] = strict_merge(cur, v, f"{path}.{k}")
        elif cur is None and isinstance(v, Mapping) and isinstance(hint, type) and is_dataclass(hint):
            out[k] = strict_merge(hint(), v, f"{path}.{k}")  # type: ignore[call-arg]
        else:
            out[k] = v

    return cast(D, type(dc_obj)(**out))


def resolve_spec(d: Dict[str, Any]) -> StudySpec:
    """Accepts a YAML dict that may be flat or already nested; merges strictly into StudySpec."""
    base = StudySpec()
    return strict_merge(base, d, path="spec")


def to_runtime(spec: StudySpec) -> RuntimeConfig:
    """Merges StudySpec into RuntimeConfig."""
    dev = spec.device or ("cuda" if _cuda_available() else "cpu")
    return RuntimeConfig(
        study_name=spec.study_name,
        run_id=spec.run_id,
        out_dir=spec.out_dir,
        seed=spec.seed,
        epochs=spec.epochs,
        device=dev,
        deterministic=spec.deterministic,
        clean_run=spec.clean_run,
        dataset=spec.data.dataset,
        subset=spec.data.subset,
        batch_size=spec.data.batch_size,
        num_workers=spec.data.num_workers,
        pin_memory=spec.data.pin_memory,
        shuffle=spec.data.shuffle,
        model_name=spec.model.name,
        hidden=spec.model.hidden,
        dropout=spec.model.dropout,
        opt_name=spec.optim.optimizer,
        lr=spec.optim.lr,
        wd=spec.optim.wd,
        momentum=spec.optim.momentum,
        sched_name=spec.sched.name,
        ema_enabled=spec.ema.enabled,
        ema_decay=spec.ema.decay,
        pin_mem=spec.ema.pin_mem,
        include_buffers=spec.ema.include_buffers,
        warmup_steps=spec.ema.warmup_steps,
        total_steps=spec.train.total_steps,
        grad_clip=spec.train.grad_clip,
        amp=spec.train.amp,
        beta_schedule=spec.diffusion.beta_schedule,
        task="diffusion" if spec.diffusion.enabled else None,
    )


def _cuda_available() -> bool:
    """Checks if cuda is avaliable."""
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def resolve_config(d: Dict[str, Any]):
    """Entry point: resolves default (RuntimeConfig) with overrides (provided cfg_dict: d)."""
    spec = resolve_spec(d)
    rt = to_runtime(spec)

    return rt, spec  # trainer uses rt; logger builder can read spec.logging, etc.
