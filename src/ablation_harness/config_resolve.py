"""
Helper for trainer.py: resolves configs (even if nested) before run.
"""

from dataclasses import fields, is_dataclass
from typing import Any, Dict, Mapping, TypeVar, cast, get_type_hints

D = TypeVar("D")


def _is_dc_type(t: Any) -> bool:
    return isinstance(t, type) and is_dataclass(t)


def strict_merge(dc_obj: D, updates: Mapping[str, Any], path: str = "cfg") -> D:
    """
    Recursively merge Mapping `updates` into dataclass instance `dc_obj`.
    - Keeps nested dataclasses as instances (no dict leakage)
    - Raises on unknown keys
    - If a field is None but its annotated type is a dataclass and the update is a dict,
      it will be constructed and merged.
    """
    if not is_dataclass(dc_obj) or isinstance(dc_obj, type):
        raise TypeError(f"{path} is not a dataclass instance (got {type(dc_obj)!r})")

    # current values
    out = {f.name: getattr(dc_obj, f.name) for f in fields(dc_obj)}
    allowed = set(out)
    unknown = set(updates) - allowed
    if unknown:
        raise KeyError(f"Unknown keys under {path}: {sorted(unknown)}")

    # type hints from the dataclass class (resolve forward refs)
    hints = get_type_hints(type(dc_obj))

    for k, v in updates.items():
        cur = out[k]
        hint = hints.get(k)

        # Case 1: current value is a dataclass instance and incoming is a mapping → recurse
        if is_dataclass(cur) and isinstance(v, Mapping):
            out[k] = strict_merge(cur, v, f"{path}.{k}")
            continue

        # Case 2: current is None, but the field type itself is a dataclass and update is a mapping → construct & merge
        if cur is None and isinstance(v, Mapping) and _is_dc_type(hint):
            out[k] = strict_merge(hint(), v, f"{path}.{k}")  # type: ignore[call-arg]
            continue

        # Fallback: replace (numbers/strings/lists/dicts etc.)
        out[k] = v

    # rebuild the same dataclass type; cast is for the type checker only
    return cast(D, type(dc_obj)(**out))


def resolve_config(d: Dict[str, Any]):
    from ablation_harness.trainer import TrainConfig

    """Entry point: resolves default with override."""
    # Start from defaults and apply user overrides strictly
    defaults = TrainConfig()

    # If your YAML nests under 'baseline', point at it explicitly:
    updates = d.get("baseline", d)
    return strict_merge(defaults, updates, path="cfg")
