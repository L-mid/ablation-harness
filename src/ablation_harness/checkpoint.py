import io
import math
import os
import tempfile
from pathlib import Path
from typing import Tuple, Union

import torch

from .run_layout import RunLayout

Best = Tuple[int, float]


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Writes bytes atomically. (native to checkpoint.py)"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def _pack(model, opt, ema, epoch, best_val):
    """
    Packs states of model peices together coherently for checkpoint saving.
    The actual ckpt saver.
    """
    buf = io.BytesIO()
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "ema": ema.state_dict(),
            "epoch": int(epoch),
            "best_val": float(best_val),
        },
        buf,
    )
    return buf.getvalue()


def try_resume(layout: RunLayout):
    """Simple ckpt resuming logic. (if p.exists (last.pt) -> continue with resume)"""
    p = layout.ckpts / "last.pt"
    if p.exists():
        return torch.load(p, map_location="cpu")
    return None


def save_last(layout: RunLayout, model, opt, ema, epoch: int, best_val: float):
    """Saves last ckpt."""
    _atomic_write_bytes(layout.ckpts / "last.pt", _pack(model, opt, ema, epoch, best_val))


best_metric = float("-inf")
best_epoch = -1


def metric_from_fid(fid: float | None) -> float:
    """Map FID → "higher is better" metric; treat None/NaN/±inf as -inf"""
    if fid is None or not math.isfinite(fid):
        return float("-inf")
    return -float(fid)


def _normalize_best(best: Union[float, Best]) -> Best:
    """Allow old float-style best; coerce to (epoch, value)"""
    if isinstance(best, tuple):
        return best
    return (-1, float(best))


def save_best_if_better(layout, model, opt, ema, epoch: int, best: Union[float, Best], metric: float, allow_ties: bool = False) -> Best:
    """Save 'best' checkpoint iff metric improved (higher is better)."""
    best_epoch, best_val = _normalize_best(best)
    improved = (metric >= best_val) if allow_ties else (metric > best_val)
    if math.isfinite(metric) and improved:
        _atomic_write_bytes(layout.ckpts / "last.pt", _pack(model, opt, ema, epoch, best_val))
        return (epoch, metric)
    return (best_epoch, best_val)
