import io
import os
import tempfile
from pathlib import Path

import torch

from .run_layout import RunLayout


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


def save_best_if_better(layout: RunLayout, model, opt, ema, epoch: int, best_val: float, metric: float) -> float:
    """Saves the latest ckpt only if ckpt's metrics improved (if metric > best_val). + chooses path for ckpt."""
    if metric > best_val:
        _atomic_write_bytes(layout.ckpts / "best_val.pt", _pack(model, opt, ema, epoch, metric))
        return metric
    return best_val
