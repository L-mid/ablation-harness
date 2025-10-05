"""
cool side thing

untested.

"""

from __future__ import annotations

import math
from typing import Any, Dict

import torch
import torch.nn as nn


def _as_matrix(w: torch.Tensor) -> torch.Tensor:
    # Linear: (out, in) -> same; Conv: (out, in, kh, kw) -> (out, in*kh*kw)
    if w.ndim == 2:
        return w
    elif w.ndim == 4:
        out, in_ch, kh, kw = w.shape
        return w.reshape(out, in_ch * kh * kw)
    else:
        return w.flatten(1)  # fallback


@torch.no_grad()
def collect_spectral_stats(model: nn.Module, topk: int = 5, eps: float = 1e-9) -> Dict[str, Any]:
    stats = {}
    for name, m in model.named_modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)) and hasattr(m, "weight"):
            W = _as_matrix(m.weight.detach())
            # For small/medium layers, exact SVD is fine:
            s = torch.linalg.svdvals(W)  # sorted desc
            k = min(topk, s.numel())
            top = s[:k]
            fro2 = (s**2).sum()
            spectral = s[0].item()
            min_sv = s[-1].item() if s.numel() > 0 else 0.0
            cond = spectral / (min_sv + eps)
            stats[name] = {
                "rank": int((s > 0).sum().item()),
                "spectral_norm": float(spectral),
                "cond": float(cond),
                "nuclear": float(s.sum().item()),
                "fro_norm": float(math.sqrt(fro2.item())),
                "sv_topk": [float(x) for x in top.tolist()],
                "sv_energy_topk": float((top**2).sum().item() / (fro2.item() + eps)),
            }
    return stats
