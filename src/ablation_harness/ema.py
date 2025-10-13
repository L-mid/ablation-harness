from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Iterator, Optional

import torch
from torch import nn


@dataclass
class EMAConfig:
    decay: float = 0.999  # 0.999–0.9999 are common for vision; smaller for short runs
    device: Optional[torch.device] = None  # keep EMA on CPU to save VRAM (optional)
    pin_mem: bool = False  # only if device is CUDA and you want pinned mem copies
    include_buffers: bool = False  # usually False; BN buffers are already moving avgs
    warmup_steps: int = 0  # delay EMA until after N steps (optional)


class EMA:
    """
    Minimal, robust Exponential Moving Average for model parameters.
    - Initializes shadow = current model weights (so no bias correction needed).
    - Call `update(model)` after each optimizer.step().
    - Use `apply_to(model)` to evaluate with EMA weights, then restore.
    """

    def __init__(self, model: nn.Module, cfg: EMAConfig = EMAConfig()):
        """Initalizes using cfg from the dataclass."""
        self.decay = float(cfg.decay)
        self.include_buffers = cfg.include_buffers
        self.warmup_steps = int(cfg.warmup_steps)
        self._step = 0

        # collect tensors we will average
        def _iter_params():
            """if a param has grad, nab it and hold."""
            for p in model.parameters():  # if a param has grad, nab it.
                if p.requires_grad:
                    yield p  # holds it until use?

        self._shadow = []
        for p in _iter_params():  # for each param with grad,
            t = p.detach().clone()  # make a clone of it,
            if cfg.device is not None:
                t = t.to(cfg.device, non_blocking=True)  # assign it to correct device,
            self._shadow.append(t)  # and append it into _shadow []

        # optional: track buffers (rarely needed)
        self._buf_shadow = []
        if self.include_buffers:
            for b in model.buffers():
                t = b.detach().clone()
                if cfg.device is not None:
                    t = t.to(cfg.device, non_blocking=True)
                self._buf_shadow.append(t)

        self._param_index = [id(p) for p in _iter_params()]  # stable mapping. Maps each param with a unique id.

        # part of tracking buffers
        if self.include_buffers:
            self._buf_index = [id(b) for b in model.buffers()]
        else:
            self._buf_index = []

    @torch.no_grad()
    def update(self, model: nn.Module):
        """
        Updates the ema's weights with the model's current weight each optimizer step.
        Can be delayed using self.warmup_steps (in dataclass).

        Call after optimizer.step().
        """
        self._step += 1
        if self._step <= self.warmup_steps:
            # during warmup, keep shadow = live weights
            self.copy_from(model)
            return

        i = 0  # step counter for the shadow explicitly
        for p in model.parameters():
            if not p.requires_grad:
                continue
            shadow = self._shadow[i]
            shadow.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)  # beatiful math!!
            i += 1

        if self.include_buffers:
            j = 0  # step counter for buffer shadows explicitly
            for b in model.buffers():
                buf_shadow = self._buf_shadow[j]
                buf_shadow.mul_(self.decay).add_(b.detach(), alpha=1.0 - self.decay)  # in the case of buffers, update their math as well
                j += 1

    @torch.no_grad()
    def copy_from(self, model: nn.Module):
        """Hard copy current model → EMA shadow (used at init / warmup). Replaces all EMA's current weights."""
        i = 0  # shadows counted here as well (likely init here).

        for p in model.parameters():
            if not p.requires_grad:
                continue
            self._shadow[i].copy_(p.detach())
            i += 1  # if updated correctly, i = optimizer step.

        if self.include_buffers:
            j = 0
            for b in model.buffers():
                self._buf_shadow[j].copy_(b.detach())
                j += 1

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        """
        Hard copy EMA shadow → model (used for eval/export).
        Interesting, not sure purpous yet.
        """
        i = 0
        for p in model.parameters():
            if not p.requires_grad:
                continue
            p.detach().copy_(self._shadow[i])
            i += 1
        if self.include_buffers:
            j = 0
            for b in model.buffers():
                b.detach().copy_(self._buf_shadow[j])
                j += 1

    @contextlib.contextmanager
    def apply_to(self, model: nn.Module) -> Iterator[None]:
        """
        Temporarily swap model weights with EMA for evaluation:
            with ema.apply_to(model):
                eval_model(...)

        Interesting call method.
        """
        # save live weights
        live = [p.detach().clone() for p in model.parameters() if p.requires_grad]
        live_buf = [b.detach().clone() for b in model.buffers()] if self.include_buffers else None
        # swap in EMA
        self.copy_to(model)
        try:
            yield  # holds off continuing until teardown.
        finally:
            # restore live weights
            i = 0
            for p in model.parameters():
                if not p.requires_grad:
                    continue
                p.detach().copy_(live[i])
                i += 1

            if self.include_buffers and live_buf is not None:
                j = 0
                for b in model.buffers():
                    b.detach().copy_(live_buf[j])
                    j += 1

    def state_dict(self):
        """
        Save EMA state dict. Has a load_state_dict counterpart.

        WARNING for future checkpointing: custom saving format. Will not serialize clean with model.parameters() alone.
        """
        return {
            "decay": self.decay,
            "step": self._step,
            "shadow": [t.cpu() for t in self._shadow],
            "buf_shadow": [t.cpu() for t in self._buf_shadow] if self.include_buffers else None,
        }

    def load_state_dict(self, sd):
        """
        Load dict of EMA. Has a load_state_dict counterpart.

        WARNING for future checkpointing: custom loading format (uses state_dict's save format).
        """
        self.decay = float(sd["decay"])
        self._step = int(sd["step"])
        for dst, src in zip(self._shadow, sd["shadow"]):
            dst.copy_(src)
        if self.include_buffers and sd.get("buf_shadow") is not None:
            for dst, src in zip(self._buf_shadow, sd["buf_shadow"]):
                dst.copy_(src)
