import json
import math
from pathlib import Path

import torch
from torchvision.utils import save_image

from ablation_harness.tasks.diffusion.samplers import DDIMSampler, DDPMSampler


def _sample(model, shape, q, sampler, nfe, seed, device):
    if str(sampler).lower() == "ddim":
        smp = DDIMSampler(q=q, nfe=nfe, eta=0.0, device=device)
    else:
        smp = DDPMSampler(q=q, nfe=nfe, device=device)
    return smp.sample(model, shape, seed=seed)


@torch.inference_mode()
def evaluate_diffusion(model_ema, eval_cfg, q, out_dir, task: str | None = None):  # noqa C901
    """
    Run diffusion eval(s). If `task` is None, run all enabled tasks in eval_cfg.
    task ∈ {"grid", "kid", "fid_milestone", "final", None}
    Returns a dict with top-level 'fid', 'kid', 'n' plus 'details' per task.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = next(model_ema.parameters()).device
    model_ema.eval()

    res = {"fid": None, "kid": None, "n": 0, "details": {}}

    def run_grid():
        g = eval_cfg.grid
        if not getattr(g, "enabled", False):
            return
        n = int(g.n_samples)
        B = int(getattr(g, "batch_size", 64))
        seed = int(getattr(g, "sample_seed", getattr(eval_cfg, "sample_seed", 0)))
        sampler = str(g.sampler).lower()
        nfe = int(g.nfe)

        imgs = []
        remaining = n
        while remaining > 0:
            b = min(B, remaining)
            imgs.append(_sample(model_ema, (b, 3, 32, 32), q, sampler, nfe, seed, device))
            remaining -= b
        imgs = torch.cat(imgs, dim=0)  # in [-1, 1]

        if getattr(g, "save_images", False):
            grid = int(math.sqrt(min(n, 256)))
            save_image((imgs[: grid * grid].clamp(-1, 1) + 1) / 2, out_dir / "grid.png", nrow=grid)

        res["details"]["grid"] = {
            "n": n,
            "path": str(out_dir / "grid.png") if getattr(g, "save_images", False) else None,
        }
        res["n"] += n

    def run_kid():
        kcfg = eval_cfg.kid
        if not getattr(kcfg, "enabled", False):
            return None
        # Placeholder: wire up real KID later (feature extractor + polynomial MMD).
        kid_now = None
        res["kid"] = kid_now
        res["details"]["kid"] = {
            "kid": kid_now,
            "n": int(kcfg.n_samples),
            "repeats": int(kcfg.repeats),
        }
        return kid_now

    def run_fid_milestone(kid_now):
        fcfg = eval_cfg.fid_milestone
        if not getattr(fcfg, "enabled", False) or not getattr(fcfg, "fid_stats", None):
            return
        gate = float(getattr(fcfg, "run_if_kid_improved_pct", 0.0))
        should_run = gate <= 0.0

        # Persist best KID across calls to make gating stateful
        best_file = out_dir / "kid_best.json"
        prev_best = None
        if best_file.exists():
            try:
                prev_best = json.loads(best_file.read_text()).get("best_kid", None)
            except Exception:
                prev_best = None

        if gate > 0.0 and (kid_now is not None) and (prev_best is not None):
            should_run = kid_now <= prev_best * (1.0 - gate / 100.0)

        # Update best if we measured KID
        if kid_now is not None:
            new_best = kid_now if prev_best is None else min(prev_best, kid_now)
            best_file.write_text(json.dumps({"best_kid": new_best}))

        if not should_run:
            res["details"]["fid_milestone"] = {"skipped": True, "reason": "gate_not_met"}
            return

        # Placeholder: wire up real FID later (use fcfg.fid_stats).
        fid_val = None
        res["fid"] = fid_val
        res["details"]["fid_milestone"] = {
            "fid": fid_val,
            "n": int(fcfg.n_samples),
            "fid_stats": fcfg.fid_stats,
        }

    def run_final():
        f = eval_cfg.final
        if not getattr(f, "enabled", False):
            return
        # Typically called only once at train-end by the trainer.
        # Placeholder: wire up real FID later.
        res["details"]["final"] = {
            "n": int(f.n_samples),
            "fid_stats": f.fid_stats,
            "sampler": str(f.sampler),
            "nfe": int(f.nfe),
        }

    # Quick clamp for smoke tests (optional)
    if getattr(eval_cfg, "quick", False):
        if hasattr(eval_cfg, "grid"):
            eval_cfg.grid.nfe = min(eval_cfg.grid.nfe, 5)
            eval_cfg.grid.n_samples = min(eval_cfg.grid.n_samples, 2)
        if hasattr(eval_cfg, "kid"):
            eval_cfg.kid.nfe = min(eval_cfg.kid.nfe, 5)
            eval_cfg.kid.n_samples = min(eval_cfg.kid.n_samples, 2)

    # Execute
    if task in (None, "grid"):
        run_grid()
    kid_now = None
    if task in (None, "kid"):
        kid_now = run_kid()
    if task in (None, "fid_milestone"):
        run_fid_milestone(kid_now)
    if task in (None, "final"):
        run_final()

    return res
