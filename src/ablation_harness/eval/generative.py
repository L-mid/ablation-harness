import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models
from torchvision.utils import save_image

from ablation_harness.tasks.diffusion.samplers import DDIMSampler, DDPMSampler

_INCEPTION_CACHE = None


def _get_inception(device: torch.device):
    """Return a cached Inception v3 backbone that outputs feature vectors."""
    global _INCEPTION_CACHE
    if _INCEPTION_CACHE is None:
        # Use torchvision Inception v3; we take the output *before* the final FC.
        weights = models.Inception_V3_Weights.DEFAULT
        net = models.inception_v3(weights=weights, transform_input=False)
        net.fc = torch.nn.Identity()  # fc now just passes through pooled features/logits
        net.eval()
        _INCEPTION_CACHE = net
    return _INCEPTION_CACHE.to(device)


def _inception_activations(x: torch.Tensor, device: torch.device) -> np.ndarray:
    """
    x: [B, 3, H, W] in [0,1]
    Returns numpy array [B, D] of features.
    """
    net = _get_inception(device)
    # Resize from e.g. 32x32 → 299x299
    x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
    with torch.inference_mode():
        feats = net(x)
    return feats.detach().cpu().numpy()


def _sqrtm_psd(mat: np.ndarray) -> np.ndarray:
    """Matrix square root for (almost) PSD matrices via eigen-decomposition."""
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, 0.0, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


def _fid_from_stats(mu_gen: np.ndarray, sigma_gen: np.ndarray, mu_ref: np.ndarray, sigma_ref: np.ndarray) -> float:
    """Standard Fréchet distance between two Gaussians."""
    mu_gen = np.atleast_1d(mu_gen)
    mu_ref = np.atleast_1d(mu_ref)
    sigma_gen = np.atleast_2d(sigma_gen)
    sigma_ref = np.atleast_2d(sigma_ref)

    diff = mu_gen - mu_ref
    cov_prod = sigma_gen.dot(sigma_ref)
    covmean = _sqrtm_psd(cov_prod)
    fid = diff.dot(diff) + np.trace(sigma_gen) + np.trace(sigma_ref) - 2.0 * np.trace(covmean)
    return float(np.real(fid))


def _fid_for_generated(
    model_ema,
    q,
    device: torch.device,
    n_samples: int,
    sampler: str,
    nfe: int,
    fid_stats_path: str | Path,
    batch_size: int = 64,
    seed: int = 0,
) -> float:
    ...
    total_batches = math.ceil(n_samples / batch_size)
    print(f"[fid] Generating {n_samples} images -- {total_batches} batches")
    print_on_batch = 1

    data = np.load(fid_stats_path)
    mu_ref = data["mu"]
    sigma_ref = data["sigma"]

    remaining = int(n_samples)
    feats_list = []

    sampler_name = str(sampler).lower()
    print(f"[fid] sampler={sampler_name}, nfe={nfe}, batch_size={batch_size}, seed={seed}")

    g_seed = int(seed)
    while remaining > 0:
        b = min(batch_size, remaining)
        batch_idx = (n_samples - remaining) // batch_size + 1
        if (batch_idx % print_on_batch) == 0:
            print(f"[fid] batch {batch_idx}/{total_batches} (size={b}, g_seed={g_seed})")

        imgs = _sample(
            model_ema,
            (b, 3, 32, 32),
            q=q,
            sampler=sampler_name,
            nfe=nfe,
            seed=g_seed,
            device=device,
        )

        # [-1,1] → [0,1]
        imgs = (imgs.clamp(-1, 1) + 1.0) / 2.0

        # TEMP: break things on purpose
        # version A: all zeros
        # imgs = torch.zeros_like(imgs)
        # version B: flip vertically
        # imgs = imgs.flip(dims=[2])
        print("[fid]   imgs mean/std:", float(imgs.mean()), float(imgs.std()))

        feats = _inception_activations(imgs, device)
        feats_list.append(feats)
        remaining -= b
        g_seed += 1

    feats_all = np.concatenate(feats_list, axis=0)
    mu_gen = np.mean(feats_all, axis=0)
    sigma_gen = np.cov(feats_all, rowvar=False)

    return _fid_from_stats(mu_gen, sigma_gen, mu_ref, sigma_ref)


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
        """Works fid: also persists kid stats to make gating for fid milestones successful."""
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

        # If we have a gate and a current KID, decide whether to run FID
        if gate > 0.0 and kid_now is not None:
            if prev_best is None:
                should_run = True
                best_file.write_text(json.dumps({"best_kid": float(kid_now)}))
            else:
                # Relative improvement (%)
                rel_improve = (prev_best - kid_now) / max(prev_best, 1e-12) * 100.0
                should_run = rel_improve >= gate
                if kid_now < prev_best:
                    best_file.write_text(json.dumps({"best_kid": float(kid_now)}))

        if not should_run:
            res["details"]["fid_milestone"] = {"skipped": True, "reason": "gate_not_met"}
            return

        n = int(fcfg.n_samples)
        sampler = getattr(fcfg, "sampler", "ddpm")
        nfe = int(getattr(fcfg, "nfe", 50))
        seed = int(getattr(eval_cfg, "sample_seed", 0))
        batch_size = int(getattr(fcfg, "batch_size", 64))

        fid_val = _fid_for_generated(
            model_ema=model_ema,
            q=q,
            device=device,
            n_samples=n,
            sampler=sampler,
            nfe=nfe,
            fid_stats_path=fcfg.fid_stats,
            batch_size=batch_size,
            seed=seed,
        )

        res["fid"] = float(fid_val)
        res["details"]["fid_milestone"] = {
            "fid": float(fid_val),
            "n": n,
            "fid_stats": fcfg.fid_stats,
            "sampler": str(sampler),
            "nfe": nfe,
        }
        res["n"] += n

    def run_final():
        f = eval_cfg.final
        if not getattr(f, "enabled", False) or not getattr(f, "fid_stats", None):
            return
        # Typically called only once at train-end by the trainer.

        n = int(f.n_samples)
        sampler = getattr(f, "sampler", "ddpm")
        nfe = int(getattr(f, "nfe", 50))
        seed = int(getattr(eval_cfg, "sample_seed", 0))
        batch_size = int(getattr(f, "batch_size", 64))

        fid_val = _fid_for_generated(
            model_ema=model_ema,
            q=q,
            device=device,
            n_samples=n,
            sampler=sampler,
            nfe=nfe,
            fid_stats_path=f.fid_stats,  # stats/<name>
            batch_size=batch_size,
            seed=seed,
        )

        res["fid"] = float(fid_val)
        res["details"]["final"] = {
            "fid": float(fid_val),
            "n": n,
            "fid_stats": f.fid_stats,
            "sampler": str(sampler),
            "nfe": nfe,
        }
        res["n"] += n

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
