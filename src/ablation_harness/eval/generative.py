import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision as tv
from torch.utils.data import DataLoader, Subset
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


def _inception_activations(
    x: torch.Tensor,
    device: torch.device,
    batch_size: int | None = None,
) -> np.ndarray:
    """
    x: [B, 3, H, W] in [0,1]
    Returns numpy array [B, D] of features.
    batch_size: chunk size for running Inception to control memory.
    """
    net = _get_inception(device)  # should be cached + .eval() inside
    B = int(x.shape[0])

    if batch_size is None:
        # conservative defaults; tweak if you want
        batch_size = 64 if device.type == "cuda" else 16
    batch_size = max(1, min(int(batch_size), B))

    feats_cpu = []
    with torch.inference_mode():
        for i in range(0, B, batch_size):
            xb = x[i : i + batch_size]

            # Move first, then resize on-device (avoids huge CPU temp + big H2D copies)
            xb = xb.to(device, non_blocking=True).float()

            xb = F.interpolate(xb, size=(299, 299), mode="bilinear", align_corners=False)

            fb = net(xb)  # make sure this returns the *FID features* (e.g. pooled 2048-d)
            fb = fb.flatten(1)  # safe if fb is [N, D, 1, 1] or already [N, D]

            feats_cpu.append(fb.cpu())

    feats = torch.cat(feats_cpu, dim=0)
    return feats.numpy()


def _make_psd(sigma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma = 0.5 * (sigma + sigma.T)

    # Only add jitter if needed
    w = np.linalg.eigvalsh(sigma)
    min_w = float(w.min())
    if min_w < 0.0:
        sigma = sigma + (-min_w + eps) * np.eye(sigma.shape[0], dtype=np.float64)
    return sigma


def _trace_sqrt_product_psd(sigma_a: np.ndarray, sigma_b: np.ndarray) -> float:
    # assumes inputs are already symmetric/PSD-stabilized
    wa, va = np.linalg.eigh(sigma_a)
    wa = np.clip(wa, 0.0, None)

    # work in eigenbasis (avoids forming sqrt_a explicitly)
    M = va.T @ sigma_b @ va
    s = np.sqrt(wa)
    A = (s[:, None] * M) * s[None, :]
    A = 0.5 * (A + A.T)

    wA = np.linalg.eigvalsh(A)
    wA = np.clip(wA, 0.0, None)
    return float(np.sum(np.sqrt(wA)))


def _fid_from_stats(mu_gen, sigma_gen, mu_ref, sigma_ref) -> float:
    mu_gen = np.atleast_1d(mu_gen).astype(np.float64)
    mu_ref = np.atleast_1d(mu_ref).astype(np.float64)

    sigma_gen = _make_psd(np.atleast_2d(sigma_gen))
    sigma_ref = _make_psd(np.atleast_2d(sigma_ref))

    diff = mu_gen - mu_ref
    tr_sqrt = _trace_sqrt_product_psd(sigma_gen, sigma_ref)

    fid = float(diff.dot(diff) + np.trace(sigma_gen) + np.trace(sigma_ref) - 2.0 * tr_sqrt)

    if fid < 0.0 and fid > -1e-6:
        fid = 0.0
    return fid


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

    feats_all = np.concatenate(feats_list, axis=0).astype(np.float64)
    mu_gen = np.mean(feats_all, axis=0)
    sigma_gen = np.cov(feats_all, rowvar=False)

    return _fid_from_stats(mu_gen, sigma_gen, mu_ref, sigma_ref)


# kid helpers:
def _kid_mmd2_unbiased_poly(X: np.ndarray, Y: np.ndarray, degree: int = 3) -> float:
    """
    Unbiased MMD^2 with polynomial kernel used for KID:
      k(x,y) = (x^T y / d + 1)^degree
    X, Y: [m, d] float64
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    assert X.ndim == 2 and Y.ndim == 2
    m, d = X.shape
    assert Y.shape == (m, d)

    Kxx = (X @ X.T) / d
    Kyy = (Y @ Y.T) / d
    Kxy = (X @ Y.T) / d

    Kxx = (Kxx + 1.0) ** degree
    Kyy = (Kyy + 1.0) ** degree
    Kxy = (Kxy + 1.0) ** degree

    # unbiased: exclude diagonal terms
    sum_xx = (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
    sum_yy = (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
    sum_xy = Kxy.mean()
    return float(sum_xx + sum_yy - 2.0 * sum_xy)


def _kid_from_pools(
    feats_gen: np.ndarray,
    feats_real: np.ndarray,
    subset_size: int,
    repeats: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(int(seed))
    n = int(min(len(feats_gen), len(feats_real)))
    feats_gen = np.asarray(feats_gen[:n], dtype=np.float64)
    feats_real = np.asarray(feats_real[:n], dtype=np.float64)

    m = int(min(subset_size, n))
    reps = int(max(1, repeats))

    vals = []
    for r in range(reps):
        ix = rng.choice(n, size=m, replace=False)
        iy = rng.choice(n, size=m, replace=False)
        vals.append(_kid_mmd2_unbiased_poly(feats_gen[ix], feats_real[iy], degree=3))

    vals = np.asarray(vals, dtype=np.float64)
    mean = float(vals.mean())
    std = float(vals.std(ddof=1)) if reps > 1 else 0.0
    sem = float(std / math.sqrt(reps)) if reps > 1 else 0.0
    return mean, std, sem


def _real_inception_feats_cifar10(
    device: torch.device,
    n_samples: int,
    batch_size: int,
    seed: int,
    split: str = "train",
    root: str = ".",
    num_workers: int = 2,
) -> np.ndarray:
    train = str(split).lower() != "test"
    ds = tv.datasets.CIFAR10(root=root, train=train, download=True, transform=tv.transforms.ToTensor())

    rng = np.random.default_rng(int(seed))
    idx = np.arange(len(ds))
    rng.shuffle(idx)
    idx = idx[: int(n_samples)]
    dl = DataLoader(
        Subset(ds, idx.tolist()),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=(device.type == "cuda"),
    )

    feats = []
    for xb, _ in dl:
        feats.append(_inception_activations(xb, device, batch_size=int(batch_size)))
    return np.concatenate(feats, axis=0).astype(np.float64)


def _gen_inception_feats(
    model_ema,
    q,
    device: torch.device,
    n_samples: int,
    sampler: str,
    nfe: int,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    remaining = int(n_samples)
    feats_list = []
    g_seed = int(seed)
    sampler_name = str(sampler).lower()

    while remaining > 0:
        b = min(int(batch_size), remaining)
        imgs = _sample(
            model_ema,
            (b, 3, 32, 32),
            q=q,
            sampler=sampler_name,
            nfe=int(nfe),
            seed=g_seed,
            device=device,
        )
        imgs = (imgs.clamp(-1, 1) + 1.0) / 2.0  # -> [0,1]
        feats_list.append(_inception_activations(imgs, device, batch_size=int(batch_size)))
        remaining -= b
        g_seed += 1

    return np.concatenate(feats_list, axis=0).astype(np.float64)


# sampling helper


def _sample(model, shape, q, sampler, nfe, seed, device):
    if str(sampler).lower() == "ddim":
        smp = DDIMSampler(q=q, nfe=nfe, eta=0.0, device=device)
    else:
        smp = DDPMSampler(q=q, nfe=nfe, device=device)
    return smp.sample(model, shape, seed=seed)


@torch.no_grad()
def evaluate_diffusion(model_ema, eval_cfg, q, out_dir, task=None, state_dir=None, step=None, kid_now=None):  # noqa C901
    """
    Run diffusion eval(s). If `task` is None, run all enabled tasks in eval_cfg.
    task ∈ {"grid", "kid", "fid_milestone", "final", None}
    Returns a dict with top-level 'fid', 'kid', 'n' plus 'details' per task.
    """

    # debug
    with torch.no_grad():
        w = next(model_ema.parameters())
        print("[generative.py debug] eval weight mean/std:", float(w.mean()), float(w.std()))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    step_i = int(step) if step is not None else None
    state_dir = Path(state_dir) if state_dir is not None else out_dir
    state_dir.mkdir(parents=True, exist_ok=True)

    kid_last_file = state_dir / "kid_last.json"
    kid_best_file = state_dir / "kid_best.json"

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
        batch_k = 0
        while remaining > 0:
            b = min(B, remaining)
            imgs.append(_sample(model_ema, (b, 3, 32, 32), q, sampler, nfe, seed + batch_k, device))
            remaining -= b
            batch_k += 1
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
        """A kid implementation."""
        kcfg = eval_cfg.kid
        if not getattr(kcfg, "enabled", False):
            return None

        n = int(getattr(kcfg, "n_samples", 1024))
        repeats = int(getattr(kcfg, "repeats", 3))
        subset_size = int(getattr(kcfg, "subset_size", 100))
        sampler = str(getattr(kcfg, "sampler", "ddim")).lower()
        nfe = int(getattr(kcfg, "nfe", 10))
        batch_size = int(getattr(kcfg, "batch_size", 64))
        seed = int(getattr(eval_cfg, "sample_seed", 0))
        real_seed = int(getattr(kcfg, "real_seed", 123))
        real_split = str(getattr(kcfg, "real_split", "train")).lower()

        # cache real feats so repeated evals don’t keep recomputing them
        real_cache = Path(kcfg.feature_cache) if getattr(kcfg, "feature_cache", None) else state_dir / f"kid_real_feats_{real_split}_n{n}_seed{real_seed}.npy"
        if real_cache.exists():
            feats_real = np.load(real_cache).astype(np.float64)
        else:
            feats_real = _real_inception_feats_cifar10(
                device=device,
                n_samples=n,
                batch_size=batch_size,
                seed=real_seed,
                split=real_split,
            )
            np.save(real_cache, feats_real)

        feats_gen = _gen_inception_feats(
            model_ema=model_ema,
            q=q,
            device=device,
            n_samples=n,
            sampler=sampler,
            nfe=nfe,
            batch_size=batch_size,
            seed=seed,
        )

        kid_mean, kid_std, kid_sem = _kid_from_pools(
            feats_gen=feats_gen,
            feats_real=feats_real,
            subset_size=subset_size,
            repeats=repeats,
            seed=seed + 999,
        )

        res["kid"] = float(kid_mean)
        res["details"]["kid"] = {
            "kid_mean": float(kid_mean),
            "kid_std": float(kid_std),
            "kid_sem": float(kid_sem),
            "n_pool": n,
            "subset_size": subset_size,
            "repeats": repeats,
            "sampler": sampler,
            "nfe": nfe,
            "real_split": real_split,
            "real_cache": str(real_cache),
        }

        # ---- persist state for later gating ----
        prev_best = None
        prev_best_step = None
        if kid_best_file.exists():
            try:
                j = json.loads(kid_best_file.read_text())
                prev_best = j.get("best_kid", None)
                prev_best_step = j.get("best_step", None)
            except Exception:
                prev_best = None

        # compute rel improvement vs previous best (percentage)
        rel_improve_pct = None
        if prev_best is not None:
            rel_improve_pct = (float(prev_best) - float(kid_mean)) / max(float(prev_best), 1e-12) * 100.0

        # update best (store best_step too)
        updated_best = False
        best_kid = float(prev_best) if prev_best is not None else None
        best_step = int(prev_best_step) if prev_best_step is not None else None

        if prev_best is None or float(kid_mean) < float(prev_best):
            updated_best = True
            best_kid = float(kid_mean)
            best_step = step_i

        kid_best_file.write_text(json.dumps({"best_kid": best_kid, "best_step": best_step}))

        # write kid_last with "best_before" so fid gating can be computed stably
        kid_last_file.write_text(
            json.dumps(
                {
                    "step": step_i,
                    "kid": float(kid_mean),
                    "best_before": float(prev_best) if prev_best is not None else None,
                    "rel_improve_pct": float(rel_improve_pct) if rel_improve_pct is not None else None,
                    "updated_best": bool(updated_best),
                }
            )
        )

        return float(kid_mean)

    def run_fid_milestone(kid_now):  # noqa C901
        """Works fid: also persists kid stats to make gating for fid milestones successful."""
        fcfg = eval_cfg.fid_milestone
        if not getattr(fcfg, "enabled", False) or not getattr(fcfg, "fid_stats", None):
            return

        gate = float(getattr(fcfg, "run_if_kid_improved_pct", 0.0))
        should_run = gate <= 0.0

        if kid_now is None and gate > 0.0:

            if kid_last_file.exists():
                try:
                    kid_last = json.loads(kid_last_file.read_text())
                except Exception:
                    kid_last = None
            else:
                kid_last = None

            if kid_last is None:
                res["details"]["fid_milestone"] = {"skipped": True, "reason": "kid_missing"}
                return

            # If step is provided, require same-step kid to avoid stale gating
            if step_i is not None and kid_last.get("step", None) != step_i:
                res["details"]["fid_milestone"] = {"skipped": True, "reason": "kid_stale", "kid_step": kid_last.get("step", None)}
                return

            kid_now = kid_last.get("kid", None)
            prev_best = kid_last.get("best_before", None)

            # decide gating from rel improvement stored by run_kid
            if prev_best is None:
                should_run = True  # first ever KID
            else:
                rel_improve = kid_last.get("rel_improve_pct", 0.0)
                should_run = float(rel_improve) >= gate

        # Persist best KID across calls to make gating stateful
        best_file = state_dir / "kid_best.json"
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
