"""
Various FID insurance.


"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Subset

from ablation_harness.data import build_cifar10  # wherever build_cifar10 lives
from ablation_harness.eval.generative import (
    _fid_from_stats,
    _inception_activations,
    _make_psd,
)

FID_STATS_PATH = Path("stats/cifar10_inception_train.npz")
FID_STATS_SMALL_PATH = Path("stats/cifar10_inception_train_n2048_seed0.npz")


def _fid_from_stats_reference(mu_gen, sigma_gen, mu_ref, sigma_ref, eps: float = 1e-6) -> float:
    """
    This fid from stats is a test of the math itself local to here.
    Compared to with repo implementation.
    """
    mu_gen = np.atleast_1d(mu_gen).astype(np.float64)
    mu_ref = np.atleast_1d(mu_ref).astype(np.float64)
    sigma_gen = _make_psd(np.atleast_2d(sigma_gen), eps=eps)
    sigma_ref = _make_psd(np.atleast_2d(sigma_ref), eps=eps)

    diff = mu_gen - mu_ref

    w, v = np.linalg.eigh(sigma_gen)
    w = np.clip(w, 0.0, None)
    sqrt_g = (v * np.sqrt(w)) @ v.T

    A = sqrt_g @ sigma_ref @ sqrt_g
    A = 0.5 * (A + A.T)
    wA = np.linalg.eigvalsh(A)
    wA = np.clip(wA, 0.0, None)
    tr_covmean = float(np.sum(np.sqrt(wA)))

    fid = float(diff.dot(diff) + np.trace(sigma_gen) + np.trace(sigma_ref) - 2.0 * tr_covmean)
    if fid < 0.0 and fid > -1e-6:
        fid = 0.0
    return fid


def _make_fake_generator_images(batch_size: int = 16) -> torch.Tensor:
    """
    Make a fake batch of generator outputs in [-1, 1], then map to [0, 1]
    exactly like the real FID path does.

    This keeps the test very close to the real diffusion→FID pipeline.
    """
    # pretend these came from your diffusion model ([-1, 1])
    x = torch.rand(batch_size, 3, 32, 32) * 2.0 - 1.0  # [-1, 1]

    # real FID path in generative.py does:
    # imgs = (imgs.clamp(-1, 1) + 1.0) / 2.0
    x = (x.clamp(-1.0, 1.0) + 1.0) / 2.0  # [0, 1]
    return x


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_inception_activations_cpu_vs_cuda_close():
    """
    For the same input images (in [0,1]), Inception activations on CPU and CUDA
    should be numerically very close.

    This is the core guarantee that lets a single stats file be shared
    across devices, as long as the environment (torch/torchvision + weights)
    is the same.
    """
    batch_size = 8
    imgs = _make_fake_generator_images(batch_size)

    device_cpu = torch.device("cpu")
    device_cuda = torch.device("cuda")

    feats_cpu = _inception_activations(imgs.to(device_cpu), device_cpu)
    feats_cuda = _inception_activations(imgs.to(device_cuda), device_cuda)

    assert feats_cpu.shape == feats_cuda.shape

    diff = np.abs(feats_cpu - feats_cuda)
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())

    # These tolerances are intentionally a bit loose to allow for tiny
    # numerical differences in interpolation / matmul implementations.
    assert max_abs < 5e-4, f"max abs diff too large: {max_abs}"
    assert mean_abs < 5e-5, f"mean abs diff too large: {mean_abs}"


def test_fid_of_self_is_zeroish():
    """
    A set of features compared to itself should give FID ~ 0.

    This guards the basic FID math: if this ever fails, something in
    _fid_from_stats or _sqrtm_psd is badly broken.
    """
    batch_size = 32
    imgs = _make_fake_generator_images(batch_size)

    device = torch.device("cpu")
    feats = _inception_activations(imgs.to(device), device)

    mu = feats.mean(axis=0)
    sigma = np.cov(feats, rowvar=False)

    fid = _fid_from_stats(mu, sigma, mu, sigma)

    assert abs(fid) < 1e-4, f"FID(self, self) should be ~0, got {fid}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fid_cpu_vs_cuda_features_near_zero():
    """
    FID between CPU and CUDA Inception features for the *same images*
    should be ~0.

    This is a stronger, FID-level version of test_inception_activations_cpu_vs_cuda_close.
    It will catch more subtle drifts (e.g. if _inception_activations changes
    preprocessing on one device).
    """
    batch_size = 32
    imgs = _make_fake_generator_images(batch_size)

    device_cpu = torch.device("cpu")
    device_cuda = torch.device("cuda")

    feats_cpu = _inception_activations(imgs.to(device_cpu), device_cpu)
    feats_cuda = _inception_activations(imgs.to(device_cuda), device_cuda)

    mu_cpu = feats_cpu.mean(axis=0)
    sigma_cpu = np.cov(feats_cpu, rowvar=False)

    mu_cuda = feats_cuda.mean(axis=0)
    sigma_cuda = np.cov(feats_cuda, rowvar=False)

    fid = _fid_from_stats(mu_cpu, sigma_cpu, mu_cuda, sigma_cuda)

    # Because feats_cpu ~= feats_cuda elementwise, mu/sigma are also very close
    # and FID should be near 0. Small numerical noise is fine.
    assert abs(fid) < 2.0, f"FID(cpu, cuda) too large: {fid}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not Path(FID_STATS_PATH).exists(), reason="FID stats file not found")
def test_fid_cifar_train_vs_stats_not_insane():
    device = torch.device("cuda")

    # 1) Get real CIFAR-10 images in [-1,1]
    tr, _ = build_cifar10(subset=5000)  # or None and just slice
    n = 2048
    dl = DataLoader(
        Subset(tr, range(n)),
        batch_size=64,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    feats = []
    for xb, _ in dl:
        xb = xb.to(device, non_blocking=True)
        xb = (xb.clamp(-1, 1) + 1.0) / 2.0

        f_np = _inception_activations(xb, device, batch_size=64)  # np.ndarray [b, D]
        f_t = torch.from_numpy(f_np)  # CPU torch.Tensor (shares memory)
        feats.append(f_t)

    feats = torch.cat(feats, dim=0).to(dtype=torch.float64)  # already CPU
    mu_real_t = feats.mean(dim=0)
    xc = feats - mu_real_t
    sigma_real_t = (xc.T @ xc) / (feats.shape[0] - 1)
    sigma_real_t = 0.5 * (sigma_real_t + sigma_real_t.T)

    mu_real = mu_real_t.numpy()
    sigma_real = sigma_real_t.numpy()

    data = np.load(FID_STATS_PATH)
    mu_ref = data["mu"].astype(np.float64)
    sigma_ref = data["sigma"].astype(np.float64)

    fid = _fid_from_stats(mu_real, sigma_real, mu_ref, sigma_ref)
    print("FID(real CIFAR via current pipeline, stats) =", fid)

    # no negative fid.
    assert fid >= -1e-3, f"FID should be nonnegative (numerical tol), got {fid}"
    # If stats were computed with this same pipeline, this should be small.
    assert fid < 20.0, f"FID(CIFAR vs stats) too large: {fid}"


@pytest.mark.skipif(os.environ.get("RUN_FID_TREND") != "1", reason="set RUN_FID_TREND=1 to run")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not FID_STATS_PATH.exists(), reason="Full FID stats file not found")
def test_fid_cifar_vs_full_stats_trend_improves_with_n():
    """
    Tests two fids against each other to check for improvement.
    Run this explicitly with: RUN_FID_TREND=1 pytest -q tests/test_fid.py::test_fid_cifar_vs_full_stats_trend_improves_with_n
    """
    device = torch.device("cuda")
    data = np.load(FID_STATS_PATH)
    mu_ref = data["mu"].astype(np.float64)
    sigma_ref = data["sigma"].astype(np.float64)

    tr, _ = build_cifar10(subset=None)

    def fid_for_n(n: int, seed: int = 0) -> float:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(tr), size=n, replace=False).astype(np.int64)
        dl = DataLoader(Subset(tr, idx.tolist()), batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

        feats = []
        for xb, _ in dl:
            xb = xb.to(device, non_blocking=True)
            xb = (xb.clamp(-1, 1) + 1.0) / 2.0
            feats.append(torch.from_numpy(_inception_activations(xb, device, batch_size=64)))

        feats = torch.cat(feats, dim=0).to(dtype=torch.float64)
        mu = feats.mean(dim=0).numpy()
        xc = feats - torch.from_numpy(mu).to(feats)
        sigma = ((xc.T @ xc) / (feats.shape[0] - 1)).numpy()
        sigma = 0.5 * (sigma + sigma.T)

        return float(_fid_from_stats(mu, sigma, mu_ref, sigma_ref))

    fid_small = fid_for_n(2048, seed=0)
    fid_big = fid_for_n(8192, seed=0)

    # allow tiny numerical weirdness slack, but overall should improve
    assert fid_big <= fid_small + 0.5, (fid_small, fid_big)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not FID_STATS_SMALL_PATH.exists(), reason="Small FID stats file not found (make using make_subset_fid_stats.py)")
def test_fid_cifar_matches_small_stats_near_zero():
    """
    Small sample generated self to self fid stats check.
    """
    device = torch.device("cuda")

    data = np.load(FID_STATS_SMALL_PATH, allow_pickle=True)
    mu_ref = data["mu"].astype(np.float64)
    sigma_ref = data["sigma"].astype(np.float64)
    idx = data["idx"].astype(np.int64)

    tr, _ = build_cifar10(subset=None)
    dl = DataLoader(Subset(tr, idx.tolist()), batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    feats = []
    for xb, _ in dl:
        xb = xb.to(device, non_blocking=True)
        xb = (xb.clamp(-1, 1) + 1.0) / 2.0  # same mapping as real eval :contentReference[oaicite:3]{index=3}
        f_np = _inception_activations(xb, device, batch_size=64)
        feats.append(torch.from_numpy(f_np))

    feats = torch.cat(feats, dim=0).to(dtype=torch.float64)
    mu = feats.mean(dim=0).numpy()
    xc = feats - torch.from_numpy(mu).to(feats)
    sigma = ((xc.T @ xc) / (feats.shape[0] - 1)).numpy()
    sigma = 0.5 * (sigma + sigma.T)

    fid = _fid_from_stats(mu, sigma, mu_ref, sigma_ref)
    assert fid >= -1e-3
    assert fid < 1.0, f"FID should be ~0 against matching small stats, got {fid}"


def test_fid_spd_mismatched_bases_matches_reference_and_nonnegative():
    """Math test."""
    rng = np.random.default_rng(0)
    d = 64

    vals1 = np.linspace(0.5, 5.0, d)
    vals2 = np.linspace(1.0, 3.0, d)[::-1]

    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    sigma1 = np.diag(vals1)
    sigma2 = Q @ np.diag(vals2) @ Q.T

    mu1 = np.zeros(d, dtype=np.float64)
    mu2 = np.zeros(d, dtype=np.float64)

    fid = _fid_from_stats(mu1, sigma1, mu2, sigma2)
    fid_ref = _fid_from_stats_reference(mu1, sigma1, mu2, sigma2)

    assert np.isfinite(fid)
    assert fid >= -1e-3
    assert abs(fid - fid_ref) < 1e-6, f"_fid_from_stats drifted from reference: {fid} vs {fid_ref}"


# Large test for emergencies


@pytest.mark.skipif(os.environ.get("RUN_FID_ABS_FULL") != "1", reason="set RUN_FID_ABS_FULL=1 to run (slow/expensive absolute check)")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not FID_STATS_PATH.exists(), reason="FID stats file not found")
def test_fid_cifar_train_vs_full_stats_absolute():
    """
    Absolute check:
      Compute μ/Σ over the full CIFAR-10 train set using the *current pipeline*,
      and compare to stats/cifar10_inception_train.npz.

    If the stats file was built with the same Inception + preprocess pipeline,
    FID should be ~0 (within small numeric tolerance).
    """
    device = torch.device("cuda")

    # Reference stats (should be full-train stats made by your stats tool)
    data = np.load(FID_STATS_PATH)
    mu_ref = data["mu"].astype(np.float64)
    sigma_ref = data["sigma"].astype(np.float64)

    # Full train set from current pipeline: build_cifar10 gives [-1,1]
    tr, _ = build_cifar10(subset=None)

    dl = DataLoader(
        tr,
        batch_size=256,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # We accumulate:
    #   sum(f) on CPU (float64) for μ
    #   sum(f^T f) on GPU (float32) for Σ (fast), then convert to float64 at the end
    sum_cpu = None
    xtx_gpu = None
    n_total = 0

    for xb, _ in dl:
        xb = xb.to(device, non_blocking=True)  # [-1,1]
        xb = (xb.clamp(-1, 1) + 1.0) / 2.0  # -> [0,1]

        f_np = _inception_activations(xb, device, batch_size=64)  # numpy [b, d], CPU
        f_np = np.asarray(f_np)

        b, d = f_np.shape
        if sum_cpu is None:
            sum_cpu = np.zeros((d,), dtype=np.float64)
        sum_cpu += f_np.sum(axis=0, dtype=np.float64)

        f_gpu = torch.from_numpy(f_np).to(device=device, dtype=torch.float32, non_blocking=True)
        if xtx_gpu is None:
            xtx_gpu = torch.zeros((d, d), device=device, dtype=torch.float32)

        xtx_gpu += f_gpu.T @ f_gpu
        n_total += b

    assert n_total == len(tr)

    mu = (sum_cpu / n_total).astype(np.float64)
    mu_gpu64 = torch.from_numpy(mu).to(device=device, dtype=torch.float64)
    xtx_gpu64 = xtx_gpu.to(dtype=torch.float64)

    # Cov = (X^T X - N * mu mu^T) / (N - 1)
    sigma_gpu64 = (xtx_gpu64 - n_total * (mu_gpu64[:, None] @ mu_gpu64[None, :])) / (n_total - 1)
    sigma = sigma_gpu64.detach().cpu().numpy().astype(np.float64)
    sigma = 0.5 * (sigma + sigma.T)

    fid = float(_fid_from_stats(mu, sigma, mu_ref, sigma_ref))

    # Numerical sanity
    assert fid >= -1e-3, f"FID should be nonnegative (numerical tol), got {fid}"

    # Absolute threshold: should be very small if stats truly match this pipeline
    # (Use 1.0 as a generous “catch real pipeline mismatch” bound.)
    assert fid < 1.0, f"FID(full CIFAR train vs saved stats) too large: {fid}"
