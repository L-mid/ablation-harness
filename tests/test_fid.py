from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Subset

from ablation_harness.data import build_cifar10  # wherever build_cifar10 lives
from ablation_harness.eval.generative import _fid_from_stats, _inception_activations


def _fid_from_stats_reference(mu_gen, sigma_gen, mu_ref, sigma_ref, eps: float = 1e-6) -> float:
    """
    This fid from stats is a test of the math itself local to here. 
    Compared to with repo implementation.
    """
    mu_gen = np.atleast_1d(mu_gen).astype(np.float64)
    mu_ref = np.atleast_1d(mu_ref).astype(np.float64)
    sigma_gen = np.atleast_2d(sigma_gen).astype(np.float64)
    sigma_ref = np.atleast_2d(sigma_ref).astype(np.float64)

    sigma_gen = (sigma_gen + sigma_gen.T) / 2.0
    sigma_ref = (sigma_ref + sigma_ref.T) / 2.0

    eye = np.eye(sigma_gen.shape[0], dtype=np.float64)
    sigma_gen = sigma_gen + eps * eye
    sigma_ref = sigma_ref + eps * eye

    diff = mu_gen - mu_ref

    w, v = np.linalg.eigh(sigma_gen)
    w = np.clip(w, 0.0, None)
    sqrt_g = (v * np.sqrt(w)) @ v.T

    A = sqrt_g @ sigma_ref @ sqrt_g
    A = (A + A.T) / 2.0
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


FID_STATS_PATH = Path("stats/cifar10_inception_train.npz")


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
    torch.cuda.empty_cache()  # optional, helps after prior GPU tests
 
    for xb, _ in dl:
        xb = xb.to(device, non_blocking=True)  # [-1,1]
        xb = (xb.clamp(-1, 1) + 1.0) / 2.0  # -> [0,1]
        feats.append(_inception_activations(xb, device, batch_size=64).cpu())

    feats = torch.cat(feats, dim=0)  # CPU tensor now
    mu_real = feats.mean(axis=0)
    sigma_real = np.cov(feats, rowvar=False)

    # 4) Load reference stats
    data = np.load(FID_STATS_PATH)
    mu_ref = data["mu"]
    sigma_ref = data["sigma"]

    fid = _fid_from_stats(mu_real, sigma_real, mu_ref, sigma_ref)
    print("FID(real CIFAR via current pipeline, stats) =", fid)

    # no negative fid.
    assert fid >= -1e-3, f"FID should be nonnegative (numerical tol), got {fid}"
    # If stats were computed with this same pipeline, this should be small.
    assert fid < 10.0, f"FID(CIFAR vs stats) too large: {fid}"


# test these on cuda later

def test_fid_spd_mismatched_bases_matches_reference_and_nonnegative():
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




