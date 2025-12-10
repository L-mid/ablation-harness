import numpy as np
import pytest
import torch

from ablation_harness.eval.generative import _fid_from_stats, _inception_activations


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
