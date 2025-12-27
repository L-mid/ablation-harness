"""Tests the sampler's nfe implementation is sound."""

import pytest
import torch

from ablation_harness.tasks.diffusion.samplers.ddpm import DDPMSampler


@torch.no_grad()
def precompute_q(betas: torch.Tensor):
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    posterior_variance = torch.zeros_like(betas)
    posterior_variance[1:] = betas[1:] * (1 - alpha_bar[:-1]) / (1 - alpha_bar[1:])
    posterior_variance[0] = 1e-20
    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bar": alpha_bar,
        "sqrt_alpha": torch.sqrt(alphas),
        "sqrt_alpha_bar": torch.sqrt(alpha_bar),
        "sqrt_one_minus_alpha_bar": torch.sqrt(1 - alpha_bar),
        "posterior_log_var_clipped": torch.log(torch.clamp(posterior_variance, min=1e-20)),
    }


class ZeroModel(torch.nn.Module):
    def forward(self, x, t):
        return torch.zeros_like(x)


def test_ddpm_hard_errors_when_nfe_not_equal_K():
    device = torch.device("cpu")
    K = 10
    q = precompute_q(torch.full((K,), 1e-4, device=device))
    smp = DDPMSampler(q=q, nfe=5, device=device)  # nfe != K

    with pytest.raises(ValueError, match=r"does not support nfe != K"):
        smp.sample(ZeroModel(), (2, 3, 8, 8), seed=0)


def test_ddpm_full_K_sampling_runs_and_returns_finite():
    device = torch.device("cpu")
    K = 10
    q = precompute_q(torch.full((K,), 1e-4, device=device))
    smp = DDPMSampler(q=q, nfe=K, device=device)  # ok

    x = smp.sample(ZeroModel(), (2, 3, 8, 8), seed=0)
    assert x.shape == (2, 3, 8, 8)
    assert torch.isfinite(x).all()


def test_ddpm_step_t0_is_deterministic_no_noise():
    device = torch.device("cpu")
    K = 10
    q = precompute_q(torch.full((K,), 1e-4, device=device))
    smp = DDPMSampler(q=q, nfe=K, device=device)

    model = ZeroModel()
    x_t = torch.randn(4, 3, 8, 8, device=device)
    t0 = torch.zeros((4,), device=device, dtype=torch.long)

    out1 = smp.step(model, x_t, t0)
    out2 = smp.step(model, x_t, t0)

    # exact equality is fine here because t=0 adds zero noise by construction
    assert torch.allclose(out1, out2, atol=0.0, rtol=0.0)
