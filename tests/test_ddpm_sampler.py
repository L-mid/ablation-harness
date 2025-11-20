"""Tests the sampler's nfe implementation is sound."""

from unittest import mock

import torch

from ablation_harness.tasks.diffusion.samplers.ddpm import DDPMSampler


class DummyModel(torch.nn.Module):
    def forward(self, x, t):
        # simple, but non-trivial: slightly denoise
        return 0.5 * x


def build_dummy_q(K=1000, device="cpu"):
    """Builds a schedule."""
    betas = torch.linspace(1e-4, 0.02, K, device=device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    posterior_log_var_clipped = torch.log(torch.clamp(betas, min=1e-5))
    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bar": alpha_bar,
        "posterior_log_var_clipped": posterior_log_var_clipped,
    }


def test_ddpm_sampler_respects_nfe():
    """Nfe means should be different."""
    device = "cpu"
    q = build_dummy_q(device=device)
    model = DummyModel().to(device)

    sampler = DDPMSampler(q, nfe=None)
    sampler.q = q
    sampler.device = device

    shape = (16, 3, 32, 32)

    sampler.nfe = 10
    x10 = sampler.sample(model, shape, seed=0)

    sampler.nfe = 50
    x50 = sampler.sample(model, shape, seed=0)

    # they should NOT be the same
    assert not torch.allclose(x10, x50)
    print("MSE:", torch.mean((x10 - x50) ** 2).item())  # MSE: 0.17308200895786285


def test_ddpm_sampler_uses_exact_nfe_steps():
    """Test sampler uses all nfe steps provided."""
    device = "cpu"
    q = build_dummy_q(device=device)
    model = DummyModel().to(device)

    sampler = DDPMSampler(q, nfe=None)
    sampler.q = q
    sampler.device = device
    sampler.nfe = 20

    with mock.patch.object(sampler, "step", wraps=sampler.step) as step_mock:
        sampler.sample(model, (4, 3, 32, 32), seed=0)

    assert step_mock.call_count == 20
