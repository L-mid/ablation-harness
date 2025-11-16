"""Tests the sampling path is valid."""

import torch

from ablation_harness.tasks.diffusion.samplers import DDPMSampler
from ablation_harness.tasks.diffusion.schedule import (
    get_beta_schedule,
    precompute_q,
    q_sample,
)


class GroundTruthEpsModel(torch.nn.Module):
    """
    Tiny model that always returns the *true* eps used in q_sample.
    This lets us test the sampler math in isolation from learning.
    """

    def __init__(self, eps):
        super().__init__()
        # store eps as a buffer so it moves to the right device
        self.register_buffer("eps", eps)

    def forward(self, x, t):
        # Ignore x and t – we pretend we are a perfect ε-predictor.
        # Shape is already [B, C, H, W].
        return self.eps


def test_ddpm_step_reduces_mse_to_x0():
    """Integration test: with perfect ε, one DDPM step moves x closer to x0."""
    device = torch.device("cpu")
    torch.manual_seed(0)

    # Small schedule for speed
    K = 128
    betas = get_beta_schedule("linear", K, device=device)
    q = precompute_q(betas)

    B, C, H, W = 8, 3, 32, 32
    x0 = torch.randn(B, C, H, W, device=device)

    # Pick a mid-range timestep
    t_scalar = K // 2
    t = torch.full((B,), t_scalar, device=device, dtype=torch.long)

    # Forward diffusion: x_t = sqrt(ā_t)x0 + sqrt(1-ā_t)ε
    x_t, eps = q_sample(x0, t, q)

    # Perfect ε-predictor model
    model = GroundTruthEpsModel(eps).to(device)

    # Use the real DDPMSampler.step
    sampler = DDPMSampler(q=q, nfe=1, device=device)
    x_prev = sampler.step(model, x_t, t)  # t_prev unused for DDPM

    mse_start = torch.mean((x_t - x0) ** 2)
    mse_prev = torch.mean((x_prev - x0) ** 2)

    assert mse_prev < mse_start, f"DDPM step did not move towards x0: {mse_prev} !< {mse_start}"
