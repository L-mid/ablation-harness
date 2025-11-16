"""
Tests the model is actually learning.
"""

import torch

from ablation_harness.tasks.diffusion.models.unet_cifar32 import UNetCifar32
from ablation_harness.tasks.diffusion.schedule import (
    ddpm_loss,
    get_beta_schedule,
    precompute_q,
    q_sample,
)


def _denoise_mse(model, x0, q, device):
    """Helper: sample one t, noising step, then decode x0_pred and return MSE(x0_pred, x0)."""
    K = q["betas"].numel()
    B = x0.size(0)
    t = torch.randint(0, K, (B,), device=device)
    x_t, eps = q_sample(x0, t, q)

    with torch.no_grad():
        eps_pred = model(x_t, t)
        sqrt_ab = q["sqrt_alpha_bar"][t].view(-1, 1, 1, 1)
        sqrt_om = q["sqrt_one_minus_alpha_bar"][t].view(-1, 1, 1, 1)
        x0_pred = (x_t - sqrt_om * eps_pred) / (sqrt_ab + 1e-8)
        mse = torch.mean((x0_pred - x0) ** 2)
    return mse


def test_unet_ddpm_loss_improves_denoising():
    """End-to-end check: UNet + ddpm_loss actually learns to denoise a toy batch."""
    device = torch.device("cpu")
    torch.manual_seed(0)

    # Small-ish schedule to keep test fast
    K = 128
    betas = get_beta_schedule("linear", K, device=device)
    q = precompute_q(betas)

    # Real model, just slightly slimmed for speed if you like
    model = UNetCifar32(
        in_channels=3,
        out_channels=3,
        base_channels=32,
        channel_mults=(1, 2, 2, 2),
        num_res_blocks=2,
        time_hidden=128,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Fixed toy batch (CIFAR-shape)
    B, C, H, W = 8, 3, 32, 32
    x0 = torch.randn(B, C, H, W, device=device)

    # Baseline denoising quality before training
    mse_before = _denoise_mse(model, x0, q, device)

    # Do a handful of ddpm_loss steps on the same batch
    n_steps = 30
    for _ in range(n_steps):
        loss = ddpm_loss(model, x0, q)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Denosing quality after a bit of training
    mse_after = _denoise_mse(model, x0, q, device)

    # We don't care about exact numbers, just that we've learned *something*
    assert mse_after < mse_before, f"mse_after={mse_after} not < mse_before={mse_before}"
