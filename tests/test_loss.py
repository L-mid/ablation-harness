import math

import pytest
import torch
import torch.nn.functional as F

from ablation_harness.tasks.diffusion.losses import (
    LossConfig,
    compute_eps_mse_with_weighting,
)


def test_constant_matches_plain_mse():
    """Generated plain mse and weighted mse have matching tensors."""
    B, C, H, W = 4, 3, 8, 8
    pred = torch.randn(B, C, H, W)
    target = torch.randn(B, C, H, W)

    K = 1000
    alphas_cumprod = torch.linspace(0.001, 0.999, steps=K)
    timesteps = torch.randint(0, K, (B,))

    plain_mse = F.mse_loss(pred, target, reduction="mean")

    cfg = LossConfig(weighting="constant", minsnr_gamma=5.0)
    loss = compute_eps_mse_with_weighting(
        model_pred=pred,
        target=target,
        timesteps=timesteps,
        alphas_cumprod=alphas_cumprod,
        loss_cfg=cfg,
    )

    assert torch.allclose(loss, plain_mse, atol=1e-6)


def test_minsnr_norm_keeps_mean_scale_near_one():
    """Test normalization on snr actually keeps near 1."""
    # Choose pred/target so per-sample MSE == 1
    B, C, H, W = 8, 3, 4, 4
    pred = torch.zeros(B, C, H, W)
    target = torch.ones(B, C, H, W)

    K = 1000
    alphas_cumprod = torch.linspace(0.001, 0.999, steps=K)
    timesteps = torch.randint(0, K, (B,))

    cfg = LossConfig(weighting="minsnr_norm", minsnr_gamma=5.0)
    loss = compute_eps_mse_with_weighting(
        model_pred=pred,
        target=target,
        timesteps=timesteps,
        alphas_cumprod=alphas_cumprod,
        loss_cfg=cfg,
    )

    # If MSE == 1 for each sample, the loss == mean(weight_t).
    # For minsnr_norm we want that mean ≈ 1.
    assert loss.item() == pytest.approx(1.0, rel=1e-3, abs=1e-3)


def _grad_norm(tensor: torch.Tensor) -> float:
    return tensor.grad.detach().norm().item()


def test_minsnr_norm_grad_norm_not_huge_vs_constant():
    """Test for grad explosion in snr normalizaion."""
    B, C, H, W = 4, 3, 8, 8
    # Treat model_pred as the "parameter" we differentiate w.r.t.
    pred = torch.nn.Parameter(torch.randn(B, C, H, W))
    target = torch.randn(B, C, H, W)

    K = 1000
    alphas_cumprod = torch.linspace(0.001, 0.999, steps=K)
    timesteps = torch.randint(0, K, (B,))

    # Constant weighting
    cfg_const = LossConfig(weighting="constant", minsnr_gamma=5.0)
    loss_const = compute_eps_mse_with_weighting(
        model_pred=pred,
        target=target,
        timesteps=timesteps,
        alphas_cumprod=alphas_cumprod,
        loss_cfg=cfg_const,
    )
    pred.grad = None
    loss_const.backward(retain_graph=True)
    gn_const = _grad_norm(pred)

    # Min-SNR normalized
    cfg_norm = LossConfig(weighting="minsnr_norm", minsnr_gamma=5.0)
    pred.grad = None
    loss_norm = compute_eps_mse_with_weighting(
        model_pred=pred,
        target=target,
        timesteps=timesteps,
        alphas_cumprod=alphas_cumprod,
        loss_cfg=cfg_norm,
    )
    loss_norm.backward()
    gn_norm = _grad_norm(pred)

    assert math.isfinite(gn_const)
    assert math.isfinite(gn_norm)
    # "LR doesn't explode": grad norm within a modest factor
    assert gn_norm < 5.0 * gn_const
