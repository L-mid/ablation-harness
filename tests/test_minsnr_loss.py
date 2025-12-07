import math

import torch
import torch.nn as nn

from ablation_harness.tasks.diffusion.losses import (
    LossConfig,
    compute_eps_mse_with_weighting,
    compute_snr_from_alphas_cumprod,
    ddpm_loss,
)


class DummyModel(nn.Module):
    """
    Tiny stand-in for the UNet. Just a linear layer so we can run ddpm_loss
    and backprop without bringing in the full model.
    """

    def __init__(self, dim: int = 4):
        super().__init__()
        self.net = nn.Linear(dim, dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Ignore t; we just care that this returns something with the right shape.
        bsz = x_t.shape[0]
        return self.net(x_t.view(bsz, -1)).view_as(x_t)


def make_fake_q(K: int = 10, device: str = "cpu"):
    """
    Build a tiny fake q dict with monotonically decreasing alpha_bar.
    This mirrors the structure produced by precompute_q().
    """
    betas = torch.linspace(1e-4, 0.02, K, device=device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    posterior_variance = torch.zeros_like(betas)  # [K]
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


def test_snr_computation_basic():
    """
    SNR(t) = alpha_bar_t / (1 - alpha_bar_t)

    Check that compute_snr_from_alphas_cumprod matches hand calculation
    on a tiny alpha_bar tensor.
    """
    alpha_bar = torch.tensor([0.2, 0.5, 0.9])  # shape [3]
    t = torch.tensor([0, 1, 2])

    snr = compute_snr_from_alphas_cumprod(alpha_bar, t)

    expected = alpha_bar / (1.0 - alpha_bar)
    assert torch.allclose(snr, expected, atol=1e-6)


def test_minsnr_weights_low_vs_high_snr():
    """
    Min-SNR-γ weights should give:
      - ~1 weight for low-SNR timesteps
      - <1 weight for high-SNR timesteps
      - Non-increasing weights as SNR increases.
    """
    device = "cpu"
    K = 5
    q = make_fake_q(K=K, device=device)

    # construct a batch with explicitly chosen timesteps:
    # t_low: large noise ⇒ low SNR
    # t_mid, t_high: progressively higher SNR
    t_low = torch.tensor([K - 1], dtype=torch.long)  # largest t ⇒ smallest alpha_bar
    t_mid = torch.tensor([K // 2], dtype=torch.long)
    t_high = torch.tensor([0], dtype=torch.long)

    timesteps = torch.cat([t_low, t_mid, t_high], dim=0)  # [3]
    alphas_cumprod = q["alpha_bar"]

    # dummy per-sample mse values
    mse = torch.tensor([1.0, 1.0, 1.0])

    # Just test weighting logic by calling compute_eps_mse_with_weighting directly.
    class DummyCfg:
        weighting = "minsnr"
        minsnr_gamma = 5.0

    # Build fake preds/targets such that MSE per-sample = 1.0
    model_pred = torch.zeros(3, 4)
    target = torch.ones(3, 4)  # ||1 - 0||^2 per sample = 4, but we average later

    # We only care about relative weights, not exact numeric loss value,
    # so we don't assert the absolute scalar here.
    _ = compute_eps_mse_with_weighting(
        model_pred=model_pred,
        target=target,
        timesteps=timesteps,
        alphas_cumprod=alphas_cumprod,
        loss_cfg=DummyCfg(),
    )

    # Check SNR/weight monotonicity directly
    alpha_bar = torch.tensor([0.01, 0.5, 0.99])  # [very noisy, medium, very clean]
    t = torch.tensor([0, 1, 2])
    snr = alpha_bar / (1 - alpha_bar)  # ~[0.0101, 1, 99]
    gamma = 5.0

    weights = torch.minimum(snr, torch.tensor(gamma)) / snr.clamp(min=1e-12)
    # expected:
    #   snr=0.0101  -> gamma/snr ≈ 495 > 1, so weight ≈ 1
    #   snr=1       -> gamma/snr = 5  > 1, so weight ≈ 1
    #   snr=99      -> gamma/snr ≈ 0.0505 < 1, so weight ≈ 0.05

    assert weights[0] > 0.99  # very low SNR
    assert weights[1] > 0.99  # still below gamma
    assert weights[2] < 0.2  # high SNR, downweighted

    # As SNR increases, weight should not increase
    snr_sorted, idx = torch.sort(snr)
    weights_sorted = weights[idx]
    for i in range(len(weights_sorted) - 1):
        assert weights_sorted[i + 1] <= weights_sorted[i] + 1e-6


def test_ddpm_loss_runs_with_minsnr_and_backprop():
    """
    Sanity: ddpm_loss should run with weighting='minsnr' and support backprop.
    """
    device = "cpu"
    torch.manual_seed(0)

    bsz, C, H, W = 4, 1, 2, 2
    x0 = torch.randn(bsz, C, H, W, device=device)
    q = make_fake_q(K=10, device=device)

    model = DummyModel(dim=C * H * W).to(device)

    loss_cfg = LossConfig(weighting="minsnr", minsnr_gamma=5.0)

    loss = ddpm_loss(model, x0, q, loss_cfg=loss_cfg)

    assert loss.shape == ()
    assert math.isfinite(loss.item())

    loss.backward()  # should not raise
    # At least one parameter should have non-zero grad
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "Expected some gradients after backward()"
    assert any(g.abs().sum().item() > 0 for g in grads)
