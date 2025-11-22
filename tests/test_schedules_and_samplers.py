"""Tests for schedules (cosine + linear)."""

import types

import torch

from ablation_harness.tasks.diffusion.samplers.ddim import DDIMSampler
from ablation_harness.tasks.diffusion.samplers.ddpm import DDPMSampler
from ablation_harness.tasks.diffusion.schedule import (
    get_beta_schedule,
    make_t_schedule,
    precompute_q,
)


class DummyModel(torch.nn.Module):
    def forward(self, x, t):
        """Simple, deterministic "noise" prediction: zeros."""
        # Good enough to exercise all schedule/sampler math.
        return torch.zeros_like(x)


def test_beta_schedule_in_range_and_monotone():
    """
    Check:
      - betas shape and (0, 1) range for both linear + cosine
      - alpha_bar in (0, 1], strictly decreasing in t
    """

    K = 1000
    for kind in ("linear", "cosine"):
        betas = get_beta_schedule(kind, K, device="cpu")
        assert betas.shape == (K,)
        assert torch.all(betas > 0)
        assert torch.all(betas < 1)

        q = precompute_q(betas)
        alpha_bar = q["alpha_bar"]

        # alpha_bar should be in (0, 1] and monotonically decreasing
        assert alpha_bar.shape == (K,)
        assert alpha_bar[0] <= 1.0 + 1e-7
        assert alpha_bar[0] > alpha_bar[-1]
        assert torch.all(alpha_bar > 0.0)

        diff = alpha_bar[1:] - alpha_bar[:-1]
        # non-increasing; allow tiny numerical noise
        assert torch.all(diff <= 1e-6)


def test_snr_from_q_is_monotone_and_consistent():
    """
    Check:
      - SNR(t) = alpha_bar / (1 - alpha_bar) is monotonically decreasing
      - SNR from alpha_bar matches SNR computed via sqrt terms
    """
    K = 1000
    for kind in ("linear", "cosine"):
        betas = get_beta_schedule(kind, K, device="cpu")
        q = precompute_q(betas)
        alpha_bar = q["alpha_bar"]

        # SNR(t) = alpha_bar / (1 - alpha_bar)
        snr = alpha_bar / (1.0 - alpha_bar + 1e-8)

        # Monotonically decreasing as diffusion proceeds
        assert snr[0] > snr[-1]
        diff = snr[1:] - snr[:-1]
        assert torch.all(diff <= 1e-5)

        # Consistency with sqrt terms:
        # sqrt_alpha_bar^2 = alpha_bar, sqrt_one_minus_alpha_bar^2 = 1 - alpha_bar
        snr_from_sqrt = (q["sqrt_alpha_bar"] ** 2) / (q["sqrt_one_minus_alpha_bar"] ** 2 + 1e-8)
        assert torch.allclose(snr, snr_from_sqrt, atol=1e-6, rtol=1e-6)


def test_sampler_indexing_and_sampling_smoke():
    """
    Focused sampler test:

    - make_t_schedule: endpoints, range, monotone
    - DDPMSampler: all t in [0, K-1], covers endpoints, descending schedule
    - DDIMSampler: t/t_prev indexing in [0, K-1], and final t_prev == -1 sentinel
    - E2E smoke: real DDPM + DDIM sampling runs without error and returns finite tensors
    """
    device = "cpu"
    K = 20
    nfe = 5
    B, C, H, W = 2, 3, 8, 8

    betas = get_beta_schedule("linear", K, device=device)
    q = precompute_q(betas)
    model = DummyModel()

    # --- make_t_schedule sanity ---
    t_sched = make_t_schedule(K, nfe, device=device)
    assert t_sched.shape == (nfe,)
    assert int(t_sched[0]) == K - 1
    assert int(t_sched[-1]) == 0
    assert torch.all(t_sched >= 0)
    assert torch.all(t_sched <= K - 1)
    assert torch.all(t_sched[:-1] >= t_sched[1:])

    # ---------- DDPMSampler: indexing pattern ----------
    ddpm = DDPMSampler.__new__(DDPMSampler)
    ddpm.q = q
    ddpm.device = device
    ddpm.nfe = nfe

    recorded_t = []

    def fake_ddpm_step(self, model, x_t, t, t_prev=None):
        """t is [B]; record the scalar value (just use first element)"""
        recorded_t.append(t[0].detach().cpu())
        return x_t  # don't change x, keep it cheap

    ddpm.step = types.MethodType(fake_ddpm_step, ddpm)
    out = ddpm.sample(model, shape=(B, C, H, W), seed=123)
    assert out.shape == (B, C, H, W)

    t_tensor = torch.stack(recorded_t)  # [nfe]
    assert t_tensor.min() >= 0
    assert t_tensor.max() <= K - 1
    assert int(t_tensor[0].item()) == K - 1
    assert int(t_tensor[-1].item()) == 0
    # descending in time
    assert torch.all(t_tensor[:-1] >= t_tensor[1:])

    # Real DDPM sampling smoke-test (no patching)
    ddpm_real = DDPMSampler.__new__(DDPMSampler)
    ddpm_real.q = q
    ddpm_real.device = device
    ddpm_real.nfe = nfe
    out_real = ddpm_real.sample(model, shape=(B, C, H, W), seed=321)
    assert out_real.shape == (B, C, H, W)
    assert torch.isfinite(out_real).all()

    # ---------- DDIMSampler: indexing pattern + -1 sentinel ----------
    ddim = DDIMSampler.__new__(DDIMSampler)
    ddim.q = q
    ddim.device = device
    ddim.nfe = nfe
    ddim.eta = 0.0

    recorded_t = []
    recorded_prev = []

    def fake_ddim_step(self, model, x_t, t, t_prev):
        recorded_t.append(t[0].detach().cpu())
        recorded_prev.append(t_prev.detach().cpu())
        return x_t

    ddim.step = types.MethodType(fake_ddim_step, ddim)
    out = ddim.sample(model, shape=(B, C, H, W), seed=456)
    assert out.shape == (B, C, H, W)

    t_tensor = torch.stack(recorded_t)
    prev_tensor = torch.stack(recorded_prev)

    # t indices are in range and descending
    assert t_tensor.min() >= 0
    assert t_tensor.max() <= K - 1
    assert int(t_tensor[0].item()) == K - 1
    assert int(t_tensor[-1].item()) == 0
    assert torch.all(t_tensor[:-1] >= t_tensor[1:])

    # All intermediate t_prev are valid indices; final one is the -1 sentinel
    assert torch.all(prev_tensor[:-1] >= 0)
    assert torch.all(prev_tensor[:-1] <= K - 1)
    assert int(prev_tensor[-1].item()) == -1

    # Real DDIM sampling smoke-test (no patching)
    ddim_real = DDIMSampler.__new__(DDIMSampler)
    ddim_real.q = q
    ddim_real.device = device
    ddim_real.nfe = nfe
    ddim_real.eta = 0.0
    out_real = ddim_real.sample(model, shape=(B, C, H, W), seed=789)
    assert out_real.shape == (B, C, H, W)
    assert torch.isfinite(out_real).all()
