import math

import pytest
import torch

from ablation_harness.tasks.diffusion.samplers.ddim import DDIMSampler, make_t_schedule
from ablation_harness.tasks.diffusion.schedule import precompute_q


class ZeroModel(torch.nn.Module):
    def forward(self, x, t):
        return torch.zeros_like(x)


class ConstEpsModel(torch.nn.Module):
    def __init__(self, c: float = 0.123):
        super().__init__()
        self.c = float(c)

    def forward(self, x, t):
        return torch.full_like(x, self.c)


def _build_q_const_beta(K: int, beta: float, device: torch.device):
    betas = torch.full((K,), float(beta), device=device, dtype=torch.float32)
    return precompute_q(betas)


def test_ddim_step_zero_model_eta0_matches_scale():
    device = torch.device("cpu")
    K = 1000

    betas = torch.full((K,), 1e-4)

    q = precompute_q(betas)  # however you build q

    B = 4
    x_t = torch.randn(B, 3, 32, 32, device=device)
    t = torch.full((B,), 200, dtype=torch.long)
    t_prev = torch.full((B,), 100, dtype=torch.long)

    smp = DDIMSampler(q=q, nfe=20, eta=0.0, device=device)

    x_prev = smp.step(ZeroModel().to(device), x_t, t, t_prev)

    sqrt_ab_t = torch.sqrt(q["alpha_bar"][t]).min().item()
    assert sqrt_ab_t > 1e-4

    ab_t = q["alpha_bar"][t].view(B, 1, 1, 1)

    ab_t = q["alpha_bar"][t].view(B, 1, 1, 1)
    x0_ideal = x_t / torch.sqrt(ab_t)
    clip_frac = (x0_ideal.abs() > 1).float().mean().item()
    print("clip_frac:", clip_frac)

    ab_prev = q["alpha_bar"][t_prev].view(B, 1, 1, 1)

    x0 = (x_t / torch.sqrt(ab_t)).clamp(-1, 1)

    expected = torch.sqrt(ab_prev) * x0

    max_abs = (x_prev - expected).abs().max().item()
    assert max_abs < 5e-5


def test_make_t_schedule_spans_full_range():
    """Would catch accidental K=nfe mistakes in schedule construction."""
    K, nfe = 1000, 25
    t = make_t_schedule(K, nfe, device="cpu")
    assert t.numel() == nfe
    assert int(t[0].item()) == K - 1
    assert int(t[-1].item()) == 0
    # monotone non-increasing
    assert bool(torch.all(t[:-1] >= t[1:]))


def test_ddim_terminal_prev_batched_uses_abar_prev_1():
    """
    Regression for the exact bug:
    if t_prev is a (B,) tensor containing -1, code must use a_bar_prev=1.0,
    not alpha_bar[-1].
    """
    device = torch.device("cpu")
    B, C, H, W = 8, 3, 8, 8

    # Make alpha_bar[-1] extremely small so the bug is loud:
    # alpha_bar[-1] = (1-beta)^K ~ 0 when beta=0.1, K=100.
    q = _build_q_const_beta(K=100, beta=0.1, device=device)
    sampler = DDIMSampler(q=q, nfe=10, eta=0.0, device=device)

    model = ConstEpsModel(0.5).to(device).eval()

    x_t = torch.randn(B, C, H, W, device=device) * 0.1

    # terminal step: t=0, t_prev=-1 (batched)
    t = torch.zeros((B,), device=device, dtype=torch.long)
    eps = model(x_t, t).float()

    t_prev = torch.full((B,), -1, device=device, dtype=torch.long)

    x_prev = sampler.step(model, x_t, t, t_prev)

    # If a_bar_prev=1.0, then c = sqrt(1 - a_bar_prev - sigma^2) = 0 (eta=0),
    # and x_prev must equal x0_pred. With t=0, a_bar_t ~ 1-beta, so x0_pred ~ x_t/sqrt(a_bar_t).
    # BUT many implementations also clamp x0 to [-1,1]; we keep x small to avoid that.
    a_bar_t = q["alpha_bar"][0].item()
    expected = (x_t - math.sqrt(1.0 - a_bar_t) * eps) / math.sqrt(a_bar_t)

    assert torch.isfinite(x_prev).all()
    assert torch.allclose(x_prev, expected, rtol=1e-4, atol=1e-4), "Likely indexed alpha_bar[-1] when t_prev was batched -1, " "or used autocast fp16 math incorrectly."


def test_ddim_skip_step_matches_closed_form_when_eps_zero():
    """
    Regression: DDIM must use alpha_bar[t_prev] (skipped) not DDPM adjacent-step posterior stuff.
    With eps=0 and eta=0, update simplifies to:
      x_prev = sqrt(alpha_bar_prev / alpha_bar_t) * x_t
    (assuming no x0 clamping is hit; keep magnitudes small.)
    """
    device = torch.device("cpu")
    B, C, H, W = 4, 3, 8, 8

    q = _build_q_const_beta(K=200, beta=1e-3, device=device)
    sampler = DDIMSampler(q=q, nfe=10, eta=0.0, device=device)

    model = ZeroModel().to(device).eval()
    x_t = torch.randn(B, C, H, W, device=device) * 0.05

    t_int = 150
    tprev_int = 50  # not t-1, a skipped step
    t = torch.full((B,), t_int, device=device, dtype=torch.long)
    t_prev = torch.full((B,), tprev_int, device=device, dtype=torch.long)

    x_prev = sampler.step(model, x_t, t, t_prev)

    abar_t = q["alpha_bar"][t_int].item()
    abar_prev = q["alpha_bar"][tprev_int].item()
    expected = x_t * math.sqrt(abar_prev / abar_t)

    assert torch.isfinite(x_prev).all()
    assert torch.allclose(x_prev, expected, rtol=1e-4, atol=1e-4), "DDIM skip-step update does not match closed form; likely using DDPM adjacent-step terms " "or wrong a_bar_prev."


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required to catch AMP/autocast regression.")
def test_ddim_step_is_amp_invariant_on_cuda_cosine_sensitive_region():
    """
    This is the big one: would have caught the cosine-only 'colored static' bug.
    We simulate the eval path running under autocast(fp16) and ensure the DDIM step
    matches a float32 reference and stays finite.
    """
    device = torch.device("cuda")
    B, C, H, W = 4, 3, 32, 32

    # Use a long K so alpha_bar at high t becomes small (cosine-sensitive regime).
    q = _build_q_const_beta(K=1000, beta=1e-3, device=device)
    sampler = DDIMSampler(q=q, nfe=25, eta=0.0, device=device)

    model = ConstEpsModel(0.1).to(device).eval()

    # Choose a high t to stress sqrt(alpha_bar_t) and the division.
    t_int = 900
    tprev_int = 850
    t = torch.full((B,), t_int, device=device, dtype=torch.long)
    t_prev = torch.full((B,), tprev_int, device=device, dtype=torch.long)

    x_fp32 = (torch.randn(B, C, H, W, device=device, dtype=torch.float32) * 0.2).requires_grad_(False)

    # Reference (force no autocast)
    with torch.autocast(device_type="cuda", enabled=False):
        ref = sampler.step(model, x_fp32, t, t_prev)
    assert torch.isfinite(ref).all()

    # Autocast path (what your eval often does)
    x_fp16 = x_fp32.half()
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        out = sampler.step(model, x_fp16, t, t_prev)

    assert torch.isfinite(out).all(), "Autocast path produced NaNs/Infs — DDIM math likely ran in fp16."
    out_fp32 = out.float()

    # If DDIM update math is properly forced to fp32 internally, these should match closely.
    assert torch.allclose(out_fp32, ref, rtol=2e-3, atol=2e-3), (
        "Autocast changed DDIM step output too much — likely doing sqrt/div in fp16. " "Force update math under autocast(enabled=False) and cast to float32 inside step()."
    )
