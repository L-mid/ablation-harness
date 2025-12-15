import torch

from ablation_harness.tasks.diffusion.samplers.ddim import DDIMSampler
from ablation_harness.tasks.diffusion.schedule import precompute_q


class ZeroModel(torch.nn.Module):
    def forward(self, x, t):
        return torch.zeros_like(x)


def test_ddim_step_zero_model_eta0_matches_scale():
    device = torch.device("cpu")
    K = 1000

    def get_betas_linear(K: int, beta_start=1e-4, beta_end=2e-2, device="cpu"):
        """Linear."""
        return torch.linspace(beta_start, beta_end, K, device=device)

    betas = get_betas_linear(K)

    q = precompute_q(betas)  # however you build q

    B = 4
    x_t = torch.randn(B, 3, 32, 32, device=device)
    t = torch.full((B,), 900, device=device, dtype=torch.long)
    t_prev = torch.full((B,), 500, device=device, dtype=torch.long)

    smp = DDIMSampler(q=q, nfe=20, eta=0.0, device=device)

    x_prev = smp.step(ZeroModel().to(device), x_t, t, t_prev)

    ab_t = q["alpha_bar"][t].view(B, 1, 1, 1)
    ab_prev = q["alpha_bar"][t_prev].view(B, 1, 1, 1)
    expected = torch.sqrt(ab_prev / ab_t) * x_t

    max_abs = (x_prev - expected).abs().max().item()
    assert max_abs < 5e-5
