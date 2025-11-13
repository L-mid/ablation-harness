import torch

# from ..schedule import make_t_schedule
from .base import BaseSampler


class DDPMSampler(BaseSampler):
    """Class based DDPM sampler."""

    @torch.no_grad()
    def step(self, model, x_t, t, t_prev=None):
        """Per step math."""
        q = self.q
        eps = model(x_t, t)

        beta_t = q["betas"][t].view(-1, 1, 1, 1)
        alpha_t = q["alphas"][t].view(-1, 1, 1, 1)
        a_bar_t = q["alpha_bar"][t].view(-1, 1, 1, 1)
        sqrt_one_minus_ab = torch.sqrt(1 - a_bar_t)

        # posterior mean μ_t(x_t, eps)
        mean = (1 / torch.sqrt(alpha_t)) * (x_t - (beta_t / (sqrt_one_minus_ab + 1e-8)) * eps)

        # β̃_t variance (already clipped/logged in precompute_q)
        var = torch.exp(q["posterior_log_var_clipped"][t]).view(-1, 1, 1, 1)
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(var) * noise

    @torch.no_grad()
    def sample(self, model, shape, seed=0):
        """Full DDPM sampler call (handles steps and schedule call)."""
        torch.manual_seed(seed)
        K = self.q["betas"].numel()
        t_schedule = torch.arange(K - 1, -1, -1, device=self.device)  # full chain
        B, C, H, W = shape
        x = torch.randn(B, C, H, W, device=self.device)
        for t in t_schedule:
            x = self.step(model, x, t.expand(B))
        return x.clamp(-1, 1)
