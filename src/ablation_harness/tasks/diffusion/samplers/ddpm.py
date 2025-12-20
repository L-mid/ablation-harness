import torch

# from ..schedule import make_t_schedule
from .base import BaseSampler


class DDPMSampler(BaseSampler):
    """Class based DDPM sampler."""

    @torch.inference_mode()
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

    @torch.inference_mode()
    def sample(self, model, shape, seed=0):
        """Full DDPM sampler call (handles steps and schedule call)."""
        torch.manual_seed(seed)
        K = self.q["betas"].numel()
        # Decide how many steps we actually take
        nfe = min(self.nfe, K) if getattr(self, "nfe", None) is not None else K
        # Choose a schedule: nfe indices between [0, K-1], then go backwards
        t_indices = torch.linspace(0, K - 1, steps=nfe, dtype=torch.long, device=self.device)
        t_schedule = t_indices.flip(0)  # from K-1 down to 0

        B, C, H, W = shape
        x = torch.randn(B, C, H, W, device=self.device)

        for t in t_schedule:
            # t is scalar; expand for batch
            x = self.step(model, x, t.expand(B))
        return x.clamp(-1, 1)
