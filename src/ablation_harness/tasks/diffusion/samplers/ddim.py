import torch

from ..schedule import make_t_schedule
from .base import BaseSampler


class DDIMSampler(BaseSampler):
    """class style sampler for ddim."""

    @torch.no_grad()
    def step(self, model, x_t, t, t_prev):
        """One step of ddim sampling math."""
        q, eta = self.q, self.eta
        eps_pred = model(x_t, t)

        a_bar_t = q["alpha_bar"][t]  # [B]
        a_bar_prev = torch.ones_like(a_bar_t) if (t_prev.dim() == 0 and t_prev.item() == -1) else q["alpha_bar"][t_prev]
        if a_bar_prev.dim() == 0:
            a_bar_prev = a_bar_prev.expand_as(a_bar_t)

        sqrt_ab_t = torch.sqrt(a_bar_t).view(-1, 1, 1, 1)
        sqrt_om_t = torch.sqrt(1 - a_bar_t).view(-1, 1, 1, 1)

        x0_pred = (x_t - sqrt_om_t * eps_pred) / (sqrt_ab_t + 1e-8)

        num = 1 - a_bar_prev
        den = 1 - a_bar_t
        frac = torch.clamp(num / den, min=0.0)
        one_minus_ratio = torch.clamp(1 - a_bar_t / a_bar_prev, min=0.0)

        sigma = eta * torch.sqrt(frac * one_minus_ratio).view(-1, 1, 1, 1)
        dir_xt = torch.sqrt(a_bar_prev).view(-1, 1, 1, 1) * x0_pred
        a_bar_prev_ = a_bar_prev.view(x_t.size(0), 1, 1, 1)
        noise = sigma * torch.randn_like(x_t)
        return dir_xt + torch.sqrt(torch.clamp(1 - a_bar_prev_ - sigma**2, min=0.0)) * eps_pred + noise

    @torch.no_grad()
    def sample(self, model, shape, seed: int = 0):
        """Sampling from ddim, called schedule and assesses all steps. Returns in (-1, 1)"""
        g = torch.Generator(device=self.device).manual_seed(int(seed))
        torch.manual_seed(seed)
        K = self.q["betas"].numel()
        t_schedule = make_t_schedule(K, self.nfe, self.device)
        B, C, H, W = shape
        x = torch.randn(B, C, H, W, device=self.device, generator=g)
        for i, t in enumerate(t_schedule):
            t = t.expand(B)
            t_prev = t_schedule[i + 1] if i + 1 < len(t_schedule) else torch.tensor(-1, device=self.device)
            x = self.step(model, x, t, t_prev)
        return x.clamp(-1, 1)
