import torch

from ..schedule import make_t_schedule
from .base import BaseSampler


class DDIMSampler(BaseSampler):
    @torch.inference_mode()
    def step(self, model, x_t, t, t_prev, *, generator: torch.Generator | None = None):
        q, eta = self.q, float(self.eta)
        B = x_t.size(0)

        # Make sure t, t_prev are (B,) long
        if t.dim() == 0:
            t = t.expand(B)
        t = t.to(dtype=torch.long)

        if t_prev.dim() == 0:
            t_prev = t_prev.expand(B)
        t_prev = t_prev.to(dtype=torch.long)

        # Model forward (can be under autocast; that's fine)
        eps_pred = model(x_t, t)

        # DDIM update math in float32, with autocast disabled
        device_type = "cuda" if x_t.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            x = x_t.float()
            eps = eps_pred.float()

            abar = q["alpha_bar"].float()  # [K]
            a_bar_t = abar.gather(0, t).view(B, 1, 1, 1)

            # Robust terminal handling for t_prev == -1 (scalar OR batched)
            terminal = (t_prev < 0).view(B, 1, 1, 1)
            t_prev_clamped = t_prev.clamp(0, abar.numel() - 1)
            a_bar_prev = abar.gather(0, t_prev_clamped).view(B, 1, 1, 1)
            a_bar_prev = torch.where(terminal, torch.ones_like(a_bar_prev), a_bar_prev)

            one = torch.tensor(1.0, device=x.device, dtype=torch.float32)

            sqrt_ab_t = torch.sqrt(torch.clamp(a_bar_t, min=1e-12))
            sqrt_om_t = torch.sqrt(torch.clamp(one - a_bar_t, min=1e-12))

            x0_pred = (x_t - sqrt_om_t * eps_pred) / sqrt_ab_t

            # (optional but helps stability)
            x0_pred = x0_pred.clamp(-1.0, 1.0)

            num = one - a_bar_prev
            den = torch.clamp(one - a_bar_t, min=1e-12)
            frac = torch.clamp(num / den, min=0.0)
            one_minus_ratio = torch.clamp(one - (a_bar_t / torch.clamp(a_bar_prev, min=1e-12)), min=0.0)

            sigma = eta * torch.sqrt(frac * one_minus_ratio)  # [B,1,1,1]

            if eta == 0.0:
                noise = torch.zeros_like(x)
            else:
                noise = sigma * torch.randn_like(x, generator=generator)

            c = torch.sqrt(torch.clamp(one - a_bar_prev - sigma * sigma, min=0.0))

            x_prev = torch.sqrt(torch.clamp(a_bar_prev, min=0.0)) * x0_pred + c * eps + noise

        return x_prev.to(dtype=x_t.dtype)

    @torch.inference_mode()
    def sample(self, model, shape, seed: int = 0):
        g = torch.Generator(device=self.device).manual_seed(int(seed))
        K = self.q["betas"].numel()
        t_schedule = make_t_schedule(K, self.nfe, self.device)
        B, C, H, W = shape
        x = torch.randn(B, C, H, W, device=self.device, generator=g)
        for i, t in enumerate(t_schedule):
            t = t.expand(B)
            t_prev = t_schedule[i + 1] if i + 1 < len(t_schedule) else torch.tensor(-1, device=self.device)
            x = self.step(model, x, t, t_prev, generator=g)
        return x.clamp(-1, 1)
