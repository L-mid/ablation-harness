import torch

from .base import BaseSampler


class DDPMSampler(BaseSampler):
    """Class based DDPM sampler."""

    @torch.inference_mode()
    def step(self, model, x_t, t, t_prev=None, *, generator=None):
        """One reverse step p_theta(x_{t-1} | x_t). Only valid for adjacent t -> t-1."""
        q = self.q

        # model forward can be under autocast outside; we'll do the math in fp32 safely
        eps = model(x_t, t)

        device_type = "cuda" if x_t.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            x = x_t.float()
            eps = eps.float()
            t_long = t.to(dtype=torch.long)

            beta_t = q["betas"].float().gather(0, t_long).view(-1, 1, 1, 1)
            alpha_t = q["alphas"].float().gather(0, t_long).view(-1, 1, 1, 1)
            a_bar_t = q["alpha_bar"].float().gather(0, t_long).view(-1, 1, 1, 1)

            sqrt_one_minus_ab = torch.sqrt(torch.clamp(1.0 - a_bar_t, min=1e-12))

            # posterior mean μ_t(x_t, eps)
            mean = (1.0 / torch.sqrt(torch.clamp(alpha_t, min=1e-12))) * (x - (beta_t / sqrt_one_minus_ab) * eps)

            # β̃_t variance (already clipped/logged in precompute_q)
            log_var = q["posterior_log_var_clipped"].float().gather(0, t_long).view(-1, 1, 1, 1)
            var = torch.exp(log_var)
            std = torch.sqrt(torch.clamp(var, min=0.0))

            # No noise at t=0 (final step deterministic)
            is_t0 = (t_long == 0).view(-1, 1, 1, 1)
            std = torch.where(is_t0, torch.zeros_like(std), std)

            noise = torch.randn_like(x)
            out = mean + std * noise

        return out.to(dtype=x_t.dtype)

    @torch.inference_mode()
    def sample(self, model, shape, seed=0):
        """Full DDPM sampler (K steps only)."""
        K = int(self.q["betas"].numel())

        # HARD ERROR: DDPM reverse step is only defined for adjacent steps.
        # If you want fewer steps, use DDIM / DPM-Solver, not "skip" DDPM.
        if getattr(self, "nfe", None) is not None and int(self.nfe) != K:
            raise ValueError(f"DDPMSampler does not support nfe != K (got nfe={int(self.nfe)} vs K={K}). " f"DDPM is only valid for adjacent t->t-1 steps. Use DDIM for accelerated sampling.")

        g = torch.Generator(device=self.device).manual_seed(int(seed))

        B, C, H, W = shape
        x = torch.randn(B, C, H, W, device=self.device)

        # Always full schedule: K-1 down to 0
        t_schedule = torch.arange(K - 1, -1, -1, device=self.device, dtype=torch.long)
        for t in t_schedule:
            x = self.step(model, x, t.expand(B))

        return x.clamp(-1, 1)
