import torch
import torch.nn.functional as F


# ---------- schedules ----------
def get_betas_linear(K: int, beta_start=1e-4, beta_end=2e-2, device="cpu"):
    """Linear."""
    return torch.linspace(beta_start, beta_end, K, device=device)


def get_betas_cosine(K: int, s=0.008, device="cpu"):
    """alpha_bar(t) = cos^2((t/T + s)/(1+s) * pi/2)"""
    steps = torch.arange(K + 1, device=device, dtype=torch.float32)
    f = torch.cos(((steps / K + s) / (1 + s)) * 3.1415926535 / 2) ** 2
    alpha_bar = f / f[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return betas.clamp(1e-8, 0.999)


def get_betas_cosine_match_linear(K: int, s=0.008, device="cpu"):
    """
    Cosine schedule whose total noise mass Σβ is scaled to match
    the linear schedule with the same K (and default beta_start/end).

    Used for E5: 'beta-scale match' control.
    """
    betas_cos = get_betas_cosine(K, s=s, device=device)
    betas_lin = get_betas_linear(K, device=device)

    sum_cos = betas_cos.sum()
    sum_lin = betas_lin.sum()

    # Avoid any weird division-by-zero (shouldn't happen in practice).
    scale = sum_lin / (sum_cos + 1e-12)

    betas = betas_cos * scale
    return betas.clamp(1e-8, 0.999)

 
def get_beta_schedule(kind: str, K: int, device="cpu"):
    if kind == "linear":
        return get_betas_linear(K, device=device)
    if kind == "cosine":
        return get_betas_cosine(K, device=device)
    if kind == "cosine_match_linear":
        return get_betas_cosine_match_linear(K, device=device)
    raise ValueError(f"unknown schedule: {kind}")


@torch.no_grad()
def precompute_q(betas: torch.Tensor):
    alphas = 1.0 - betas  # [K]
    alpha_bar = torch.cumprod(alphas, dim=0)  # [K]

    # β̃_t = β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t), with β̃_0 defined as a tiny epsilon
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


# ---------- training loss ----------
def sample_timesteps(bsz: int, K: int, device):
    return torch.randint(0, K, (bsz,), device=device)


def q_sample(x0, t, q):
    """x_t = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * eps"""
    eps = torch.randn_like(x0)
    sqrt_ab = q["sqrt_alpha_bar"][t].view(-1, 1, 1, 1)
    sqrt_om = q["sqrt_one_minus_alpha_bar"][t].view(-1, 1, 1, 1)
    return sqrt_ab * x0 + sqrt_om * eps, eps


def ddpm_loss(model, x0, q, timesteps=None):
    """Vanilla simple mse ddpm loss fn, native to schedule."""
    bsz = x0.size(0)
    device = x0.device
    K = q["betas"].numel()
    t = sample_timesteps(bsz, K, device) if timesteps is None else timesteps
    x_t, eps = q_sample(x0, t, q)
    eps_pred = model(x_t, t)
    return F.mse_loss(eps_pred, eps)


# ---------- sampling (subset steps, DDIM-like with eta=1 → stochastic) ----------
@torch.no_grad()
def make_t_schedule(K: int, nfe: int, device):
    idx = torch.linspace(K - 1, 0, nfe, device=device).long()
    return torch.clamp(idx, 0, K - 1)


@torch.no_grad()
def p_sample_step(model, x_t, t, t_prev, q, eta=1.0):
    eps_pred = model(x_t, t)  # t: [B]

    a_bar_t = q["alpha_bar"][t]  # [B]

    if t_prev.dim() == 0 and t_prev.item() == -1:
        a_bar_prev = torch.ones_like(a_bar_t)  # ᾱ_{-1} = 1
    else:
        a_bar_prev = q["alpha_bar"][t_prev]  # scalar or [B]
        if a_bar_prev.dim() == 0:
            a_bar_prev = a_bar_prev.expand_as(a_bar_t)

    sqrt_ab_t = torch.sqrt(a_bar_t).view(-1, 1, 1, 1)
    sqrt_om_t = torch.sqrt(1 - a_bar_t).view(-1, 1, 1, 1)

    x0_pred = (x_t - sqrt_om_t * eps_pred) / (sqrt_ab_t + 1e-8)

    # DDIM-like update; guard numerics & shapes
    num = 1 - a_bar_prev
    den = 1 - a_bar_t
    frac = torch.clamp(num / den, min=0.0)
    one_minus_ratio = torch.clamp(1 - a_bar_t / a_bar_prev, min=0.0)

    sigma = eta * torch.sqrt(frac * one_minus_ratio).view(-1, 1, 1, 1)
    dir_xt = torch.sqrt(a_bar_prev).view(-1, 1, 1, 1) * x0_pred

    noise = sigma * torch.randn_like(x_t)

    B = x_t.shape[0]
    # instead of leaving it as (B,)
    a_bar_prev_ = a_bar_prev.view(B, 1, 1, 1)

    x_prev = dir_xt + torch.sqrt(torch.clamp(1 - a_bar_prev_ - sigma**2, min=0.0)) * eps_pred + noise
    return x_prev


@torch.no_grad()
def sample_ddpm(model, shape, q, nfe=50, eta=1.0, seed=0, device="cpu"):
    torch.manual_seed(seed)
    K = q["betas"].numel()
    t_schedule = make_t_schedule(K, nfe, device)
    B, C, H, W = shape
    x = torch.randn(B, C, H, W, device=device)
    for i, t in enumerate(t_schedule):
        t = t.expand(B)
        # use -1 on the final step; not 0
        t_prev = t_schedule[i + 1] if i + 1 < len(t_schedule) else torch.tensor(-1, device=device)
        x = p_sample_step(model, x, t, t_prev, q, eta=eta)
    return x.clamp(-1, 1)
