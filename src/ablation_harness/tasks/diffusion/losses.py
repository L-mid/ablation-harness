from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .schedule import q_sample, sample_timesteps

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------


@dataclass
class LossConfig:
    """
    Configuration for epsilon-prediction loss weighting.

    weighting:
        - "constant": plain eps-MSE (no timestep weighting).
        - "minsnr":   Min-SNR-γ weighting (w_t = min(γ / SNR_t, 1)).
        - "minsnr_norm":
            Same as "minsnr", but the per-batch weights are normalized
            to have mean 1. This preserves the relative shape over t
            while keeping the *average* effective scale fixed.
    """

    weighting: str = "constant"
    minsnr_gamma: float = 5.0  # γ hyperparameter


# ---------------------------------------------------------------------
# SNR helpers
# ---------------------------------------------------------------------


def compute_snr_from_alphas_cumprod(
    alphas_cumprod: torch.Tensor,  # shape [K]
    timesteps: torch.LongTensor,  # shape [B]
) -> torch.Tensor:
    """
    SNR(t) = alpha_bar_t / (1 - alpha_bar_t) for DDPM-style forward process:
      x_t = sqrt(alpha_bar_t) x0 + sqrt(1 - alpha_bar_t) eps
    """
    alpha_bar_t = alphas_cumprod[timesteps]  # [B]
    one_minus = (1.0 - alpha_bar_t).clamp(min=1e-12)
    snr = alpha_bar_t / one_minus
    return snr


def _make_snr(q: dict, t: torch.LongTensor, device: torch.device) -> torch.Tensor:
    """Compute SNR(t) from the q schedule dict and the batch timesteps t."""
    alphas_cumprod = q["alpha_bar"].to(device=device)
    snr = compute_snr_from_alphas_cumprod(
        alphas_cumprod=alphas_cumprod,
        timesteps=t.to(device=device),
    )  # [B]
    return snr


def _make_gamma(loss_cfg: LossConfig, snr: torch.Tensor) -> torch.Tensor:
    """Create γ tensor from config, on the same device/dtype as snr."""
    gamma = torch.as_tensor(
        loss_cfg.minsnr_gamma,
        dtype=snr.dtype,
        device=snr.device,
    )
    return gamma


def normalize_mean(weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize positive weights so that mean == 1, preserving relative shape.
    """
    mean = weights.mean()
    return weights / (mean + eps)


def _compute_minsnr_base_weight(snr: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """
    Base Min-SNR weight: min(γ / snr, 1) == min(snr, γ) / snr.
    """
    base_weight = torch.minimum(snr, gamma) / snr.clamp(min=1e-12)
    return base_weight  # [B]


# ---------------------------------------------------------------------
# Core weighting logic (for ddpm_loss_with_info)
# ---------------------------------------------------------------------


def _build_loss_with_weighting(  # noqa C901
    loss_cfg: Optional[LossConfig],
    q: dict,
    t: torch.LongTensor,
    device: torch.device,
    mse: torch.Tensor,  # [B]
    log_per_t_mse: bool,
    info: Dict[str, float],
) -> Tuple[torch.Tensor, Dict[str, float]]:  # noqa C901
    """
    Apply configured weighting to per-sample MSE and aggregate logging stats.

    Returns:
        loss: scalar tensor
        info: updated dict with logging stats
    """
    # Plain eps loss (no weighting)
    if loss_cfg is None or loss_cfg.weighting == "constant":
        loss = mse.mean()

        if log_per_t_mse:
            # Unweighted MSE per-t, so E1/E2/E4 are comparable.
            unique_t, inv = torch.unique(t, return_inverse=True)
            for i in range(unique_t.numel()):
                mask = inv == i
                if not mask.any():
                    continue
                mse_mean_t = mse[mask].mean().item()
                t_val = int(unique_t[i].item())
                info[f"mse_per_t/mse_t{t_val:04d}"] = float(mse_mean_t)

        return loss, info

    # -------------------------
    # Min-SNR weighting branch
    # -------------------------
    if loss_cfg.weighting == "minsnr":
        # You probably already have snr_t available from q and t.
        # Typical: snr_t = alpha_bar[t] / (1 - alpha_bar[t])
        a_bar_t = q["alpha_bar"][t].to(device)  # [B]
        snr_t = a_bar_t / (1.0 - a_bar_t + 1e-8)  # [B]

        gamma = float(getattr(loss_cfg, "minsnr_gamma", 5.0))

        # Standard Min-SNR weights: w = min(snr, gamma) / snr
        weights = torch.minimum(snr_t, torch.tensor(gamma, device=device)) / (snr_t + 1e-8)  # [B]

        loss = (weights * mse).mean()

        # Logging (only valid here because weights exists)
        with torch.no_grad():
            w = weights.detach()
            info["mins_snr/snr_mean"] = float(snr_t.mean().item())
            info["mins_snr/weight_mean"] = float(w.mean().item())
            info["mins_snr/weight_min"] = float(w.min().item())
            info["mins_snr/weight_max"] = float(w.max().item())
            info["mins_snr/weight_zero_frac"] = float((w == 0).float().mean().item())
            info["mins_snr/weight_p01"] = float(torch.quantile(w, 0.01).item())
            info["mins_snr/weight_p50"] = float(torch.quantile(w, 0.50).item())
            info["mins_snr/weight_p99"] = float(torch.quantile(w, 0.99).item())

        if log_per_t_mse:
            # Weighted MSE per-t (optional; keep separate keyspace)
            unique_t, inv = torch.unique(t, return_inverse=True)
            for i in range(unique_t.numel()):
                mask = inv == i
                if not mask.any():
                    continue
                mse_mean_t = (weights[mask] * mse[mask]).mean().item()
                t_val = int(unique_t[i].item())
                info[f"wmse_per_t/wmse_t{t_val:04d}"] = float(mse_mean_t)

        return loss, info

        # Min-SNR branches -------------------------------------------------
    if loss_cfg.weighting not in {"minsnr", "minsnr_norm"}:
        raise ValueError(f"Unknown loss.weighting: {loss_cfg.weighting}")

    snr = _make_snr(q, t, device)  # [B]
    gamma = _make_gamma(loss_cfg, snr)  # scalar-like [B] or broadcastable
    base_weight = _compute_minsnr_base_weight(snr, gamma)  # [B]

    if loss_cfg.weighting == "minsnr":
        weights = base_weight
    else:  # "minsnr_norm"
        # Normalize per-batch weights to have mean 1.
        weights = normalize_mean(base_weight)

    # Weighted loss
    loss = (weights * mse).mean()

    # Per-batch summary stats for logging
    info.update(
        {
            # which part of the trajectory this batch hit
            "mins_snr/t_mean": float(t.float().mean().item()),
            "mins_snr/t_min": float(t.min().item()),
            "mins_snr/t_max": float(t.max().item()),
            # SNR stats
            "mins_snr/snr_mean": float(snr.mean().item()),
            "mins_snr/snr_min": float(snr.min().item()),
            "mins_snr/snr_max": float(snr.max().item()),
            # weight stats (after normalization if minsnr_norm)
            "mins_snr/weight_mean": float(weights.mean().item()),
            "mins_snr/weight_min": float(weights.min().item()),
            "mins_snr/weight_max": float(weights.max().item()),
        }
    )

    if log_per_t_mse:
        # Bucket *unweighted* MSE by timestep in this batch.
        unique_t, inv = torch.unique(t, return_inverse=True)
        for i in range(unique_t.numel()):
            mask = inv == i
            if not mask.any():
                continue
            mse_mean_t = mse[mask].mean().item()
            t_val = int(unique_t[i].item())
            info[f"mse_per_t/mse_t{t_val:04d}"] = float(mse_mean_t)

    return loss, info


# ---------------------------------------------------------------------
# Public DDPM loss APIs
# ---------------------------------------------------------------------


def ddpm_loss_with_info(
    model: torch.nn.Module,
    x0: torch.Tensor,
    q: dict,
    loss_cfg: Optional[LossConfig] = None,
    timesteps: Optional[torch.LongTensor] = None,
    log_per_t_mse: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    DDPM epsilon-prediction loss with optional Min-SNR(-norm) weighting.

    Returns:
        loss: scalar tensor
        info: dict[str, float] with e.g. Min-SNR stats (if enabled)
    """
    bsz = x0.size(0)
    device = x0.device
    K = q["betas"].numel()

    # Timesteps
    if timesteps is None:
        if loss_cfg is None:
            t = sample_timesteps(bsz, K, device)
        else:
            t = sample_timesteps(
                bsz,
                K,
                device,
                t_min=getattr(loss_cfg, "t_min", 0),
                t_max=getattr(loss_cfg, "t_max", None),
            )
    else:
        t = timesteps
    x_t, eps = q_sample(x0, t, q)  # forward noising

    # Model prediction
    eps_pred = model(x_t, t)

    # Per-sample MSE
    mse = F.mse_loss(eps_pred.float(), eps.float(), reduction="none")
    mse = mse.view(mse.shape[0], -1).mean(dim=-1)  # [B]

    info: Dict[str, float] = {}

    loss, info = _build_loss_with_weighting(
        loss_cfg=loss_cfg,
        q=q,
        t=t,
        device=device,
        mse=mse,
        log_per_t_mse=log_per_t_mse,
        info=info,
    )

    return loss, info


def ddpm_loss(
    model: torch.nn.Module,
    x0: torch.Tensor,
    q: dict,
    loss_cfg: Optional[LossConfig] = None,
    timesteps: Optional[torch.LongTensor] = None,
) -> torch.Tensor:
    """
    Vanilla simple DDPM epsilon-prediction loss.

    This wrapper keeps the old API: it returns only the scalar loss,
    and is used by older code/tests.
    """
    loss, _ = ddpm_loss_with_info(
        model=model,
        x0=x0,
        q=q,
        loss_cfg=loss_cfg,
        timesteps=timesteps,
        log_per_t_mse=False,
    )
    return loss


# ---------------------------------------------------------------------
# Standalone eps-MSE with weighting (for harness/tests)
# ---------------------------------------------------------------------


def compute_eps_mse_with_weighting(
    model_pred: torch.Tensor,
    target: torch.Tensor,
    timesteps: torch.LongTensor,
    alphas_cumprod: torch.Tensor,
    loss_cfg: Optional[LossConfig],
) -> torch.Tensor:
    """
    Epsilon-prediction MSE loss with optional Min-SNR-γ weighting.

    For epsilon parameterization, Min-SNR-γ weight is usually:
        w_t = min(γ / SNR_t, 1)

    so low-SNR (hard) steps get full weight and high-SNR (easy) steps are downweighted.

    With "minsnr_norm", we additionally normalize the per-batch weights
    to have mean 1 to keep the average scale comparable to a constant loss.
    """
    # per-sample MSE
    mse = F.mse_loss(model_pred.float(), target.float(), reduction="none")
    mse = mse.view(mse.shape[0], -1).mean(dim=-1)  # [B]

    if loss_cfg is None or loss_cfg.weighting == "constant":
        return mse.mean()

    device = model_pred.device
    snr = compute_snr_from_alphas_cumprod(
        alphas_cumprod.to(device=device),
        timesteps.to(device=device),
    )  # [B]

    gamma = torch.as_tensor(
        loss_cfg.minsnr_gamma,
        dtype=snr.dtype,
        device=snr.device,
    )

    base_weight = _compute_minsnr_base_weight(snr, gamma)  # [B]

    if loss_cfg.weighting == "minsnr":
        weights = base_weight
    elif loss_cfg.weighting == "minsnr_norm":
        weights = normalize_mean(base_weight)
    else:
        raise ValueError(f"Unknown loss.weighting: {loss_cfg.weighting}")

    weighted = (weights * mse).mean()
    return weighted
