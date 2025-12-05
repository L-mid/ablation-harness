from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from .schedule import q_sample, sample_timesteps


@dataclass
class LossConfig:
    weighting: str = "constant"  # "constant" or "minsnr"
    minsnr_gamma: float = 5.0  # γ hyperparameter


def ddpm_loss_with_info(
    model: torch.nn.Module,
    x0: torch.Tensor,
    q: dict,
    loss_cfg: Optional[LossConfig] = None,
    timesteps: Optional[torch.LongTensor] = None,
    log_per_t_mse: bool = False,  # 🔹 new
) -> tuple[torch.Tensor, dict]:
    """
    Same as ddpm_loss, but also returns a dict of logging stats.

    Returns:
        loss: scalar tensor
        info: dict[str, float] with e.g. Min-SNR stats (if enabled)
    """
    bsz = x0.size(0)
    device = x0.device
    K = q["betas"].numel()

    # Timesteps
    t = sample_timesteps(bsz, K, device) if timesteps is None else timesteps  # [B]
    x_t, eps = q_sample(x0, t, q)  # forward noising

    # Model prediction
    eps_pred = model(x_t, t)

    # Per-sample MSE
    mse = F.mse_loss(eps_pred.float(), eps.float(), reduction="none")
    mse = mse.view(mse.shape[0], -1).mean(dim=-1)  # [B]

    info: dict[str, float] = {}

    # No special weighting
    if loss_cfg is None or loss_cfg.weighting == "constant":
        loss = mse.mean()
        return loss, info

    if loss_cfg.weighting == "minsnr":
        # SNR(t)
        alphas_cumprod = q["alpha_bar"].to(device=device)
        snr = compute_snr_from_alphas_cumprod(
            alphas_cumprod=alphas_cumprod,
            timesteps=t.to(device=device),
        )  # [B]

        gamma = torch.as_tensor(
            loss_cfg.minsnr_gamma,
            dtype=snr.dtype,
            device=snr.device,
        )

        # base_weight = min(γ / snr, 1) = min(snr, γ) / snr
        base_weight = torch.minimum(snr, gamma) / snr.clamp(min=1e-12)  # [B]

        loss = (base_weight * mse).mean()

        # Per-batch summary stats for logging
        info = {
            # which part of the trajectory this batch hit
            "mins_snr/t_mean": float(t.float().mean().item()),
            "mins_snr/t_min": float(t.min().item()),
            "mins_snr/t_max": float(t.max().item()),
            # SNR stats
            "mins_snr/snr_mean": float(snr.mean().item()),
            "mins_snr/snr_min": float(snr.min().item()),
            "mins_snr/snr_max": float(snr.max().item()),
            # weight stats
            "mins_snr/weight_mean": float(base_weight.mean().item()),
            "mins_snr/weight_min": float(base_weight.min().item()),
            "mins_snr/weight_max": float(base_weight.max().item()),
        }

        if log_per_t_mse:
            # Bucket per-sample MSE by timestep in this batch.
            # This is unweighted MSE (so you can compare E1/E2/E3 directly).
            unique_t, inv = torch.unique(t, return_inverse=True)  # each t present in this batch
            for i in range(unique_t.numel()):
                mask = inv == i
                if not mask.any():
                    continue
                mse_mean_t = mse[mask].mean().item()
                t_val = int(unique_t[i].item())
                info[f"mse_per_t/mse_t{t_val:04d}"] = float(mse_mean_t)

        return loss, info

    raise ValueError(f"Unknown loss.weighting: {loss_cfg.weighting}")


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
    )
    return loss


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
    """
    # per-sample MSE
    mse = F.mse_loss(model_pred.float(), target.float(), reduction="none")
    mse = mse.view(mse.shape[0], -1).mean(dim=-1)  # [B]

    if loss_cfg is None or loss_cfg.weighting == "constant":
        return mse.mean()

    if loss_cfg.weighting == "minsnr":
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

        # min(γ / snr, 1)  ==  min(snr, γ) / snr
        base_weight = torch.minimum(snr, gamma) / snr.clamp(min=1e-12)  # [B]

        weighted = (base_weight * mse).mean()
        return weighted

    raise ValueError(f"Unknown loss.weighting: {loss_cfg.weighting}")
