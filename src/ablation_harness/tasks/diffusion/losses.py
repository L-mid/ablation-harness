from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from .schedule import q_sample, sample_timesteps


@dataclass
class LossConfig:
    weighting: str = "constant"  # "constant" or "minsnr"
    minsnr_gamma: float = 5.0  # γ hyperparameter


def ddpm_loss(
    model: torch.nn.Module,
    x0: torch.Tensor,
    q: dict,
    loss_cfg: Optional[LossConfig] = None,
    timesteps: Optional[torch.LongTensor] = None,
) -> torch.Tensor:
    """
    Vanilla simple DDPM epsilon-prediction loss, with optional Min-SNR weighting.

    - Samples timesteps t if not provided.
    - Runs forward noising: x_t, eps = q_sample(x0, t, q)
    - Predicts eps_hat = model(x_t, t)
    - Applies either constant MSE or Min-SNR-γ weighting over t.
    """
    bsz = x0.size(0)
    device = x0.device
    K = q["betas"].numel()  # same as your original code

    t = sample_timesteps(bsz, K, device) if timesteps is None else timesteps  # [B]
    x_t, eps = q_sample(x0, t, q)  # same forward noising as before

    eps_pred = model(x_t, t)

    return compute_eps_mse_with_weighting(
        model_pred=eps_pred,
        target=eps,
        timesteps=t,
        alphas_cumprod=q["alpha_bar"],  # or whatever key you use
        loss_cfg=loss_cfg,
    )


#
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
