"""
Debug script: visualize local denoising behaviour of the trained diffusion model.

Usage (example):

    python -m ablation_harness.tools.debug_local_denoise \
      --ckpt docs/assets/E2/last.pt \
      --out docs/assets/E2/debug_denoise

It will save a few PNGs like:
  debug_denoise_t0.png, debug_denoise_t250.png, ...
"""

import argparse
from pathlib import Path

import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image

# --- adjust these imports to your actual package layout ---
from ablation_harness.tasks.diffusion.models.unet_cifar32 import UNetCifar32
from ablation_harness.tasks.diffusion.schedule import (
    get_betas_linear,
    precompute_q,
    q_sample,
)

# if you have a central EMA builder, import that instead
# from ablation_harness.tasks.diffusion.ema import build_ema


def build_model(device: torch.device) -> torch.nn.Module:
    """Mirror training config for unet_cifar32"""
    model = UNetCifar32(
        in_channels=3,
        out_channels=3,
        base_channels=32,
        channel_mults=(1, 2, 2, 2),
        num_res_blocks=2,
        dropout=0.1,
        time_hidden=512,
        gn_groups=32,
    )
    return model.to(device)


def load_ema_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    state = torch.load(ckpt_path, map_location=device)

    model = build_model(device)
    model.load_state_dict(state["model"])

    # If you have an EMA class / builder, use it here.
    # This is a template – adapt to your actual EMA API.
    ema_state = state.get("ema", None)
    if ema_state is not None:
        from ablation_harness.ema import EMA  # adjust path

        ema = EMA(model, decay=0.9999)
        ema.load_state_dict(ema_state)
        # apply EMA weights into a fresh copy so we don't mutate training model
        ema_model = build_model(device)
        ema.copy_to(ema_model)
        ema_model.eval()
        return ema_model

    # fallback: just use plain model
    model.eval()
    return model


def get_cifar_batch(device: torch.device, batch_size: int = 8) -> torch.Tensor:
    # Use torchvision directly – this doesn’t have to go through your harness
    tfm = transforms.Compose([transforms.ToTensor()])
    ds = datasets.CIFAR10(root="data/cifar10", train=True, download=True, transform=tfm)
    x0, _ = zip(*[ds[i] for i in range(batch_size)])
    x0 = torch.stack(x0, dim=0).to(device)  # [B,3,32,32] in [0,1]
    x0 = x0 * 2.0 - 1.0  # → [-1,1], like training
    return x0


@torch.no_grad()
def debug_local_denoise(model, q, x0, timesteps, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    device = x0.device
    B = x0.size(0)
    K = q["betas"].numel()

    for t_scalar in timesteps:
        t_scalar = max(0, min(K - 1, int(t_scalar)))
        t = torch.full((B,), t_scalar, device=device, dtype=torch.long)

        x_t, eps = q_sample(x0, t, q)  # your forward process

        eps_pred = model(x_t, t)
        sqrt_ab = q["sqrt_alpha_bar"][t].view(-1, 1, 1, 1)
        sqrt_om = q["sqrt_one_minus_alpha_bar"][t].view(-1, 1, 1, 1)
        x0_pred = (x_t - sqrt_om * eps_pred) / (sqrt_ab + 1e-8)

        # Map to [0,1] for saving
        grid = torch.cat(
            [
                (x0.clamp(-1, 1) + 1) / 2,
                (x_t.clamp(-1, 1) + 1) / 2,
                (x0_pred.clamp(-1, 1) + 1) / 2,
            ],
            dim=0,
        )
        out_path = out_dir / f"debug_denoise_t{t_scalar}.png"
        save_image(grid, out_path, nrow=x0.size(0))
        print(f"[debug_denoise] Saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True, help="Path to diffusion checkpoint (with model + ema).")
    p.add_argument("--out", type=Path, default=Path("debug_denoise"))
    p.add_argument("--K", type=int, default=1000, help="Number of diffusion steps used in training.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    betas = get_betas_linear(args.K, device=device)
    q = precompute_q(betas)  # your q dict: sqrt_alpha_bar, etc.

    model = load_ema_model(args.ckpt, device)
    x0 = get_cifar_batch(device, batch_size=8)

    timesteps = [0, args.K // 4, args.K // 2, 3 * args.K // 4, args.K - 1]
    debug_local_denoise(model, q, x0, timesteps, args.out)


if __name__ == "__main__":
    main()
