"""
Debug script: directly sample from the trained model with DDPMSampler,
bypassing evaluate_diffusion / FID.

Usage:

    python -m ablation_harness.tools.debug_manual_sample \
      --ckpt docs/assets/E2/last.pt \
      --out docs/assets/E2/debug_samples_nfe_1000.png \
      --nfe 1000
"""

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

# --- adjust imports to your actual layout ---
from ablation_harness.tasks.diffusion.models.unet_cifar32 import UNetCifar32
from ablation_harness.tasks.diffusion.samplers.ddpm import DDPMSampler
from ablation_harness.tasks.diffusion.schedule import get_betas_linear, precompute_q

# from ablation_harness.tasks.diffusion.ema import EMA   # or build_ema


def build_model(device: torch.device) -> torch.nn.Module:
    """Builds model with same weights as checkpoint."""
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

    ema_state = state.get("ema", None)
    if ema_state is not None:
        from ablation_harness.ema import EMA  # adjust path

        ema = EMA(model, decay=0.9999)
        ema.load_state_dict(ema_state)
        ema_model = build_model(device)
        ema.copy_to(ema_model)
        ema_model.eval()
        return ema_model

    model.eval()
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("debug_samples.png"))
    p.add_argument("--K", type=int, default=1000)
    p.add_argument("--nfe", type=int, default=50)
    p.add_argument("--num", type=int, default=16)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    betas = get_betas_linear(args.K, device=device)
    q = precompute_q(betas)

    model = load_ema_model(args.ckpt, device)

    sampler = DDPMSampler(q=q, nfe=args.nfe, device=device)
    imgs = sampler.sample(model, shape=(args.num, 3, 32, 32), seed=0)

    # map [-1,1] → [0,1] for saving
    imgs_vis = (imgs.clamp(-1, 1) + 1) / 2.0
    save_image(imgs_vis, args.out, nrow=8)
    print(f"[debug_manual_sample] Saved samples to {args.out}")


if __name__ == "__main__":
    main()
