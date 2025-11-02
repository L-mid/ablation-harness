import math
import os

import torch
from torchvision.utils import save_image

from ablation_harness.tasks.diffusion.core import sample_ddpm


@torch.no_grad()
def evaluate_diffusion(model_ema, eval_cfg, q, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    sampler = eval_cfg.get("sampler", "ddpm")
    nfe = int(eval_cfg.get("nfe", 50))
    n_samples = int(eval_cfg.get("n_samples", 1000))
    seed = int(eval_cfg.get("sample_seed", 0))
    save_images = bool(eval_cfg.get("save_images", False))
    fid_stats = eval_cfg.get("fid_stats", None)
    B = 64
    images = []
    remaining = n_samples
    while remaining > 0:
        b = min(B, remaining)
        imgs = sample_ddpm(model_ema, (b, 3, 32, 32), q, nfe=nfe, eta=1.0, seed=seed, device=next(model_ema.parameters()).device)
        images.append(imgs)
        remaining -= b
    imgs = torch.cat(images, dim=0)
    # scale to [0,1] for saving
    if save_images:
        grid = int(math.sqrt(min(n_samples, 256)))
        save_image((imgs[: grid * grid] + 1) / 2, os.path.join(out_dir, "samples_grid.png"), nrow=grid)
    # TODO: integrate real FID/KID here; for S1 just return stub
    return {"fid": None, "kid": None, "n": imgs.size(0)}
