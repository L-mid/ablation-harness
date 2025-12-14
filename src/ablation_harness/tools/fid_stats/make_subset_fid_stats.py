"""
Use on cuda.

Useage:
python -m ablation_harness.tools.make_subset_fid_stats \
  --train --device cuda --batch-size 256 \
  --n 2048 --seed 0 \
  --out stats/cifar10_inception_train_n2048_seed0.npz

"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchvision as tv
from torch.utils.data import DataLoader, Subset

from ablation_harness.data import build_cifar10  # your build_cifar10 ([-1,1])
from ablation_harness.eval.generative import _inception_activations


def compute_inception_stats_subset(
    train: bool,
    n: int | None,
    seed: int,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = torch.device(device)

    # Use your canonical dataset pipeline to avoid drift:
    # build_cifar10() gives [-1,1], we map back to [0,1] like the real FID path. :contentReference[oaicite:1]{index=1}
    tr, va = build_cifar10(subset=None)
    ds = tr if train else va

    total = len(ds)
    if n is None:
        idx = np.arange(total, dtype=np.int64)
    else:
        rng = np.random.default_rng(seed)
        idx = rng.choice(total, size=int(n), replace=False).astype(np.int64)

    subset = Subset(ds, idx.tolist())

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    feats_list = []
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=(device.type == "cuda"))
        xb = (xb.clamp(-1, 1) + 1.0) / 2.0  # [-1,1] -> [0,1] (same as eval) :contentReference[oaicite:2]{index=2}
        f_np = _inception_activations(xb, device, batch_size=64)  # np [b, D]
        feats_list.append(f_np)

    feats_all = np.concatenate(feats_list, axis=0).astype(np.float64)
    mu = feats_all.mean(axis=0)
    sigma = np.cov(feats_all, rowvar=False)
    return mu, sigma, idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--train", action="store_true")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n", type=int, default=None, help="If set, use a seeded random subset of size n")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    mu, sigma, idx = compute_inception_stats_subset(
        train=args.train,
        n=args.n,
        seed=args.seed,
        batch_size=args.batch_size,
        device=args.device,
    )

    meta = {
        "split": "train" if args.train else "test",
        "n": int(len(idx)),
        "seed": int(args.seed),
        "device": str(args.device),
        "torch": torch.__version__,
        "torchvision": tv.__version__,
        "inception_weights": "torchvision.models.Inception_V3_Weights.DEFAULT",
        "note": "dataset=build_cifar10() then mapped back to [0,1] like generative FID path",
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, mu=mu, sigma=sigma, idx=idx, meta=json.dumps(meta))


if __name__ == "__main__":
    main()
