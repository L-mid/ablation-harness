"""
Do this on cuda.

Usage:
    python -m ablation_harness.tools.fid_stats.make_cifar10_fid_stats \
    --train --device cuda \
    --out stats/cifar10_inception_train.npz


"""

# from ablation_harness.tools.make_cifar10_fid_stats

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torchvision as tv
from torch.utils.data import DataLoader

from ablation_harness.eval.generative import _inception_activations


def build_cifar10_dataset(root: str = ".", train: bool = True):
    """CIFAR-10 with just ToTensor(), matching training loader."""
    tfm = tv.transforms.ToTensor()  # gives [0,1]
    ds = tv.datasets.CIFAR10(root=root, train=train, download=True, transform=tfm)
    return ds


def compute_inception_stats(
    dataset,
    batch_size: int = 256,
    device: str | torch.device = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the dataset through Inception and return (mu, sigma).
    """
    device = torch.device(device)

    pin_memory = False
    if device == "cuda":
        pin_memory = True

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=pin_memory,
    )

    num_batches = math.ceil(len(dataset) / batch_size)
    print(f"[stats] Total batches: {num_batches}")
    feats_list = []
    for i, (images, labels) in enumerate(loader):
        if i % 5 == 0:  # show every 5 batches
            print(f"[stats] batch {i+1}/{num_batches}")

        images = images.to(device, non_blocking=(device.type == "cuda"))  # [B, 3, 32, 32] in [0,1]
        feats = _inception_activations(images, device)  # [B, D] numpy
        feats_list.append(feats)

        if (i + 1) % 50 == 0:
            print(f"[stats] Processed { (i + 1) * loader.batch_size } images...")

    feats_all = np.concatenate(feats_list, axis=0)
    mu = np.mean(feats_all, axis=0)
    sigma = np.cov(feats_all, rowvar=False)
    return mu, sigma


def main():
    """Primary orchestrator: Fid stats file cal."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=str,
        default="cifar10_inception_stats.npz",
        help="Where to save the FID stats.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Root directory for CIFAR-10 data.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Use CIFAR-10 train split (default: test split if not set).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )

    args = parser.parse_args()
    # device = args.device   # might want to add later
    device = "cuda"  # do not argue

    split = "train" if args.train else "test"
    print(f"[stats] Building CIFAR-10 ({split} split)...")
    ds = build_cifar10_dataset(root=args.root, train=args.train)

    print(f"[stats] Computing Inception stats on {len(ds)} images using device={device}...")
    mu, sigma = compute_inception_stats(ds, batch_size=args.batch_size, device=device)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, mu=mu, sigma=sigma, num_samples=len(ds))

    print(f"[stats] Saved FID stats to: {out_path.resolve()}")
    print(f"[stats] mu shape: {mu.shape}, sigma shape: {sigma.shape}")


if __name__ == "__main__":
    main()
