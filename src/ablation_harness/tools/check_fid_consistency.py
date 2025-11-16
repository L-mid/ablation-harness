"""
Useage:
    python -m ablation_harness.tools.check_fid_consistency \
    --stats stats/cifar10_inception_train.npz \
    --root . \
    --train \
    --device cpu \
    --n-samples 50
  """

import argparse
from pathlib import Path

import numpy as np
import torch
import torchvision as tv
from torch.utils.data import DataLoader, Subset

from ablation_harness.eval.generative import _fid_from_stats, _inception_activations


def pick_device() -> torch.device:
    """Pick CUDA if available, otherwise CPU. Never crash on CPU-only builds."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_cifar10_dataset(root: str = ".", train: bool = True):
    """CIFAR-10 with ToTensor(), should be same as training/eval loaders."""
    tfm = tv.transforms.ToTensor()  # [0,1] floats
    ds = tv.datasets.CIFAR10(root=root, train=train, download=True, transform=tfm)
    return ds


def compute_subset_stats(
    dataset,
    n_samples: int | None,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Compute Inception μ, Σ for a subset (or full set) of the dataset.

    Returns (mu, sigma, actual_n).
    """
    if n_samples is not None and n_samples < len(dataset):
        indices = np.random.RandomState(0).permutation(len(dataset))[:n_samples]
        dataset = Subset(dataset, indices)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0 if device.type == "cpu" else 4,
        pin_memory=(device.type == "cuda"),
    )

    feats_list = []
    total = 0

    for images, _ in loader:
        images = images.to(device, non_blocking=(device.type == "cuda"))  # [B,3,32,32] in [0,1]
        feats = _inception_activations(images, device)  # [B,D] numpy
        feats_list.append(feats)
        total += feats.shape[0]

        if total % 1000 == 0:
            print(f"[check_fid] Processed {total} images...")

    feats_all = np.concatenate(feats_list, axis=0)
    mu = np.mean(feats_all, axis=0)
    sigma = np.cov(feats_all, rowvar=False)
    return mu, sigma, total


def main():
    p = argparse.ArgumentParser(description="Sanity-check FID pipeline by comparing real CIFAR to its own stats.")
    p.add_argument(
        "--stats",
        type=str,
        required=True,
        help="Path to FID stats .npz (with 'mu' and 'sigma').",
    )
    p.add_argument(
        "--root",
        type=str,
        default=".",
        help="Root directory for CIFAR-10 data.",
    )
    p.add_argument(
        "--train",
        action="store_true",
        help="Use CIFAR-10 train split (default: test split if omitted). " "Use the same split you used when *creating* the stats file.",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Number of CIFAR images to use (default: all).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for Inception feature extraction.",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g. 'cpu', 'cuda'). Default: auto.",
    )

    args = p.parse_args()

    device = pick_device()
    print(f"[check_fid] Using device={device}")

    stats_path = Path(args.stats)
    if not stats_path.is_file():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")

    data = np.load(stats_path)
    mu_ref = data["mu"]
    sigma_ref = data["sigma"]
    print(f"[check_fid] Loaded stats from {stats_path}")
    print(f"           mu shape={mu_ref.shape}, sigma shape={sigma_ref.shape}")

    split = "train" if args.train else "test"
    print(f"[check_fid] Building CIFAR-10 ({split} split)...")
    ds = build_cifar10_dataset(root=args.root, train=args.train)

    print(f"[check_fid] Computing subset stats on " f"{'all' if args.n_samples is None else args.n_samples} images...")
    mu_gen, sigma_gen, actual_n = compute_subset_stats(
        ds,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        device=device,
    )

    print(f"[check_fid] Actual images used: {actual_n}")
    fid = _fid_from_stats(mu_gen, sigma_gen, mu_ref, sigma_ref)
    print(f"[check_fid] FID(real-subset vs stats): {fid:.4f}")

    # Simple heuristic: FID should be very small if stats & pipeline match.
    if fid < 3.0:
        print("[check_fid] ✅ Looks good: FID is very low.")
    elif fid < 10.0:
        print("[check_fid] ⚠️ OK-ish: FID is small but non-trivial.")
    else:
        print("[check_fid] ❌ Warning: FID is large; check that stats and pipeline match.")


if __name__ == "__main__":
    main()
