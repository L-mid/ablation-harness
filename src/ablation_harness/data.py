from types import ModuleType
from typing import Callable, Optional, Tuple

import torch
from torch.utils.data import Dataset, TensorDataset

try:
    import torchvision as tv
except Exception:
    tv: Optional[ModuleType] = None


# synthetic MLP dataset
def build_synthetic_moons(n=1024, seed=0) -> Tuple[TensorDataset, TensorDataset]:
    from sklearn.datasets import make_moons

    X, y = make_moons(n_samples=n, noise=0.15, random_state=seed)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    ntr = int(0.8 * n)
    return (TensorDataset(X[:ntr], y[:ntr]), TensorDataset(X[ntr:], y[ntr:]))


def build_cifar10(subset=None):
    assert tv is not None, "torchvision required for CIFAR10"
    tfm = tv.transforms.Compose([tv.transforms.ToTensor()])
    tr = tv.datasets.CIFAR10(root=".", train=True, download=True, transform=tfm)
    va = tv.datasets.CIFAR10(root=".", train=False, download=True, transform=tfm)
    if subset is not None and subset < len(tr):
        from torch.utils.data import Subset

        tr = Subset(tr, list(range(subset)))
    return tr, va


def build_dataset(cfg) -> tuple[Dataset, Dataset, Optional[Callable]]:
    """selects moons/cifar10; returns train, val, collate_fn or None"""
    # no collate_fn implementation
    # --- Data ---
    if cfg.dataset == "moons":
        train_ds, val_ds = build_synthetic_moons(n=1024, seed=cfg.seed)
    elif cfg.dataset == "cifar10":
        train_ds, val_ds = build_cifar10(subset=cfg.subset)
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")

    return train_ds, val_ds, None  # ?
