import random
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ablation_harness.seed_utils import seed_everything, seed_worker

try:
    import torchvision as tv
except Exception:
    tv: Optional[ModuleType] = None

"""
Change the roots in datasets (not in repo).
"""


# -------------------------
# Config
# -------------------------
@dataclass
class TrainConfig:
    model: str = "mlp"  # "mlp" | "tinycnn"
    hidden: int = 64
    dropout: float = 0.0
    lr: float = 1e-3
    wd: float = 0.0
    epochs: int = 4
    batch_size: int = 64
    seed: int = 0
    dataset: str = "moons"  # "moons" | "fakedata" | "mnist" | "cifar10"
    subset: Optional[int] = None  # e.g., 1000
    num_workers: int = 0
    pin_memory: bool = False


# --------------------------
# Utils
# --------------------------


def _set_seed(s: int):  # unused because seed_utils.py
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------
# Models
# ----------------------


# MLP
class MLP(nn.Module):
    def __init__(self, hidden=64, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 2))

    def forward(self, x):
        return self.net(x)


# TinyCNN
class TinyCNN(nn.Module):
    def __init__(self, num_classes=10, dropout=0.0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)  # [B, 64, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 64]
        return self.classifier(x)


# ----------------------
# Data builders
# ----------------------


# synthetic dataset
def build_synthetic_moons(n=1024, seed=0) -> Tuple[TensorDataset, TensorDataset]:
    from sklearn.datasets import make_moons

    X, y = make_moons(n_samples=n, noise=0.15, random_state=seed)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    ntr = int(0.8 * n)
    return (TensorDataset(X[:ntr], y[:ntr]), TensorDataset(X[ntr:], y[ntr:]))


def build_fakedata(size=2000, seed=0, subset=None):
    assert tv is not None, "torchvision not available for FakeData"
    g = torch.Generator().manual_seed(seed)
    tfm = tv.transforms.Compose([tv.transforms.ToTensor()])
    ds = tv.datasets.FakeData(
        size=size,
        image_size=(3, 32, 32),
        num_classes=10,
        transform=tfm,
        generator=g,  # g might not be a fakedata param
    )
    if subset is not None and subset < len(ds):
        from torch.utils.data import Subset

        ds = Subset(ds, list(range(subset)))
    ntr = int(0.8 * len(ds))
    idx = list(range(len(ds)))
    return torch.utils.data.Subset(ds, idx[:ntr]), torch.utils.data.Subset(ds, idx[ntr:])


# image datasets
def build_mnsit(subset=None):
    assert tv is not None, "torchvision required for MNIST."
    tfm = tv.transforms.Compose([tv.transforms.ToTensor()])
    tr = tv.datasets.MNIST(root=".", train=False, download=True, transform=tfm)
    va = tv.datasets.MNIST(root=".", train=False, download=True, transform=tfm)
    if subset is not None and subset < len(tr):
        from torch.utils.data import Subset

        tr = Subset(tr, list(range(subset)))
    return tr, va


def build_cifar10(subset=None):
    assert tv is not None, "torchvision required for CIFAR10"
    tfm = tv.transforms.Compose([tv.transforms.ToTensor()])
    tr = tv.datasets.CIFAR10(root=".", train=True, download=True, transform=tfm)
    va = tv.datasets.CIFAR10(root=".", train=False, download=True, transform=tfm)
    if subset is not None and subset < len(tr):
        from torch.utils.data import Subset

        tr = Subset(tr, list(range(subset)))
    return tr, va


# -----------------------
# Train/Eval
# -----------------------
def _accuracy(logits, y):
    return (logits.argmax(1) == y).float().mean().item()


def _evaluate(model, loader, crit, dev):
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            logits = model(xb)
            loss = crit(logits, yb)
            total_loss += loss.item() * yb.numel()
            correct += (logits.argmax(1) == yb).long().sum().item()
            count += yb.numel()
        return {
            "loss": total_loss / max(count, 1),
            "acc": correct / max(count, 1),
        }


def train_and_eval(cfg: TrainConfig) -> Dict[str, Any]:
    cfg = cfg
    seed_everything(cfg.seed)
    dev = device()

    # seeding dataloader + workers:
    g = seed_everything(1337)

    # --- Data ---
    if cfg.dataset == "moons":
        train_ds, val_ds = build_synthetic_moons(n=1024, seed=cfg.seed)
        _, num_classes = (2,), 2
    elif cfg.dataset == "fakedata":
        train_ds, val_ds = build_fakedata(size=2000, seed=cfg.seed, subset=cfg.subset or 2000)
        _, num_classes = (3, 32, 32), 10
    elif cfg.dataset == "mnist":
        train_ds, val_ds = build_mnsit(subset=cfg.subset)
        _, num_classes = (1, 28, 28), 10
    elif cfg.dataset == "cifar10":
        train_ds, val_ds = build_cifar10(subset=cfg.subset)
        _, num_classes = (3, 32, 32), 10
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")

    # --- Model ---
    if cfg.model == "mlp" or (cfg.dataset == "moons" and cfg.model == "auto"):
        model = MLP(hidden=cfg.hidden, dropout=cfg.dropout)
    else:
        # Coerce to 3ch for simplicity (MNIST will be repeated to 3ch)
        model = TinyCNN(num_classes=num_classes, dropout=cfg.dropout)

    def _collate(batch):
        xs, ys = zip(*batch)
        x = torch.stack([t if t.ndim == 1 else t for t in xs])
        y = torch.tensor(ys, dtype=torch.long)
        if x.ndim == 4 and x.size(1) == 1:  # MNIST
            x = x.repeat(1, 3, 1, 1)
        return x, y

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        generator=g,
        worker_init_fn=seed_worker,
        pin_memory=cfg.pin_memory,
        collate_fn=_collate if cfg.dataset in {"mnist"} else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=256,
        shuffle=False,
        generator=g,
        worker_init_fn=seed_worker,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=_collate if cfg.dataset in {"mnist"} else None,
    )

    model.to(dev)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

    best_val = -1.0
    last_val = {}
    for _ in range(cfg.epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            logits = model(xb)
            loss = crit(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        last_val = _evaluate(model, val_loader, crit, dev)
        best_val = max(best_val, last_val["acc"])

    return {
        "val/acc": float(best_val),
        "val/loss": float(last_val.get("loss", 0.0)),
        "params": int(sum(p.numel() for p in model.parameters())),
        "dataset": cfg.dataset,
        "model_used": model.__class__.__name__,
    }


# ---------------------------
# ablate.py entrypoints
# ---------------------------
def run(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entry used by ablate.py -- keep this signature stable.
    """
    # Fill defualts via dataclass; allow unknown kets to override if present.
    cfg = TrainConfig(**{**TrainConfig().__dict__, **config_dict})
    return train_and_eval(cfg)


# for dry-runs/preflights


def preflight(cfg: dict) -> dict:
    try:
        # build model/dataset quickly, no downloads/writes
        model = MLP()
        model.eval()
        x = torch.zeros(1, 2)
        with torch.no_grad():
            _ = model(x)
            return {
                "ok": True,
                "params": sum(p.numel() for p in model.parameters()),
                "input_shape": list(x.shape),
                "artifacts": ["results.jsonl", "summary.csv", "ckpt.pt"],
            }
    except Exception as e:
        return {"ok": False, "error": str(e)}
