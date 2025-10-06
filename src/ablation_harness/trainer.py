"""
TODO:
    Change the roots in datasets (out of repo).
    _make_run_dir: goes to arbirary place. Get a logger working.
    Got big and complicated fast: break out pieces.
    Not all datasets are tested to be working: remove or test.
    batch spectrals


"""

import json
import os
from dataclasses import dataclass, replace
from types import ModuleType
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ablation_harness.jsonl_metric_logger import MetricLogger
from ablation_harness.seed_utils import make_generator, seed_everything, seed_worker
from ablation_harness.spectral import collect_spectral_stats

try:
    import torchvision as tv
except Exception:
    tv: Optional[ModuleType] = None


# -------------------------
# Config
# -------------------------


@dataclass
class SpectralDiagCfg:
    enabled: bool = False
    every_n_epochs: int = 1
    topk: int = 5
    save_dir: Optional[str] = None  # if None, use run_dir


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
    out_dir: Optional[str] = "runs/logs"  # for checkpointing + (make run dir in gernal)
    run_id: Optional[str] = None
    _study: Optional[str] = None
    _variant: Optional[str] = None
    log_every: int = 1
    spectral_diag: Optional[SpectralDiagCfg] = None


# --------------------------
# Utils
# --------------------------


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
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x)  # [B, 32, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 32]
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


def _make_run_dir(cfg: TrainConfig) -> Tuple[str, str]:
    """
    right now, dirs are solely handled by ablate.py:
    using the --out_dir CI command. --out_dir OUT

    which is parsed to:

    os.makedirs(args.out_dir, exist_ok=True)
    jsonl_path = os.path.join(args.out_dir, "results.jsonl")
    csv_path = os.path.join(args.out_dir, "summary.csv")


    Basically, this _make_run_dir is not hooked up and goes to runs/ckpts (or outdir).
    Outputs from this file should be logged in future, not saved to disk (ckpts idk).
    """

    base = cfg.out_dir
    rid = cfg.run_id
    run_dir = os.path.join(base, rid)
    os.makedirs(run_dir, exist_ok=True)
    return rid, run_dir


def train_and_eval(cfg: TrainConfig) -> Dict[str, Any]:  # noqa: C901
    cfg = cfg
    global_seed = cfg.seed

    rank = int(os.environ.get("RANK", 0))
    process_seed = global_seed + rank
    seed_everything(cfg.seed, deterministic=getattr(cfg, "deterministic", True))
    seed_everything(process_seed, deterministic=getattr(cfg, "deterministic", True))

    g = make_generator(process_seed)

    dev = device()

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

    # should be a train_loop.py?

    model.to(dev)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    global_step = 0

    rid, run_dir = _make_run_dir(cfg)
    ckpt_path = os.path.join(run_dir, "ckpts.pt")

    log_every = getattr(cfg, "log_every", 1)
    loss_log_path = os.path.join(run_dir, "loss.jsonl")
    best_val = float("-inf")
    last_val = {}
    spect_stats = None

    with MetricLogger(loss_log_path, fmt="jsonl") as mlog:
        for epoch in range(cfg.epochs):
            print(f"[trainer.py:] current epoch: {epoch}")
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                logits = model(xb)
                loss = crit(logits, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                global_step += 1

                if global_step % log_every == 0:
                    mlog.log(global_step, **{"train/loss": float(loss.item())})

            last_val = _evaluate(model, val_loader, crit, dev)

            # spectral:
            if cfg.spectral_diag and cfg.spectral_diag.enabled:
                if epoch % cfg.spectral_diag.every_n_epochs == 0:
                    spect_stats = collect_spectral_stats(model, topk=cfg.spectral_diag.topk)

                    # dump to disk for now since no logger yet OR skip.
                    save_dir = cfg.spectral_diag.save_dir or run_dir
                    os.makedirs(save_dir, exist_ok=True)
                    out_path = os.path.join(save_dir, f"epoch_spectrals/spectral_epoch{epoch:03d}.json")
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump({"epoch": epoch, "stats": spect_stats}, f, ensure_ascii=False, indent=2)

            if last_val["acc"] >= best_val:
                best_val = last_val["acc"]  # small ckpt skel, saves only model
                torch.save(
                    {"model_state": model.state_dict(), "cfg": cfg.__dict__, "val": last_val},
                    ckpt_path,
                )

        # spectral:
        if cfg.spectral_diag and cfg.spectral_diag.enabled:
            final_stats = collect_spectral_stats(model, topk=9999)  # big number == "as many as exist"
            # another todo logger hook:
            save_dir = cfg.spectral_diag.save_dir or run_dir
            with open(os.path.join(save_dir, "spectral_final.json"), "w", encoding="utf-8") as f:
                json.dump({"epoch": "final", "stats": final_stats}, f, ensure_ascii=False, indent=2)

        return {
            "seed": cfg.seed,
            "val/acc": float(best_val),
            "val/loss": float(last_val.get("loss", 0.0)),
            "params": int(sum(p.numel() for p in model.parameters())),
            "dataset": cfg.dataset,
            "model_used": model.__class__.__name__,
            "run_id": rid,
            "run_dir": run_dir,
            "ckpt": ckpt_path,
            "spect_stats": spect_stats,
            "loss_log": loss_log_path,
        }


# ---------------------------
# ablate.py entrypoints
# ---------------------------
def run(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entry used by ablate.py -- keep this signature stable.
    """
    # Fill defualts via dataclass. Does not allow unknown keys.
    defaults = TrainConfig()
    sd = config_dict.get("spectral_diag")
    if isinstance(sd, dict):
        config_dict = {**config_dict, "spectral_diag": SpectralDiagCfg(**sd)}
    cfg = replace(defaults, **config_dict)
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
        print("[trainer.py: WARN: error this run]")
        return {"ok": False, "error": str(e)}
