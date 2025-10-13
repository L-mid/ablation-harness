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
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ablation_harness.builders import build_ema
from ablation_harness.jsonl_metric_logger import MetricLogger
from ablation_harness.logger import build_logger
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


# LOGGING: stub, not being accessed correctly (not affected by yaml). Currently uses dataclasses as law.
@dataclass
class WandbCfg:
    project: str = "ablation-harness"
    entity: Optional[str] = None
    run_name: Optional[str] = "wk2_adam_sgd_ema_param_sweep"  # remeber to replace this with 'generic name'
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    mode: Literal["online", "offline", "disabled"] = "online"  # turn off for CLI
    # wandb dir gets pointed at run_dir by the logger builder


@dataclass
class TensorBoardCfg:
    flush_secs: int = 10


@dataclass
class LoggingCfg:
    enable: bool = True
    dir: str = "runs/logs"  # this dir is not overrided rn
    backends: list[str] = field(default_factory=lambda: ["wandb", "tensorboard"])  # wandb cannot be enabled in CLI. Remove it.
    wandb: WandbCfg = field(default_factory=WandbCfg)
    tensorboard: TensorBoardCfg = field(default_factory=TensorBoardCfg)
    log_every_n_steps: int = 10


@dataclass
class TrainConfig:
    model: str = "mlp"  # "mlp" | "tinycnn"
    hidden: int = 64
    dropout: float = 0.0
    lr: float = 1e-3
    wd: float = 0.0
    scheduler: str = "cosine"
    momentum: int = 0
    epochs: int = 4
    batch_size: int = 64
    seed: int = 0
    dataset: str = "moons"  # "moons" | "fakedata" | "mnist" | "cifar10"
    subset: Optional[int] = None  # e.g., 1000
    num_workers: int = 0
    pin_memory: bool = False
    out_dir: str = "runs/any_logs"  # for checkpointing + (make run dir in gernal)
    run_id: str = "generic_any_id"
    _study: Optional[str] = None
    _variant: Optional[str] = None
    optimizer: Any = None
    ema: Any = None
    decay: float = 0.9999

    spectral_diag: SpectralDiagCfg = field(default_factory=SpectralDiagCfg)
    logging: LoggingCfg = field(default_factory=LoggingCfg)


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


def _evaluate(model, loader, crit, dev, ema):
    """
    Evaluates.

    Note for EMA:
        Evaluation: Always report both live and EMA once or twice to learn how much stability you gain;
        for your weekly report, pick one (usually EMA) and be consistent.
    """

    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    with torch.no_grad(), ema.apply_to(model):
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

    def _choose_optimizer(cfg):
        if cfg.optimizer == "adam":
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
        elif cfg.optimizer == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd, momentum=cfg.momentum)
        else:
            # AdamW as default Optim
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

        return optimizer

    def _choose_lr_sched(cfg):
        """Returns some base sched for Optim lr."""
        sched = None
        if cfg.scheduler == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
        elif cfg.scheduler == "step":
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=cfg.step_size, gamma=cfg.gamma)
        elif cfg.scheduler == "onecycle":
            sched = torch.optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=cfg.lr,
                steps_per_epoch=len(train_loader),
                epochs=cfg.epochs,
            )

        return sched

    # should be a train.py?

    model.to(dev)
    crit = nn.CrossEntropyLoss()
    opt = _choose_optimizer(cfg)
    sched = _choose_lr_sched(cfg)

    rid, run_dir = _make_run_dir(cfg)
    ema = build_ema(model, cfg)

    # ckpts
    ckpt_dir = os.path.join(run_dir, "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "ckpts.pt")

    # logging
    logger = build_logger(cfg)
    logger.on_run_start(cfg)

    loss_log_path = os.path.join(run_dir, "loss.jsonl")

    # loss + metric extra stats
    best_val = float("-inf")
    last_val = {}
    spect_stats = None

    global_step = 0

    with MetricLogger(loss_log_path, fmt="jsonl") as mlog:
        for epoch in range(cfg.epochs):
            logger.on_epoch_start(epoch)
            print(f"[trainer.py:] current epoch: {epoch}")
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                logits = model(xb)
                loss = crit(logits, yb)

                # backward, step, etc.
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                ema.update(model)

                # per-batch schedulers:
                if isinstance(sched, (torch.optim.lr_scheduler.OneCycleLR, torch.optim.lr_scheduler.CyclicLR)):
                    sched.step()

                global_step += 1

                if global_step % cfg.logging.log_every_n_steps == 0:
                    mlog.log(global_step, **{"train/loss": float(loss.item()), "epoch": epoch})  # saves locally
                    logger.log_metrics(
                        {
                            "train/loss": float(loss.item()),
                            "train/lr": float(getattr(sched, "get_last_lr", lambda: [opt.param_groups[0]["lr"]])()[0]),
                        },
                        step=global_step,
                    )

            last_val = _evaluate(model, val_loader, crit, dev, ema)

            # per-epoch schedulers:
            if sched and not isinstance(sched, (torch.optim.lr_scheduler.OneCycleLR, torch.optim.lr_scheduler.CyclicLR)):
                if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    sched.step(last_val["loss"])  # after you compute it
                else:
                    sched.step()

            # spectral:
            if cfg.spectral_diag and cfg.spectral_diag.enabled:
                if epoch % cfg.spectral_diag.every_n_epochs == 0:
                    spect_stats = collect_spectral_stats(model, topk=cfg.spectral_diag.topk)

                    # dump to disk for now since no logger yet OR skip.
                    save_dir = cfg.spectral_diag.save_dir or run_dir
                    os.makedirs(save_dir, exist_ok=True)
                    out_path = os.path.join(save_dir, f"epoch_spectrals/spectral_epoch{epoch:03d}.json")
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump({"epoch": epoch, "stats": spect_stats}, f, ensure_ascii=False, indent=2)  # local save

                        # to logger:
                        logger.log_artifact(out_path, name="spectral_stats.jsonl")

            if last_val["acc"] >= best_val:
                best_val = last_val["acc"]  # small ckpt skel, saves all (not scheduler)
                ckpt = {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "ema": ema.state_dict(),
                    "epoch": epoch,
                }
                torch.save(
                    {"model_state": ckpt, "cfg": cfg.__dict__, "val": last_val},
                    ckpt_path,
                )
            """
            Useage of loading this checkpointer (ema specifically):

                ckpt = torch.load("checkpoint.pt", map_location="cpu")
                model.load_state_dict(ckpt["model"])
                optimizer.load_state_dict(ckpt["optimizer"])
                ema = EMA(model, EMAConfig(decay=0.999))
                ema.load_state_dict(ckpt["ema"])
            """

            # saving loss to the loss.jsonl explicitly (local):
            mlog.log(global_step, **{"val/loss": float(last_val["loss"]), "val/acc": float(last_val["acc"]), "epoch": epoch})

            logger.log_metrics({"val/loss": float(last_val["loss"]), "val/acc": float(last_val["acc"])}, step=global_step)
            logger.on_epoch_end(epoch)
        # (Optional) log files/plots you generate
        # logger.log_artifact("runs/.../loss.png", name="loss_plot.png")

        # spectral:
        if cfg.spectral_diag and cfg.spectral_diag.enabled:
            final_stats = collect_spectral_stats(model, topk=9999)  # big number == "as many as exist"
            # another todo logger hook:
            save_dir = cfg.spectral_diag.save_dir or run_dir
            with open(os.path.join(save_dir, "spectral_final.json"), "w", encoding="utf-8") as f:
                json.dump({"epoch": "final", "stats": final_stats}, f, ensure_ascii=False, indent=2)

        logger.on_run_end()

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

    from ablation_harness.config_resolve import resolve_config

    cfg: TrainConfig = resolve_config(config_dict)

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
