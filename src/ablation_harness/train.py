"""
Training entrypoint.

"""

from __future__ import annotations

import time
from typing import Any, Dict

import torch

from .builders import build_ema
from .checkpoint import save_best_if_better, save_last, try_resume  # 🔹
from .config_resolve import resolve_config
from .configs import RuntimeConfig, StudySpec
from .data import build_dataset
from .determinism import detect_flaky_ops  # 🔹
from .logging.jsonl_metric_logger import MetricLogger
from .logging.multi import build_logger
from .loop import evaluate, train_one_epoch  # 🔹
from .models.builder import build_model
from .optimizers import build_optimizer, build_scheduler
from .run_layout import resolve_run_layout  # 🔹
from .seed_utils import make_generator, seed_everything, seed_worker


def run(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Running in the trainer."""
    rt: RuntimeConfig
    spec: StudySpec
    rt, spec = resolve_config(config_dict)

    # 1) Setup
    seed_everything(rt.seed, deterministic=rt.deterministic)
    rank = int(torch.cuda.current_device() if torch.cuda.is_available() else 0)
    g = make_generator(rt.seed + rank)
    device = rt.device
    detect_flaky_ops(device=device)

    # 2) Paths/layout decided elsewhere
    layout = resolve_run_layout(base=rt.out_dir, run_id=rt.run_id, clean=rt.clean_run)

    # 3) Build data/model/optim/sched
    train_ds, val_ds, collate = build_dataset(rt)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=rt.batch_size,
        shuffle=True,
        num_workers=rt.num_workers,
        pin_memory=rt.pin_memory,
        generator=g,
        worker_init_fn=seed_worker,
        collate_fn=collate,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=256,
        shuffle=False,
        num_workers=rt.num_workers,
        pin_memory=rt.pin_memory,
        generator=g,
        worker_init_fn=seed_worker,
        collate_fn=collate,
    )

    model = build_model(rt).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = build_optimizer(rt, model)
    scheduler = build_scheduler(rt, optimizer, train_loader)
    ema = build_ema(model, rt)

    # 4) Logging
    logger = build_logger(spec.logging, run_dir=layout.root)  # ensures TB/W&B point to run_dir
    metrics_path = layout.results.with_suffix(".jsonl")  # beware, not all take results.jsonl for logging.
    logger.on_run_start(rt)

    # 5) (Optional) resume
    start_epoch, best_val = 0, float("-inf")
    state = try_resume(layout)
    if state:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        ema.load_state_dict(state["ema"])
        start_epoch = int(state["epoch"]) + 1
        best_val = float(state.get("best_val", best_val))

    # 6) Train
    t0 = time.perf_counter()
    last_val = {}
    with MetricLogger(str(metrics_path), fmt="jsonl") as mlog:
        for epoch in range(start_epoch, rt.epochs):
            logger.on_epoch_start(epoch)
            train_stats = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, ema, device, rt, logger, mlog)
            last_val = evaluate(model, val_loader, criterion, device, ema)
            # log scalar metrics
            mlog.log(train_stats["global_step"], epoch=epoch, **{f"val/{k}": float(v) for k, v in last_val.items()})
            logger.log_metrics({f"val/{k}": float(v) for k, v in last_val.items()})
            # checkpoints
            new_best = save_best_if_better(layout, model, optimizer, ema, epoch, best_val, metric=float(last_val["acc"]))
            best_val = new_best
            save_last(layout, model, optimizer, ema, epoch, best_val)
            logger.on_epoch_end(epoch)

    logger.on_run_end()

    return {
        "seed": rt.seed,
        "val/acc": float(best_val),
        "val/loss": float(last_val.get("loss", 0.0)),
        "params": int(sum(p.numel() for p in model.parameters())),
        "dataset": rt.dataset,
        "model_used": model.__class__.__name__,
        "run_id": layout.run_id,
        "run_dir": str(layout.root),
        "ckpt_last": str(layout.ckpts / "last.pt"),
        "ckpt_best": str(layout.ckpts / "best_val.pt"),
        "loss_log": str(metrics_path),
        "run_time_s": time.perf_counter() - t0,
    }
