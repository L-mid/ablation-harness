"""
Training entrypoint.

"""

from __future__ import annotations

from typing import Any, Dict

import torch

from .config_resolve import resolve_config
from .configs import RuntimeConfig, StudySpec
from .data import build_dataset
from .determinism import detect_flaky_ops  # 🔹
from .eval.generative import evaluate_diffusion
from .logging.multi import build_logger
from .run_layout import resolve_run_layout  # 🔹
from .seed_utils import make_generator, seed_everything, seed_worker


def _is_diffusion(rt) -> bool:
    # Prefer explicit flag set by resolve_config; fallback to model name check.
    task = getattr(rt, "task", None)
    if task is not None:
        return task == "diffusion"
    name = getattr(rt, "model_name", "") or getattr(getattr(rt, "model", None), "name", "")
    return "unet" in str(name).lower()


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
        shuffle=rt.data_shuffle,
        num_workers=rt.num_workers,
        pin_memory=rt.pin_memory,
        generator=g,
        worker_init_fn=seed_worker,
        collate_fn=collate,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=256,
        shuffle=rt.data_shuffle,
        num_workers=rt.num_workers,
        pin_memory=rt.pin_memory,
        generator=g,
        worker_init_fn=seed_worker,
        collate_fn=collate,
    )

    # 4) Logging
    logger = build_logger(spec.logging, run_dir=layout.root)  # ensures TB/W&B point to run_dir
    metrics_path = layout.results.with_suffix(".jsonl")  # beware, not all take results.jsonl for logging.
    logger.on_run_start(rt)

    # ----
    # NEW: branch for diffusion vs classification ----
    if _is_diffusion(rt):
        return _run_diffusion(rt, spec, device, g, layout, train_loader, logger, metrics_path)
    else:
        return _run_classification(rt, spec, device, g, layout, train_loader, val_loader, logger, metrics_path)


def _run_classification(rt, spec, device, g, layout, train_loader, val_loader, logger, metrics_path):
    import time

    from .builders import build_ema
    from .checkpoint import save_best_if_better, save_last, try_resume
    from .logging.jsonl_metric_logger import MetricLogger
    from .loop import evaluate, train_one_epoch
    from .models.builder import build_model
    from .optimizers import build_optimizer, build_scheduler

    model = build_model(rt).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = build_optimizer(rt, model)
    scheduler = build_scheduler(rt, optimizer, train_loader)
    ema = build_ema(model, rt)

    start_epoch, best_val = 0, float("-inf")
    state = try_resume(layout)
    if state:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        ema.load_state_dict(state["ema"])
        start_epoch = int(state["epoch"]) + 1
        best_val = float(state.get("best_val", best_val))

    t0 = time.perf_counter()
    last_val = {}
    with MetricLogger(str(metrics_path), fmt="jsonl") as mlog:
        for epoch in range(start_epoch, rt.epochs):
            logger.on_epoch_start(epoch)
            train_stats = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, ema, device, rt, logger, mlog)
            last_val = evaluate(model, val_loader, criterion, device, ema)
            mlog.log(train_stats["global_step"], epoch=epoch, **{f"val/{k}": float(v) for k, v in last_val.items()})
            logger.log_metrics({f"val/{k}": float(v) for k, v in last_val.items()})
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


def _run_diffusion(rt, spec, device, g, layout, train_loader, logger, metrics_path):
    """
    Minimal diffusion runner: K=1000 training steps, subset-NFE sampling for eval,
    EMA eval weights, and checkpoint on -FID (so 'higher is better' still holds).
    """
    import os
    import time

    from .builders import build_ema
    from .checkpoint import save_best_if_better, save_last, try_resume
    from .diffusion.core import ddpm_loss, get_beta_schedule, precompute_q
    from .logging.jsonl_metric_logger import MetricLogger
    from .models.unet_cifar32 import build_unet_model
    from .optimizers import build_optimizer

    # ----- Build model/optimizer/EMA -----
    model = build_unet_model(rt).to(device)  # uses your UNet builder
    optimizer = build_optimizer(rt, model)
    ema = build_ema(model, rt)

    # Diffusion schedule
    K = 1000
    betas = get_beta_schedule(getattr(rt, "beta_schedule", "linear"), K, device=device)
    q = precompute_q(betas)

    # Steps config
    total_steps = getattr(rt, "total_steps", 10_000)  # set this from your adapter/resolve_config
    log_every = spec.logging.log_every_n_steps
    eval_every = getattr(rt, "eval_every", 5_000)  # choose via config; fallback ok

    # Resume
    best_metric = 0, float("-inf")  # we’ll store -FID here
    state = try_resume(layout)
    if state:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        ema.load_state_dict(state["ema"])
        start_epoch = int(state["epoch"]) + 1
        best_metric = float(state.get("best_val", best_metric))

    # Train loop (by steps)
    t0 = time.perf_counter()
    global_step = 0
    from itertools import cycle

    with MetricLogger(str(metrics_path), fmt="jsonl") as mlog:
        for batch in cycle(train_loader):
            global_step += 1
            model.train()

            # Pull images (normalize to [-1,1] if your dataset isn't already)
            x0 = batch["images"].to(device)
            # x0 = x0 * 2.0 - 1.0  # uncomment if your loader gives [0,1]

            loss = ddpm_loss(model, x0, q)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if rt.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), rt.grad_clip)
            optimizer.step()
            if getattr(rt, "ema_enabled", True):
                ema.update()

            # Logging
            if (global_step % log_every) == 0:
                logger.log_metrics({"train/loss": float(loss.detach().cpu()), "step": global_step})
                mlog.log(global_step, epoch=0, **{"train/loss": float(loss.detach().cpu())})

            # Periodic eval (EMA weights if available)
            if (global_step % eval_every) == 0:
                model_eval = build_unet_model(rt).to(device)
                if getattr(rt, "ema_enabled", True):
                    ema.copy_to(model_eval)
                else:
                    model_eval.load_state_dict(model.state_dict())

                eval_out_dir = os.path.join(layout.root, "eval", f"step_{global_step:06d}")
                scores = evaluate_diffusion(model_eval, spec.eval, q, eval_out_dir)
                last_eval = scores
                fid = scores.get("fid", None)
                if fid is not None:
                    logger.log_metrics({"val/fid": float(fid), "step": global_step})
                    mlog.log(global_step, epoch=0, **{"val/fid": float(fid)})

                # Checkpoint on -FID so 'higher is better' semantics remain
                metric_for_ckpt = -float(fid) if (fid is not None) else float("-inf")
                new_best = save_best_if_better(layout, model, optimizer, ema, 0, best_metric, metric=metric_for_ckpt)
                best_metric = new_best
                save_last(layout, model, optimizer, ema, 0, best_metric)

            if global_step >= total_steps:
                break

    logger.on_run_end()

    # Compose return dict; expose best FID if known
    best_fid = (-best_metric) if best_metric != float("-inf") else None
    return {
        "seed": rt.seed,
        "val/fid": float(best_fid) if best_fid is not None else None,
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
