"""
Training entrypoint.

"""

from __future__ import annotations

import subprocess
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
    """Prefer explicit flag set by resolve_config; fallback to model name check."""
    task = getattr(rt, "task", None)
    if task is not None:
        return task == "diffusion"  # returns true or false
    return False


def _get_git_hash() -> str | None:
    """Return the current git commit hash, or None if not in a git repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip()
    except Exception:
        return None


def _compute_grad_stats(model) -> Dict[str, float]:
    """Simple global gradient stats for diagnostics."""

    total_norm_sq = 0.0
    abs_chunks = []

    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        total_norm_sq += float(g.norm(2).item() ** 2)
        abs_chunks.append(g.abs().reshape(-1))

    if not abs_chunks:
        return {}

    abs_all = torch.cat(abs_chunks)
    return {
        "train/grad_global_L2": total_norm_sq**0.5,
        "train/grad_abs_mean": float(abs_all.mean().item()),
        "train/grad_abs_max": float(abs_all.max().item()),
    }


def run(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Running in the trainer."""
    rt: RuntimeConfig
    spec: StudySpec
    rt, spec = resolve_config(config_dict)

    # --- git hash ---
    git_hash = _get_git_hash()

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
        shuffle=rt.shuffle,
        num_workers=rt.num_workers,
        pin_memory=rt.pin_memory,
        generator=g,
        worker_init_fn=seed_worker,
        collate_fn=collate,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=256,
        shuffle=rt.shuffle,
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
        return _run_diffusion(rt, spec, device, g, layout, train_loader, logger, metrics_path, git_hash)
    else:
        return _run_classification(rt, device, layout, train_loader, val_loader, logger, metrics_path, git_hash)


def _run_classification(rt, device, layout, train_loader, val_loader, logger, metrics_path, git_hash):
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
            metric = float(last_val["acc"])
            new_best = save_best_if_better(layout, model, optimizer, ema, epoch, best_val, metric)
            # new_best is (best_epoch, best_metric); we only keep the metric
            _, best_val = new_best
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
        "git_hash": git_hash,
    }


def _run_diffusion(rt, spec, device, g, layout, train_loader, logger, metrics_path, git_hash):  # noqa C901
    """
    Minimal diffusion runner: K=1000 training steps, subset-NFE sampling for eval,
    EMA eval weights, and checkpoint on -FID (so 'higher is better' still holds).
    """
    import os
    import time

    from .builders import build_ema
    from .checkpoint import metric_from_fid, save_best_if_better, save_last, try_resume
    from .logging.jsonl_metric_logger import MetricLogger
    from .metrics.hutchinsion_trace import estimate_hutchinson_trace
    from .optimizers import build_optimizer
    from .tasks.diffusion.losses import (
        compute_snr_from_alphas_cumprod,
        ddpm_loss_with_info,
    )
    from .tasks.diffusion.models.unet_cifar32 import build_unet_model
    from .tasks.diffusion.schedule import get_beta_schedule, precompute_q

    # ----- Build model/optimizer/EMA -----
    model = build_unet_model(spec).to(device)
    optimizer = build_optimizer(rt, model)
    ema = build_ema(model, rt)

    # Diffusion schedule
    K = 1000
    betas = get_beta_schedule(getattr(rt, "beta_schedule", "linear"), K, device=device)
    q = precompute_q(betas)

    # --- LOG: diffusion build confirmation ---
    msg = f"[diffusion] diffusion_enabled=True, " f"beta_schedule={getattr(rt, 'beta_schedule', 'linear')}, " f"K={K}"
    print(msg)

    # Steps config
    total_steps = getattr(rt, "total_steps", 1)  # set this from adapter/resolve_config
    print("[debug] total_steps:", total_steps)
    log_every = spec.logging.log_every_n_steps

    # Resume
    best_metric = (0, float("-inf"))  # we’ll store -FID here
    state = try_resume(layout)
    if state:
        best_metric = float(state.get("best_val", best_metric))
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

        # --- Optional: log Min-SNR theoretical weight curve at step 0 ---
        loss_cfg = getattr(spec, "loss", None)
        if getattr(loss_cfg, "weighting", "constant") == "minsnr":
            with torch.no_grad():
                alpha_bar = q["alpha_bar"]  # [K]
                K = alpha_bar.shape[0]
                t_all = torch.arange(K, device=alpha_bar.device, dtype=torch.long)
                snr_all = compute_snr_from_alphas_cumprod(alpha_bar, t_all)
                gamma = torch.as_tensor(
                    getattr(loss_cfg, "minsnr_gamma", 5.0),
                    dtype=snr_all.dtype,
                    device=snr_all.device,
                )
                weight_all = torch.minimum(snr_all, gamma) / snr_all.clamp(min=1e-12)

            curve_metrics = {
                # arrays get stored as JSON lists in the jsonl logger
                "mins_snr_curve/t": [int(x) for x in t_all.tolist()],
                "mins_snr_curve/weight": [float(x) for x in weight_all.tolist()],
            }
            logger.log_metrics({**curve_metrics, "step": 0})
            mlog.log(0, epoch=0, **curve_metrics)

        loss_cfg = getattr(spec, "loss", None)
        curvature_cfg = getattr(spec, "curvature", None)

        for batch in cycle(train_loader):
            global_step += 1
            print_global_step = 100

            if (global_step % print_global_step) == 0:
                print("[train.py] Current step is:", global_step)

            model.train()

            # Pull images (normalize to [-1,1] if your dataset isn't already)
            images, labels = batch
            x0 = images.to(device)
            # x0 = x0 * 2.0 - 1.0  # uncomment if your loader gives [0,1]

            log_this_step = (global_step % log_every) == 0

            loss, loss_info = ddpm_loss_with_info(
                model=model,
                x0=x0,
                q=q,
                loss_cfg=getattr(spec, "loss", None),
                log_per_t_mse=log_this_step,  # 🔹 only compute per-t MSE on logging steps
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if rt.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), rt.grad_clip)

            # Grad stats AFTER clipping, BEFORE the step
            grad_stats = _compute_grad_stats(model)

            # --- Hutchinson curvature probe (optional) ---
            curvature_metrics = {}
            if curvature_cfg is not None and getattr(curvature_cfg, "enabled", False):
                if (global_step % log_every) == 0:
                    hutch_stats = estimate_hutchinson_trace(
                        model=model,
                        x0=x0,
                        q=q,
                        loss_cfg=loss_cfg,
                        curvature_cfg=curvature_cfg,
                        device=device,
                    )
                    prefix = getattr(curvature_cfg, "log_prefix", "curvature/hutch")
                    curvature_metrics = {
                        f"{prefix}_trace_mean": float(hutch_stats["mean"]),
                        f"{prefix}_trace_std": float(hutch_stats["std"]),
                    }

            optimizer.step()
            if getattr(rt, "ema_enabled", True):
                ema.update(model)

            # --- Logging ---
            if (global_step % log_every) == 0:
                train_metrics = {
                    "train/loss": float(loss.detach().cpu()),
                    **grad_stats,
                    **loss_info,
                    **curvature_metrics,
                }

                logger.log_metrics({**train_metrics, "step": global_step})
                mlog.log(global_step, epoch=0, **train_metrics)

                # === EVAL: schedule & run (grid / kid / fid_milestone) ===
                E = spec.eval  # EvalCfg with nested sections we added earlier

                do_grid = getattr(E, "grid", None) and E.grid.enabled and (global_step % E.grid.every == 0)
                do_kid = getattr(E, "kid", None) and E.kid.enabled and (global_step % E.kid.every == 0)
                do_fidM = getattr(E, "fid_milestone", None) and E.fid_milestone.enabled and (global_step % E.fid_milestone.every == 0)

                if do_grid or do_kid or do_fidM:
                    # Build an EMA eval copy only once for all tasks due this step
                    model_eval = build_unet_model(spec).to(device)
                    if getattr(rt, "ema_enabled", True):
                        ema.copy_to(model_eval)
                    else:
                        model_eval.load_state_dict(model.state_dict())

                    with torch.no_grad():
                        s = 0.0
                        n = 0
                        for p_tr, p_ev in zip(model.parameters(), model_eval.parameters()):
                            s += (p_tr - p_ev).abs().mean().item()
                            n += 1
                        print(f"[debug (train.py)] mean |model - model_eval| per-param avg: {s / n:.6f}")

                step_dir = os.path.join(layout.root, "eval", f"step_{global_step:06d}")
                os.makedirs(step_dir, exist_ok=True)

                # Grid (cheap visual check)
                if do_grid:
                    out = evaluate_diffusion(model_eval, E, q, os.path.join(step_dir, "grid"), task="grid")
                    # (No metrics to checkpoint against; it just writes a grid.)

                # KID (moderate; we’ll log it if you wire real KID later)
                kid_now = None
                if do_kid:
                    out = evaluate_diffusion(model_eval, E, q, os.path.join(step_dir, "kid"), task="kid")
                    kid_now = out.get("kid", None)
                    if kid_now is not None:
                        logger.log_metrics({"val/kid": float(kid_now), "step": global_step})
                        mlog.log(global_step, epoch=0, **{"val/kid": float(kid_now)})

                # FID milestone (heavy, gated internally by best-KID if you enabled that)
                if do_fidM:
                    out = evaluate_diffusion(model_eval, E, q, os.path.join(step_dir, "fid_milestone"), task="fid_milestone")
                    fid_val = out.get("fid", None)
                    if fid_val is not None:
                        logger.log_metrics({"val/fid": float(fid_val), "step": global_step})
                        mlog.log(global_step, epoch=0, **{"val/fid": float(fid_val)})

                        # Checkpoint **on FID only** so “best” reflects an external metric
                        metric_for_ckpt = metric_from_fid(fid_val)  # converts FID to a "higher is better" score
                        best_metric = save_best_if_better(layout, model, optimizer, ema, global_step, best_metric, metric_for_ckpt)

                # Always keep a rolling "last" checkpoint at eval boundaries (optional but handy)
                save_last(layout, model, optimizer, ema, global_step, best_metric[1])

            if global_step >= total_steps:
                break

    # === FINAL EVAL AT TRAIN END ===
    model_eval = build_unet_model(spec).to(device)
    if getattr(rt, "ema_enabled", True):
        ema.copy_to(model_eval)
    else:
        model_eval.load_state_dict(model.state_dict())

    # --- LOG: final evaluation configuration ---
    E = spec.eval
    FE = getattr(E, "final", None)

    if FE and FE.enabled:
        sampler = FE.sampler
        nfe = FE.nfe
        n_samples = FE.n_samples

        msg = f"[eval.final] Running final evaluation: " f"sampler={sampler}, nfe={nfe}, n_samples={n_samples}"
        print(msg)

    final_dir = os.path.join(layout.root, "eval", "final")
    out = evaluate_diffusion(model_eval, spec.eval, q, final_dir, task="final")

    # If your final task computes FID (once you wire it), you can also log/checkpoint here:
    fid_final = out.get("fid", None)
    if fid_final is not None:
        logger.log_metrics({"val/fid_final": float(fid_final)})
        # Optional: update best on final too
        metric_for_ckpt = metric_from_fid(fid_final)
        best_metric = save_best_if_better(layout, model, optimizer, ema, global_step, best_metric, metric_for_ckpt)

    save_last(layout, model, optimizer, ema, global_step, best_metric[1])

    logger.on_run_end()

    # Compose return dict; expose best FID if known
    _, best_val = best_metric  # best_val is the "higher is better" metric = -FID
    best_fid = -best_val if best_val != float("-inf") else None
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
        "git_hash": git_hash,
    }
