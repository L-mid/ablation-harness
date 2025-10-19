from typing import Any, Optional

import torch
from torch.optim.lr_scheduler import _LRScheduler


def build_optimizer(cfg, model) -> torch.optim.Optimizer:
    """Builds chosen str optimizer from cfg."""
    if cfg.opt_name == "adam":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    elif cfg.opt_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd, momentum=cfg.momentum)
    else:
        # AdamW as default Optim
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

    return optimizer


def build_scheduler(cfg, opt, train_loader) -> Optional[_LRScheduler | Any]:
    """Returns some base sched for Optim lr."""
    sched = None
    if cfg.sched_name == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    elif cfg.sched_name == "step":
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=cfg.step_size, gamma=cfg.gamma)
    elif cfg.sched_name == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=cfg.lr,
            steps_per_epoch=len(train_loader),
            epochs=cfg.epochs,
        )

    return sched
