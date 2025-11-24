import torch


def train_one_epoch(model, loader, criterion, opt, sched, ema, dev, cfg, logger, mlog):
    """Train a single epoch. (training + logging when settings)"""
    model.train()
    global_step = 0
    for xb, yb in loader:
        xb, yb = xb.to(dev), yb.to(dev)
        logits = model(xb)
        loss = criterion(logits, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        ema.update(model)
        if sched and hasattr(sched, "step") and sched.__class__.__name__ in {"OneCycleLR", "CyclicLR"}:
            sched.step()
        global_step += 1
        if logger.should_log(global_step):
            mlog.log(global_step, **{"train/loss": float(loss.item())})
            logger.log_metrics({"train/loss": float(loss.item())})
    if sched and sched.__class__.__name__ not in {"OneCycleLR", "CyclicLR"}:
        sched.step()
    return {"global_step": global_step}


@torch.no_grad()
def evaluate(model, loader, criterion, dev, ema):
    """Sets model to eval and evaluates it against test split of data."""
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    if ema is not None:
        ema.copy_to(model)
    for xb, yb in loader:
        xb, yb = xb.to(dev), yb.to(dev)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * yb.numel()
        correct += (logits.argmax(1) == yb).long().sum().item()
        count += yb.numel()
    return {"loss": total_loss / max(count, 1), "acc": correct / max(count, 1)}
