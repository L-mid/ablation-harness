from ablation_harness.tasks.diffusion.losses import ddpm_loss_with_info


def estimate_hutchinson_trace(
    model,
    x0,
    q,
    loss_cfg,
    curvature_cfg,
    device,
):
    """
    Estimate trace(H) for H = d^2 L / dθ^2 using Hutchinson's trick.

    Returns:
        {"mean": float, "std": float}
    """
    import torch

    probes = getattr(curvature_cfg, "probes", 8)

    # Keep original training/eval mode to restore later
    was_training = model.training
    model.eval()

    # Only trainable params
    params = [p for p in model.parameters() if p.requires_grad]

    traces = []

    for _ in range(probes):
        # Fresh graph each probe
        model.zero_grad(set_to_none=True)

        # 1) Forward + scalar loss (no extra per-t logging)
        loss, _ = ddpm_loss_with_info(
            model=model,
            x0=x0,
            q=q,
            loss_cfg=loss_cfg,
            log_per_t_mse=False,
        )

        # 2) First-order grads
        grads = torch.autograd.grad(
            loss,
            params,
            create_graph=True,  # keep graph to get Hessian-vector
            retain_graph=True,
            allow_unused=False,
        )

        # 3) Random Rademacher vector v (±1, same shapes as params)
        vs = [(torch.randint_like(p, low=0, high=2, device=device) * 2 - 1) for p in params]

        # 4) g·v
        g_dot_v = sum((g * v).sum() for g, v in zip(grads, vs))

        # 5) Hessian-vector product Hv = ∂(g·v)/∂θ
        hv = torch.autograd.grad(
            g_dot_v,
            params,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )

        # 6) Hutchinson estimate vᵀHv
        trace_est = sum((v * h).sum() for v, h in zip(vs, hv))

        traces.append(trace_est.detach())

    traces = torch.stack(traces)
    mean = traces.mean().item()
    std = traces.std(unbiased=False).item()

    # Restore mode and clear grads
    model.train(was_training)
    model.zero_grad(set_to_none=True)

    return {"mean": mean, "std": std}
