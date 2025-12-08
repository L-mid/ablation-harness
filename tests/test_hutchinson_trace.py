import math
from types import SimpleNamespace

import torch
from torch import nn

from ablation_harness.metrics import hutchinsion_trace as ht


def test_hutchinson_trace_matches_diagonal_quadratic_trace(monkeypatch):
    """
    For L(w) = 0.5 * (a1 * w1^2 + a2 * w2^2), the Hessian is diag(a1, a2),
    so trace(H) = a1 + a2. Hutchinson should recover this exactly for any v.
    """

    device = torch.device("cpu")

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            # 2D parameter vector
            self.w = nn.Parameter(torch.tensor([0.3, -1.7], dtype=torch.float32))

    model = TinyModel().to(device)

    # Diagonal curvature coefficients ⇒ Hessian = diag(a1, a2)
    a = torch.tensor([1.0, 4.0], dtype=torch.float32, device=device)  # trace = 5.0

    # Fake ddpm_loss_with_info that ignores x0, q, loss_cfg and uses our quadratic
    def fake_ddpm_loss_with_info(model, x0, q, loss_cfg, log_per_t_mse=False):
        w = model.w
        loss = 0.5 * (a * w.pow(2)).sum()
        info = {}
        return loss, info

    # Patch the symbol your helper imports at module level
    monkeypatch.setattr(ht, "ddpm_loss_with_info", fake_ddpm_loss_with_info)

    curvature_cfg = SimpleNamespace(probes=7)  # any >1; each probe is exact here
    loss_cfg = SimpleNamespace()  # unused by our fake loss
    x0 = None
    q = None

    stats = ht.estimate_hutchinson_trace(
        model=model,
        x0=x0,
        q=q,
        loss_cfg=loss_cfg,
        curvature_cfg=curvature_cfg,
        device=device,
    )

    trace_true = float(a.sum().item())  # 1.0 + 4.0 = 5.0

    # Mean should be extremely close to the true trace
    assert math.isclose(
        stats["mean"],
        trace_true,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ), f"Expected trace {trace_true}, got {stats['mean']}"

    # For this diagonal quadratic, vᵀHv = trace(H) for ANY Rademacher v,
    # so each probe returns exactly the same value ⇒ std ≈ 0.
    assert stats["std"] < 1e-5, f"Expected almost-zero std, got {stats['std']}"
