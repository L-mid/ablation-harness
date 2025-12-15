import torch

from ablation_harness.tasks.diffusion.schedule import get_beta_schedule, precompute_q


def test_q_identity_reconstructs_x0_float32():
    """
    Reconstructs x0 perfectly (should).

    catches:
        wrong alpha_bar indexing (off-by-one),
        wrong shapes/broadcasting,
        wrong schedule returned (e.g. K mismatch),
        any accidental dtype/device mismatch.
    """

    torch.manual_seed(0)
    B, C, H, W = 8, 3, 32, 32
    K = 1000

    x0 = torch.rand(B, C, H, W) * 2 - 1  # [-1,1]
    eps = torch.randn_like(x0)

    betas = get_beta_schedule("cosine", K, device=torch.device("cpu"))
    q = precompute_q(betas)
    alpha_bar = q["alpha_bar"]

    t = torch.randint(0, K, (B,), dtype=torch.long)
    ab = alpha_bar[t].view(B, 1, 1, 1)

    sqrt_ab = ab.sqrt()
    sqrt_1mab = (1 - ab).clamp(min=0).sqrt()

    xt = sqrt_ab * x0 + sqrt_1mab * eps
    x0_hat_true = (xt - sqrt_1mab * eps) / sqrt_ab.clamp(min=1e-8)

    max_abs = (x0_hat_true - x0).abs().max().item()
    assert max_abs < 5e-5
