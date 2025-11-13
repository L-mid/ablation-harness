import torch

from ablation_harness.tasks.diffusion.models.unet_cifar32 import UNetCifar32
from ablation_harness.tasks.diffusion.schedule import (
    ddpm_loss,
    get_beta_schedule,
    precompute_q,
    sample_ddpm,
)


def test_ddpm_loss_forward():
    torch.manual_seed(0)
    model = UNetCifar32(base_channels=32)  # small & fast
    model.eval()
    K = 1000
    betas = get_beta_schedule("linear", K, device="cpu")
    q = precompute_q(betas)
    x0 = torch.randn(4, 3, 32, 32)  # pretend images in [-1,1]
    loss = ddpm_loss(model, x0, q)
    assert torch.isfinite(loss), "Loss should be finite"


def test_sampling_shape_and_range():
    torch.manual_seed(0)
    model = UNetCifar32(base_channels=32)
    K = 1000
    q = precompute_q(get_beta_schedule("linear", K, device="cpu"))
    imgs = sample_ddpm(model, (8, 3, 32, 32), q, nfe=5, seed=123, device="cpu")
    assert imgs.shape == (8, 3, 32, 32)
    assert imgs.min() >= -1.001 and imgs.max() <= 1.001
