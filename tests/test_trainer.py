import time

import torch

from ablation_harness.trainer import TinyCNN


def test_params_under_10k():
    m = TinyCNN(num_classes=10, dropout=0.0)
    assert sum(p.numel() for p in m.parameters()) < 10_000


def test_forward_shape():
    m = TinyCNN(num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    y = m(x)
    assert y.shape == (4, 10)


def test_forward_speed_under_1s():
    m = TinyCNN(num_classes=10)
    x = torch.randn(64, 3, 32, 32)
    t0 = time.time()
    _ = m(x)
    assert time.time() - t0 < 1.0  # generous on CPU
