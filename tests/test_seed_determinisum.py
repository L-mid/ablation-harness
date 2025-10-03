import random

import numpy as np
import pytest
import torch

from ablation_harness.seed_utils import seed_everything
from ablation_harness.trainer import TinyCNN


@pytest.mark.skipif(torch.cuda.is_available(), reason="CPU determinism only")
def test_cpu_determinism_forward():
    seed_everything(1337, single_thread=True)
    x = torch.randn(4, 3, 32, 32)
    m1 = TinyCNN()
    y1 = m1(x)

    seed_everything(1337)
    x2 = torch.randn(4, 3, 32, 32)
    m2 = TinyCNN()
    y2 = m2(x2)

    assert torch.allclose(y1, y2, atol=0.0, rtol=0.0)


def test_random_streams_match():
    seed_everything(42)
    a = (random.random(), np.random.rand(), torch.rand(1).item())
    seed_everything(42)
    b = (random.random(), np.random.rand(), torch.rand(1).item())
    assert a == b
