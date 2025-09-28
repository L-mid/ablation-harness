import numpy as np

from ablation_harness.seed_utils import set_seed


def test_set_seed_repro():
    set_seed(123)
    a = np.random.randn(5)
    set_seed(123)
    b = np.random.randn(5)
    assert (a == b).all()
