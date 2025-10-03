import numpy as np

from ablation_harness.seed_utils import seed_everything


def test_set_seed_repro():
    seed_everything(123)
    a = np.random.randn(5)
    seed_everything(123)
    b = np.random.randn(5)
    assert (a == b).all()
