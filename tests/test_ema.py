import copy
import math

import pytest
import torch
from torch import nn

from ablation_harness.ema import EMA, EMAConfig


class Toy(nn.Module):
    """Toy 'model'. (a linear layer)"""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3, 1, bias=False)
        # one buffer to test buffer behavior
        self.register_buffer("scale", torch.tensor(1.0))

    def forward(self, x):
        return self.lin(x) * self.scale


def _fixed_batch(device):
    """Gives tensors: x (data), y (labels)."""
    x = torch.ones(4, 3, device=device)
    y = torch.zeros(4, 1, device=device)
    return x, y


@pytest.mark.parametrize("decay", [0.9, 0.999])
def test_update_math_one_step(decay):
    """Ensures the core of EMA: it's update math is valid."""
    dev = torch.device("cpu")
    m = Toy().to(dev)
    # make weights known
    with torch.no_grad():
        m.lin.weight.fill_(1.0)  # theta0 = 1
    ema = EMA(m, EMAConfig(decay=decay))

    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    x, y = _fixed_batch(dev)

    # One step to change weights from 1.0 → something new (theta1)
    opt.zero_grad(set_to_none=True)
    loss = (m(x) - y).pow(2).mean()
    loss.backward()
    opt.step()

    theta1 = m.lin.weight.detach().clone()  # shape [1,3]
    theta1_00 = theta1[0, 0].item()  # scalar
    print(theta1)
    # Perform EMA update once
    ema.update(m)

    # Expected ema = decay*theta0 + (1-decay)*theta1; theta0 was 1.0
    expected = decay * 1.0 + (1.0 - decay) * theta1_00
    got = ema._shadow[0][0, 0].item()
    assert math.isfinite(got)
    assert abs(got - expected) < 1e-6


def test_warmup_copies_not_averaging():
    """Ensure the warmup copies are indeed NOT doing anything."""
    dev = torch.device("cpu")
    m = Toy().to(dev)
    with torch.no_grad():
        m.lin.weight.fill_(2.0)  # fill_
    ema = EMA(m, EMAConfig(decay=0.9, warmup_steps=2))  # 2 warmup steps, hold off averaging for that long.

    # step 1 (warmup): shadow becomes exact copy (2.0)
    ema.update(m)
    assert ema._shadow[0][0, 0].item() == pytest.approx(2.0)

    # change live weights
    with torch.no_grad():
        m.lin.weight.fill_(5.0)
    # step 2 (still warmup): shadow again equals live (5.0), no averaging
    ema.update(m)
    assert ema._shadow[0][0, 0].item() == pytest.approx(5.0)


def test_apply_to_swaps_and_restores():
    """EMA does not: override model, do nothing, and works in/with an eval (torch.no_grad)."""
    dev = torch.device("cpu")
    m = Toy().to(dev)
    with torch.no_grad():
        m.lin.weight.copy_(torch.tensor([[3.0]]))
    ema = EMA(m, EMAConfig(decay=0.9))
    # Manually set EMA shadow to a different value
    with torch.no_grad():
        ema._shadow[0].fill_(7.0)

    live_before = m.lin.weight.detach().clone()
    with ema.apply_to(m):
        # inside: model params must equal EMA shadow (7.0)
        assert m.lin.weight.detach()[0, 0].item() == pytest.approx(7.0)
    # after: restored to live_before (3.0)
    assert m.lin.weight.detach()[0, 0].item() == pytest.approx(live_before[0, 0].item())


def test_state_dict_roundtrip(tmp_path):
    """Tests checkpointing with it's custom dicts actually works, full roundtrip."""
    dev = torch.device("cpu")
    m = Toy().to(dev)
    ema1 = EMA(m, EMAConfig(decay=0.99))
    # mutate EMA a bit
    ema1.update(m)
    sd = copy.deepcopy(ema1.state_dict())

    # new instance, load
    ema2 = EMA(m, EMAConfig(decay=0.5))  # different decay; should be overwritten by load
    ema2.load_state_dict(sd)

    assert ema2.decay == pytest.approx(sd["decay"])
    assert ema2._step == sd["step"]
    for t1, t2 in zip(sd["shadow"], ema2._shadow):
        assert torch.allclose(t1, t2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cpu_shadow_cuda_model():
    """No device troubles, shadows on cpu by default."""
    device = torch.device("cuda")
    m = Toy().to(device)
    ema = EMA(m, EMAConfig(decay=0.999, device=torch.device("cpu")))  # keep shadow on CPU

    # Run a couple updates; should not raise
    for _ in range(3):
        ema.update(m)

    # Shadow tensors should be on CPU
    for t in ema._shadow:
        assert t.device.type == "cpu"


def test_no_grads_and_skips_frozen_params():
    """Frozen params are skipped, no grads."""
    dev = torch.device("cpu")
    m = Toy().to(dev)
    # Freeze param
    for p in m.lin.parameters():
        p.requires_grad = False

    ema = EMA(m, EMAConfig(decay=0.9))
    # If param is frozen, shadow list should be empty (or unchanged)
    assert len(ema._shadow) == 0

    # Ensure no grad history on any EMA tensors
    for t in getattr(ema, "_shadow", []):
        assert not t.requires_grad  # make sure it does not need params.


def test_buffers_not_included_by_default():
    """Buffers off with include_buffers=False."""
    dev = torch.device("cpu")
    m = Toy().to(dev)
    with torch.no_grad():
        m.scale.fill_(5.0)
    ema = EMA(m, EMAConfig(decay=0.9, include_buffers=False))
    # No buffer shadows tracked by default
    assert getattr(ema, "_buf_shadow", []) == []


def test_convergence_when_live_constant():
    """EMA should converge noise -> theta."""
    dev = torch.device("cpu")
    m = Toy().to(dev)
    with torch.no_grad():
        m.lin.weight.fill_(4.0)
    ema = EMA(m, EMAConfig(decay=0.9))
    for _ in range(50):
        ema.update(m)
    # After many updates with constant theta, EMA ≈ theta
    assert ema._shadow[0][0, 0].item() == pytest.approx(4.0, rel=0, abs=1e-5)
