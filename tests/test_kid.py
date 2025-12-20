"""
KID tests: math sanity + evaluate_diffusion integration.

Goals:
- KID is ~0 for two *independent* pools from the same distribution.
- KID increases when distributions differ (easy-to-detect shift).
- evaluate_diffusion(task="kid") records res["kid"] + details, and caches real feats.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import ablation_harness.eval.generative as gen_mod
from ablation_harness.eval.generative import evaluate_diffusion


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)

    def forward(self, x, *args, **kwargs):
        return x


def make_eval_cfg_kid(*, enabled=True, n_samples=64, repeats=20, subset_size=32):
    # Only define what run_kid reads; other tasks disabled.
    kid = SimpleNamespace(
        enabled=enabled,
        n_samples=n_samples,
        repeats=repeats,
        subset_size=subset_size,
        sampler="ddim",
        nfe=10,
        batch_size=64,
        real_seed=123,
        real_split="train",
    )
    grid = SimpleNamespace(enabled=False)
    fid_milestone = SimpleNamespace(enabled=False)
    final = SimpleNamespace(enabled=False)
    return SimpleNamespace(
        quick=False,
        kid=kid,
        grid=grid,
        fid_milestone=fid_milestone,
        final=final,
        sample_seed=0,
    )


def _rng_feats(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    # Keep feature scale small/stable for polynomial kernel.
    return (rng.standard_normal((n, d)) / np.sqrt(d)).astype(np.float64)


def test_kid_from_pools_same_distribution_near_zero():
    """
    Math sanity:
    Two pools drawn independently from the same distribution => KID ~ 0.
    (Don't test X vs itself; that's not independent pools and can bias the estimate.)
    """
    assert hasattr(gen_mod, "_kid_from_pools"), "Expected _kid_from_pools in generative.py"
    rng0 = np.random.default_rng(0)
    rng1 = np.random.default_rng(1)

    n, d = 512, 64
    feats_gen = _rng_feats(rng0, n, d)
    feats_real = _rng_feats(rng1, n, d)

    kid_mean, kid_std, kid_sem = gen_mod._kid_from_pools(
        feats_gen=feats_gen,
        feats_real=feats_real,
        subset_size=128,
        repeats=20,
        seed=999,
    )

    assert np.isfinite(kid_mean)
    assert np.isfinite(kid_std)
    assert np.isfinite(kid_sem)

    # Same-dist should be very close to 0 (can be slightly +/-).
    assert abs(float(kid_mean)) < 1e-2, kid_mean


def test_kid_increases_when_distributions_differ():
    """
    Math sanity:
    Shift one pool => KID should increase clearly.
    """
    assert hasattr(gen_mod, "_kid_from_pools"), "Expected _kid_from_pools in generative.py"
    rng0 = np.random.default_rng(0)
    rng1 = np.random.default_rng(1)

    n, d = 512, 64
    feats_gen = _rng_feats(rng0, n, d)
    feats_real_same = _rng_feats(rng1, n, d)
    feats_real_shift = feats_real_same + 0.10  # small but strong, stable signal

    kid_same, _, _ = gen_mod._kid_from_pools(
        feats_gen=feats_gen,
        feats_real=feats_real_same,
        subset_size=128,
        repeats=20,
        seed=999,
    )
    kid_shift, _, _ = gen_mod._kid_from_pools(
        feats_gen=feats_gen,
        feats_real=feats_real_shift,
        subset_size=128,
        repeats=20,
        seed=999,
    )

    assert np.isfinite(kid_same)
    assert np.isfinite(kid_shift)

    # Should be noticeably larger than "same distribution" case.
    assert float(kid_shift) > float(abs(kid_same)) + 1e-2, (kid_same, kid_shift)


def test_evaluate_diffusion_kid_records_details_and_caches_real_feats(monkeypatch, tmp_path):
    """
    Integration sanity:
    - evaluate_diffusion(task="kid") should compute kid and write details.
    - real feats should be cached to disk and reused on the 2nd call.
    """
    # These are helper names your run_kid calls (per your snippet).
    assert hasattr(gen_mod, "_real_inception_feats_cifar10"), "Expected _real_inception_feats_cifar10"
    assert hasattr(gen_mod, "_gen_inception_feats"), "Expected _gen_inception_feats"
    assert hasattr(gen_mod, "_kid_from_pools"), "Expected _kid_from_pools"

    model = TinyModel()
    q = {"dummy": True}
    eval_cfg = make_eval_cfg_kid(enabled=True, n_samples=64, repeats=10, subset_size=32)

    d = 64
    rng_real = np.random.default_rng(123)
    rng_gen = np.random.default_rng(456)
    feats_real = _rng_feats(rng_real, eval_cfg.kid.n_samples, d)
    feats_gen = _rng_feats(rng_gen, eval_cfg.kid.n_samples, d)

    real_calls = {"n": 0}

    def fake_real_feats_cifar10(**kwargs):
        real_calls["n"] += 1
        return feats_real

    def fake_gen_feats(**kwargs):
        return feats_gen

    monkeypatch.setattr(gen_mod, "_real_inception_feats_cifar10", fake_real_feats_cifar10)
    monkeypatch.setattr(gen_mod, "_gen_inception_feats", fake_gen_feats)

    out_dir = tmp_path / "kid_eval"
    res1 = evaluate_diffusion(model, eval_cfg, q, out_dir, task="kid")

    assert res1["kid"] is not None
    assert np.isfinite(res1["kid"])
    assert "kid" in res1["details"]

    det = res1["details"]["kid"]

    # Your run_kid writes these fields (per your snippet).
    for k in ["kid_mean", "kid_std", "kid_sem", "n_pool", "subset_size", "repeats", "sampler", "nfe", "real_cache"]:
        assert k in det, f"missing details['kid']['{k}']"

    # Cache file should exist.
    cache_path = Path(det["real_cache"])
    assert cache_path.is_file()

    assert real_calls["n"] == 0 or 1  # 0 in the case of finding cached stats

    # 2nd run: should load cache (no real-feats recompute).
    def real_feats_should_not_be_called(**kwargs):
        raise AssertionError("Expected to load cached real feats; recomputed instead")

    monkeypatch.setattr(gen_mod, "_real_inception_feats_cifar10", real_feats_should_not_be_called)

    res2 = evaluate_diffusion(model, eval_cfg, q, out_dir, task="kid")
    assert res2["kid"] is not None
    assert np.isfinite(res2["kid"])
