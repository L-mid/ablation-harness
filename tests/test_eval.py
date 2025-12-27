from pathlib import Path
from types import SimpleNamespace

import torch

import ablation_harness.eval.generative as gen_mod

# Adjust to your real package layout
from ablation_harness.eval.generative import _sample, evaluate_diffusion


def precompute_q(betas: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    betas: [K] tensor on the right device.
    Returns all the q-schedule tensors needed for both training + sampling.
    """
    # forward process
    alphas = 1.0 - betas  # [K]
    alpha_bar = torch.cumprod(alphas, dim=0)  # [K]
    alpha_bar_prev = torch.cat(
        [torch.ones(1, device=betas.device, dtype=betas.dtype), alpha_bar[:-1]],
        dim=0,
    )

    # posterior q(x_{t-1} | x_t, x_0)
    posterior_var = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
    posterior_log_var_clipped = torch.log(torch.clamp(posterior_var, min=1e-20))

    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bar": alpha_bar,
        "alpha_bar_prev": alpha_bar_prev,
        "sqrt_alpha_bar": torch.sqrt(alpha_bar),
        "sqrt_one_minus_alpha_bar": torch.sqrt(1.0 - alpha_bar),
        "posterior_var": posterior_var,
        "posterior_log_var_clipped": posterior_log_var_clipped,
    }


betas = torch.linspace(1e-1, 1e-4, 5)
pre_q = precompute_q(betas)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)

    def forward(self, x, *args, **kwargs):
        return x


def make_eval_cfg(
    *,
    quick=False,
    grid_enabled=False,
    kid_enabled=False,
    fid_enabled=False,
    final_enabled=False,
):
    grid = SimpleNamespace(
        enabled=grid_enabled,
        n_samples=16,
        batch_size=8,
        sample_seed=0,
        sampler="ddim",
        nfe=10,
        save_images=True,
    )
    kid = SimpleNamespace(
        enabled=kid_enabled,
        n_samples=32,
        repeats=3,
        nfe=10,
    )
    fid_milestone = SimpleNamespace(
        enabled=fid_enabled,
        n_samples=64,
        nfe=5,
        fid_stats="stats/cifar10_inception_train.npz",
        run_if_kid_improved_pct=0.0,
    )
    final = SimpleNamespace(
        enabled=final_enabled,
        n_samples=2,
        sampler="ddpm",
        nfe=5,
        fid_stats="stats/cifar10_inception_train.npz",
    )
    """Makes cfg for eval."""
    return SimpleNamespace(
        quick=quick,
        grid=grid,
        kid=kid,
        fid_milestone=fid_milestone,
        final=final,
        sample_seed=0,
    )


def test__sample_uses_ddim_and_ddpm(monkeypatch):
    """Tests sample can diff between DDIM and DDPM samplers."""
    calls = []

    class DummyDDIM:
        def __init__(self, q, nfe, eta, device):
            calls.append(("ddim", q, nfe, eta, str(device)))

        def sample(self, model, shape, seed):
            return torch.zeros(shape)

    class DummyDDPM:
        def __init__(self, q, nfe, device):
            calls.append(("ddpm", q, nfe, None, str(device)))

        def sample(self, model, shape, seed):
            return torch.ones(shape)

    monkeypatch.setattr(gen_mod, "DDIMSampler", DummyDDIM)
    monkeypatch.setattr(gen_mod, "DDPMSampler", DummyDDPM)

    model = TinyModel()
    device = next(model.parameters()).device
    q = {"dummy": True}

    imgs_ddim = _sample(model, (2, 3, 32, 32), q, "ddim", 7, seed=123, device=device)
    imgs_ddpm = _sample(model, (2, 3, 32, 32), q, "ddpm", 9, seed=456, device=device)

    # Check call trace
    assert calls[0][:2] == ("ddim", q)
    assert calls[0][2] == 7  # nfe
    assert calls[1][:2] == ("ddpm", q)
    assert calls[1][2] == 9  # nfe

    # Check return shapes
    assert imgs_ddim.shape == (2, 3, 32, 32)
    assert imgs_ddpm.shape == (2, 3, 32, 32)


def test_grid_saves_image_and_respects_n(monkeypatch, tmp_path):
    """Tests grid configs save images and respect n."""
    model = TinyModel()
    q = {"dummy": True}
    eval_cfg = make_eval_cfg(grid_enabled=True)

    samples = []

    def fake_sample(model_ema, shape, q_, sampler, nfe_, seed, device):
        samples.append((shape, sampler, nfe_))
        b, c, h, w = shape
        return torch.zeros(b, c, h, w)

    monkeypatch.setattr(gen_mod, "_sample", fake_sample)

    out_dir = tmp_path / "grid_eval"
    res = evaluate_diffusion(model, eval_cfg, q, out_dir, task="grid")

    # Should have called _sample enough times to cover n_samples=16
    total_b = sum(s[0][0] for s in samples)
    assert total_b == eval_cfg.grid.n_samples

    # Sampler / NFE respected
    for shape, sampler, nfe in samples:
        assert sampler == "ddim"
        assert nfe == eval_cfg.grid.nfe

    # Details + output image
    detail = res["details"]["grid"]
    assert detail["n"] == eval_cfg.grid.n_samples
    assert Path(detail["path"]).is_file()
    assert res["n"] == eval_cfg.grid.n_samples


def test_quick_mode_clamps_grid_and_kid(monkeypatch, tmp_path):
    """Tests that quick mode affects both grid and kid."""
    model = TinyModel()
    q = {"dummy": True}
    eval_cfg = make_eval_cfg(
        quick=True,
        grid_enabled=True,
        kid_enabled=True,
    )

    # make grid.n_samples & kid.n_samples "large" so we see clamp
    eval_cfg.grid.n_samples = 16
    eval_cfg.grid.nfe = 10
    eval_cfg.kid.n_samples = 32
    eval_cfg.kid.nfe = 10

    def fake_sample(model_ema, shape, q, sampler, nfe, seed, device):
        return torch.zeros(shape)

    monkeypatch.setattr(gen_mod, "_sample", fake_sample)

    res = evaluate_diffusion(model, eval_cfg, q, tmp_path / "quick", task=None)

    # After quick-clamp, grid and kid configs should be reduced
    assert eval_cfg.grid.nfe <= 5
    assert eval_cfg.grid.n_samples <= 2
    assert eval_cfg.kid.nfe <= 5
    assert eval_cfg.kid.n_samples <= 2

    # The details should reflect the clamped values
    assert res["details"]["grid"]["n"] == eval_cfg.grid.n_samples
    assert res["details"]["kid"]["n_pool"] == eval_cfg.kid.n_samples


def test_fid_milestone_runs_when_gate_zero(tmp_path):
    """When gate on kid improvment is 0, fid runs"""
    model = TinyModel()
    q = pre_q
    eval_cfg = make_eval_cfg(fid_enabled=True)
    eval_cfg.fid_milestone.run_if_kid_improved_pct = 0.0  # gate off

    res = evaluate_diffusion(model, eval_cfg, q, tmp_path / "fid_zero", task="fid_milestone")

    detail = res["details"]["fid_milestone"]
    # In the stub, fid is None but the entry exists and is not "skipped"
    assert "skipped" not in detail
    assert detail["n"] == eval_cfg.fid_milestone.n_samples
    assert detail["fid_stats"] == eval_cfg.fid_milestone.fid_stats


def test_fid_milestone_skips_when_gate_positive_and_no_kid(tmp_path):
    """Fid milestone actually skips when gate is active."""
    model = TinyModel()
    q = {"dummy": True}
    eval_cfg = make_eval_cfg(fid_enabled=True)
    eval_cfg.fid_milestone.run_if_kid_improved_pct = 5.0  # positive gate

    # We call only 'fid_milestone' task; kid_now remains None
    res = evaluate_diffusion(model, eval_cfg, q, tmp_path / "fid_gated", task="fid_milestone")

    detail = res["details"]["fid_milestone"]
    assert detail["skipped"] is True
    assert detail["reason"] == "kid_missing"


def test_final_records_sampler_nfe_and_n(tmp_path):
    """Tests recording in final dict happens."""
    model = TinyModel()
    q = pre_q
    eval_cfg = make_eval_cfg(final_enabled=True)
    eval_cfg.final.sampler = "ddpm"
    eval_cfg.final.nfe = 5
    eval_cfg.final.n_samples = 123

    res = evaluate_diffusion(model, eval_cfg, q, tmp_path / "final_only", task="final")

    assert "final" in res["details"]
    detail = res["details"]["final"]
    assert detail["n"] == 123
    assert detail["fid_stats"] == eval_cfg.final.fid_stats
    assert detail["sampler"] == "ddpm"
    assert detail["nfe"] == 5
