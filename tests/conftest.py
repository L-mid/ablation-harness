import json
import os
import pathlib
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

os.environ.setdefault("MPLBACKEND", "Agg")  # must be set before importing pyplot

import matplotlib  # noqa: E402

matplotlib.use("Agg", force=True)  # guarantee no GUI backend is used


@pytest.fixture
def make_study_yaml(tmp_path: pathlib.Path):
    """
    Factory fixture that writes a study/v1 YAML and returns its Path.

    Usage:
        study_yaml = make_study_yaml()                               # defaults
        study_yaml = make_study_yaml({"epochs": 5})                   # override top-level
        study_yaml = make_study_yaml({"optim": {"lr": 3e-4}})         # override nested
        study_yaml = make_study_yaml({"model": {"dropout": 0.3}})

    Returns YAML path (does not return a loaded cfg).
    """

    def _deep_update(dst: dict, src: dict) -> dict:
        """Simple recursive merge: dicts merge, everything else overwrites."""
        for k, v in (src or {}).items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                _deep_update(dst[k], v)
            else:
                dst[k] = v
        return dst

    def _make(overrides: dict | None = None, filename: str = "study.yaml"):
        cfg = {
            "schema": "study/v1",
            "study_name": "test_cli_determinism",
            "out_dir": str(tmp_path / "runs"),
            "seed": 42,
            "metric": "val/acc",
            "goal": "max",
            # Small defaults so tests stay fast; your trainer can still
            # fill in anything missing if that's your contract.
            "epochs": 1,
            "device": None,  # auto
            "deterministic": True,
            "clean_run": False,
            "data": {
                "dataset": "cifar10",
                "subset": 64,
                "batch_size": 64,
                "num_workers": 0,
                "pin_memory": False,
            },
            "model": {
                "name": "tinycnn",
                "hidden": 64,
                "dropout": 0.0,
            },
            "optim": {
                "optimizer": "adam",
                "lr": 1e-3,
                "wd": 0.0,
                "momentum": 0.9,
            },
            "sched": {"name": "cosine"},
            "ema": {"enabled": False, "decay": 0.9999},
            # Parser-friendly: empty list instead of None
            "variants": [],
            # Third-party logging config (ignore if your runtime treats as no-op)
            "logging": {
                "enable": True,
                "backends": ["tensorboard"],  # or ["tensorboard","wandb"]
                "log_every_n_steps": 1,
                "wandb": {
                    "mode": "offline",
                    "project": "ablation-harness",
                    "run_name": None,
                    "tags": ["test"],
                    "notes": "fixture default",
                },
                "tensorboard": {"flush_secs": 10},
            },
        }

        if overrides:
            _deep_update(cfg, overrides)

        path = tmp_path / filename
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        return path

    return _make


@pytest.fixture
def write_jsonl_rows():
    """Factory: write study/v1-shaped JSONL rows to `p` and return `p`. Study/v1 output mimicing."""

    def _write(p: Path, *, rows: int = 5) -> Path:
        p.parent.mkdir(parents=True, exist_ok=True)

        base_cfg = {
            "schema": "study/v1",
            "study_name": "tinycnn_cifar10_wk2",
            "metric": "val/acc",
            "goal": "max",
            "out_dir": "runs/logs",
            "seed": 0,
            "epochs": 1,
            "data": {"dataset": "cifar10", "subset": 256, "batch_size": 128, "num_workers": 0, "pin_memory": False},
            "model": {"name": "tinycnn", "dropout": 0.0},
            "optim": {"optimizer": "adam", "lr": 3e-4, "wd": 0.0, "momentum": 0.0},
            "sched": {"name": "none"},
            "ema": {"enabled": False, "decay": 0.9999},
            "variants": [],
        }

        grid = [
            {"optim": {"optimizer": "adam", "lr": 3e-4, "wd": 0.0}, "ema": {"enabled": False}},
            {"optim": {"optimizer": "adam", "lr": 1e-3, "wd": 0.0}, "ema": {"enabled": False}},
            {"optim": {"optimizer": "sgd", "lr": 1e-2, "wd": 0.0, "momentum": 0.9}, "ema": {"enabled": False}},
            {"optim": {"optimizer": "sgd", "lr": 1e-2, "wd": 0.0, "momentum": 0.9}, "ema": {"enabled": True}},
            {"optim": {"optimizer": "adamw", "lr": 3e-4, "wd": 0.01}, "ema": {"enabled": True}},
        ][:rows]

        rows_out = []
        for i, ov in enumerate(grid):
            cfg = deepcopy(base_cfg)
            for k, v in ov.items():
                cfg[k].update(v) if isinstance(v, dict) else cfg.__setitem__(k, v)

            run_id = f"generic_any_id_{i}"
            out_dir = cfg["out_dir"]
            out = {
                "seed": cfg["seed"],
                "val/acc": 0.10 + 0.01 * i,
                "val/loss": 2.3048 - 0.01 * i,
                "params": 7738,
                "dataset": cfg["data"]["dataset"],
                "model_used": "TinyCNN",
                "run_id": run_id,
                "run_dir": f"{out_dir}/{run_id}",
                "ckpt": f"{out_dir}/{run_id}/ckpts/ckpt.pt",
                "spect_stats": None,
                "loss_log": f"{out_dir}/{run_id}/loss.jsonl",
                "_elapsed_sec": 5.5 + i,
            }
            rows_out.append({"cfg": cfg, "out": out, "_i": i})

        with p.open("w", encoding="utf-8") as f:
            for r in rows_out:
                f.write(json.dumps(r) + "\n")

        return p

    return _write
