import os
import pathlib
from collections.abc import Mapping

import pytest
import yaml

os.environ.setdefault("MPLBACKEND", "Agg")  # must be set before importing pyplot

import matplotlib  # noqa: E402

matplotlib.use("Agg", force=True)  # guarantee no GUI backend is used


def _deep_update(base: dict, updates: dict) -> dict:
    for k, v in updates.items():
        if isinstance(v, Mapping) and isinstance(base.get(k), Mapping):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


@pytest.fixture
def make_study_yaml(tmp_path: pathlib.Path):
    """
    Factory fixture that writes a minimal study/v1 YAML and returns its Path.
    - Usage:
        - study_yaml = make_study_yaml()                      # defaults
        - study_yaml = make_study_yaml({"epochs": 5})         # override top-level
        - study_yaml = make_study_yaml({"baseline": {"lr": 3e-4}})


    Returns YAML explicitly (does not return a cfg).
    """

    def _make(overrides: dict | None = None, filename: str = "study.yaml"):
        cfg = {
            "schema": "study/v1",
            "study_name": "test_cli_determinism",
            "metric": "val/acc",
            "goal": "max",
            # Small defaults to keep tests fast; your trainer can still
            # fill missing fields if that's your contract.
            "baseline": {
                "dataset": "cifar10",
                "model": "tinycnn",
                "subset": 64,
                "epochs": 1,
                "batch_size": 64,
                "seed": 42,
                "optimizer": "adam",
                "lr": 1e-3,
                "wd": 0.0,
                "dropout": 0.0,
                "ema": False,
                "run_id": "study_yaml_test",
                "out_dir": str(tmp_path / "runs"),  # stringify for YAML/JSON
            },
            # parser-friendly: empty list instead of None
            "variants": [],
        }

        # commits overides
        if overrides:
            _deep_update(cfg, overrides)

        # writer to yaml (yaml.safe_dump). No Omegaconf.
        path = tmp_path / filename
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        return path

    return _make


@pytest.fixture()
def dummy_sweep_schema_yaml():
    """WIP, for potental use."""
    cfg = {
        "base": {
            "model": "mlp",
            "epochs": 1,
            "batch_size": 64,
            "dataset": "moons",
            "seed": 0,
        },
        "grid": {
            "hidden": [16, 64],
            "lr": [0.01, 0.001],
        },
        "metric": "val/acc",
        "goal": "max",
    }
