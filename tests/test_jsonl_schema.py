"""
Uses ablate.py to generate REAL fields. (integration test, needs maintenence)

Below is additonally a local field checker (imports from nowhere) (no integration, fast).
"""

import json
import pathlib
import subprocess
import sys
from typing import Any, Dict

# ---------- Schema we expect in each line ----------
TOP_REQUIRED = {"cfg", "out", "_i"}

# Top-level cfg keys that should always be present after resolution
CFG_REQUIRED_TOP = {
    # "schema",
    "study_name",
    "metric",
    "goal",
    "out_dir",
    "seed",
    "epochs",
}

# Nested blocks required by study/v1
CFG_REQUIRED_NESTED = {
    "data": {"dataset", "subset", "batch_size"},  # keep minimal/portable
    "model": {"name"},  # model fields vary by impl
    "optim": {"optimizer", "lr", "wd"},
    "ema": {"enabled", "decay"},
    "sched": {"name"},  # allow simple sched name
    # "logging" is optional for schema purposes (impl-specific)
}

# 'out' fields that should be present and stable enough to assert
OUT_REQUIRED = {
    "seed",
    "val/acc",
    "val/loss",
    "params",
    "dataset",
    "run_id",
    # model field may appear as 'model_used' or 'model' depending on writer
}

# Fields that are expected to vary even when deterministic (paths/timing)
VOLATILE_OUT = {"run_dir", "ckpt_last", "ckpt_best", "loss_log", "_elapsed_sec", "run_time_s"}

VOLATILE_CFG = {"out_dir"}  # cfg on the top level.


def make_dummy_result_record() -> Dict[str, Any]:
    """Mockup jsonl contents approximation for test_jsonl_schema_shape_only."""
    return {
        "cfg": {
            "schema": "study/v1",
            "study_name": "dummy_study",
            "metric": "val/acc",
            "goal": "max",
            "out_dir": "runs/dummy",
            "seed": 42,
            "epochs": 1,
            # Optional convenience flags
            "device": None,
            "deterministic": True,
            "clean_run": False,
            "data": {
                "dataset": "cifar10",
                "subset": 64,
                "batch_size": 64,
                # extra (impl-dependent) keys OK:
                "num_workers": 0,
                "pin_memory": False,
            },
            "model": {
                "name": "tinycnn",
                "hidden": 64,
                "dropout": 0.3,
            },
            "optim": {
                "optimizer": "adam",
                "lr": 1e-3,
                "wd": 0.0,
                "momentum": 0.9,
            },
            "sched": {"name": "cosine"},
            "ema": {"enabled": False, "decay": 0.9999},
            # logging block optional for schema; included here for realism
            "logging": {
                "enable": True,
                "backends": ["tensorboard"],
                "log_every_n_steps": 1,
                "wandb": {"mode": "offline", "project": "ablation-harness"},
            },
        },
        "out": {
            "seed": 42,
            "val/acc": 0.123,  # float
            "val/loss": 2.345,  # float
            "params": 7738,  # int
            "dataset": "cifar10",
            "model_used": "TinyCNN",  # alias accepted; may be 'model' instead
            "run_id": "dummy_run",
            "run_dir": "runs/dummy/dummy_run",
            "ckpt": "runs/dummy/dummy_run/ckpts/ckpt.pt",
            "spect_stats": None,  # allowed to be null
            "loss_log": "runs/dummy/dummy_run/loss.jsonl",
            "_elapsed_sec": 0.001,
        },
        "_i": 0,
    }


# Small numeric tolerance for floats (even on CPU tiny diffs can appear with
# different BLAS or Python minor versions)
EPS = 1e-8


def _read_jsonl(path: pathlib.Path):
    """A very compact jsonl reader."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _assert_keys_present(d: Dict[str, Any], required: set, where: str):
    """Asserts missing keys if not present. Prints 'where' (the cfg provided) and 'missing'."""
    missing = required - set(d.keys())
    assert not missing, f"{where} keys missing: {missing}"


def _validate_schema(rec: Dict[str, Any]):
    """
    Validates resulting jsonl contents against the updated study/v1 schema core.
    Strict on required core keys, permissive on implementation extras.
    """

    # top level keys
    assert set(rec.keys()) == TOP_REQUIRED, f"Top-level keys mismatch: {set(rec.keys())}"
    cfg = rec["cfg"]
    out = rec["out"]

    # ---- cfg checks
    _assert_keys_present(cfg, CFG_REQUIRED_TOP, "cfg (top)")
    for block, req in CFG_REQUIRED_NESTED.items():
        assert block in cfg, f"cfg missing block '{block}'"
        assert isinstance(cfg[block], dict), f"cfg.{block} must be a dict"
        _assert_keys_present(cfg[block], req, f"cfg.{block}")

    # spot-check types on some critical cfg fields
    assert isinstance(cfg["seed"], int)
    assert isinstance(cfg["epochs"], int)
    assert isinstance(cfg["study_name"], str)
    assert isinstance(cfg["metric"], str)
    assert cfg["goal"] in {"max", "min"}

    # nested type spot-checks (lightweight)
    assert isinstance(cfg["data"]["dataset"], str)
    assert isinstance(cfg["data"]["subset"], int)
    assert isinstance(cfg["data"]["batch_size"], int)
    assert isinstance(cfg["optim"]["lr"], (int, float))
    assert isinstance(cfg["ema"]["enabled"], bool)
    assert isinstance(cfg["ema"]["decay"], (int, float))
    assert isinstance(cfg["model"]["name"], str)

    # ---- out checks
    _assert_keys_present(out, OUT_REQUIRED, "out")

    # spot-check out types
    assert isinstance(out["seed"], int)
    assert isinstance(out["params"], int)
    assert isinstance(out["val/acc"], (int, float))
    assert isinstance(out["val/loss"], (int, float))


def _normalize_paths(s: str) -> str:
    """Replaces '\\' with '/'. (Windows path resolution.)"""
    # Make Windows/Unix paths comparable
    return s.replace("\\", "/")


def _normalize_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Drop volatile fields and normalize separators,
    leaving only stable content.
    This shouldn't affect new/unschemaed fields (_valdiate_schema doesn't know = unknown).
    """
    rec = json.loads(json.dumps(rec))  # deep copy

    # OUT: drop volatile, normalize slashes
    out = rec["out"]
    for k in list(out.keys()):
        if k in VOLATILE_OUT:
            out.pop(k, None)
        elif isinstance(out[k], str):
            out[k] = _normalize_paths(out[k])

    # CFG: drop volatile, normalize slashes (for any remaining path-like fields)
    cfg = rec["cfg"]
    for k in list(cfg.keys()):
        if k in VOLATILE_CFG:
            cfg.pop(k, None)
        elif isinstance(cfg[k], str):
            cfg[k] = _normalize_paths(cfg[k])

    return rec


def _metrics_close(a: Dict[str, Any], b: Dict[str, Any], eps: float = EPS):
    """
    Cross compares the outputs of 2 runs on identical seed.
    1) compares val/acc to val/acc.
    2) compares val/loss to val/loss.
    3) compares ALL params. (nested in an out.params)

    """
    # Compare selected numeric outputs exactly/with tolerance
    assert abs(a["out"]["val/acc"] - b["out"]["val/acc"]) <= eps, f"val/acc differ: {a['out']['val/acc']} vs {b['out']['val/acc']}"
    assert abs(a["out"]["val/loss"] - b["out"]["val/loss"]) <= eps, f"val/loss differ: {a['out']['val/loss']} vs {b['out']['val/loss']}"
    assert a["out"]["params"] == b["out"]["params"], "param counts differ"


# @pytest.mark.timeout(120) not yet implemented
def test_cli_determinism_same_seed_same_record(tmp_path: pathlib.Path, make_study_yaml, monkeypatch):
    """
    Run the CLI twice with the same seed/config and assert the normalized results.jsonl are identical.
    Kept tiny to be CI-friendly.

    This test will probably fail if you've done anything to cli.py/planner.py OR train.py recently,
    see: _validate_schema & _metrics_close .
    """

    # Force CPU to avoid CUDA nondeterminism (and speed up CI)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    # 1) Write a tiny study YAML (single run) in a temp dir
    study_yaml = make_study_yaml()  # may pass overrides here if needed.
    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"

    # 2) Run CLI twice to two different out dirs
    def _run(out_dir: pathlib.Path):
        cmd = [
            sys.executable,
            "-m",  # add python?
            "ablation_harness.cli",
            "run",
            "--config",
            str(study_yaml),
            "--out_dir",
            str(out_dir),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr)
        assert proc.returncode == 0, "CLI run failed"

        # ensure results.josnl actually exists after a run.
        results = out_dir / "results.jsonl"
        assert results.exists(), "results.jsonl not produced"
        lines = _read_jsonl(results)
        assert len(lines) >= 1, "results.jsonl must have at least one record"
        return lines

    # Using run entrypoint:
    lines_a = _run(out_a)
    lines_b = _run(out_b)

    # For this tiny study we expect exactly one record, but handle N>=1 safely
    assert len(lines_a) == len(lines_b), "Different number of result lines"

    # 3) Validate schema and determinism line-by-line (uses zip for tuples)
    for rec_a, rec_b in zip(lines_a, lines_b):
        _validate_schema(rec_a)  # schema validation done BEFORE volitile norm removal.
        _validate_schema(rec_b)

        na = _normalize_record(rec_a)
        nb = _normalize_record(rec_b)

        # Configs should match exactly (after path normalization)
        assert na["cfg"] == nb["cfg"], "cfg differ between identical-seed runs"

        # Non-volatile outputs should be identical aside from tiny float epsilon
        # First compare dicts ignoring the two metric fields (checked with tolerance below)
        na_out = {k: v for k, v in na["out"].items() if k not in {"val/acc", "val/loss"}}
        nb_out = {k: v for k, v in nb["out"].items() if k not in {"val/acc", "val/loss"}}
        assert na_out == nb_out, f"stable out fields differ: {na_out.keys() ^ nb_out.keys()}"

        # Now we validate the metric fields:
        _metrics_close(na, nb, eps=EPS)


# Local test (does not require outside imports).
def test_jsonl_schema_shape_only(tmp_path):
    """
    Ensures sanity of _validate_schema given dummy aprox (lastest updated: study/v1).
    (dummy MAY be comparable to real results.jsonl output IF maintained).
    """

    # see legacy/current schematics above
    rec = make_dummy_result_record()
    p = tmp_path / "results.jsonl"

    # might work to append info:
    p.write_text(json.dumps(rec) + "\n")

    # Tests sanity of the jsonl reader.
    back = _read_jsonl(p)[0]

    _validate_schema(back)  # tests keys vs schema sanity in of itself
