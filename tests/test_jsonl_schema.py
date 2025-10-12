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

CFG_REQUIRED = {
    "dataset",
    "subset",
    "epochs",
    "batch_size",
    "seed",
    "model",
    "optimizer",
    "lr",
    "wd",
    "ema",
    "out_dir",
    "dropout",
    "run_id",
    "_study",
    "_variant",
}

OUT_REQUIRED = {
    "seed",
    "val/acc",
    "val/loss",
    "params",
    "dataset",
    "model_used",
    "run_id",
    "run_dir",
    "ckpt",
    "spect_stats",  # may be null
    "loss_log",
    "_elapsed_sec",
}

# Fields that are expected to vary even when deterministic (paths/timing)
VOLATILE_OUT = {"run_dir", "ckpt", "loss_log", "_elapsed_sec"}  # some (run_id) not included here.


def make_dummy_result_record() -> Dict[str, Any]:
    """Mockup jsonl contents approximation for test_jsonl_schema_shape_only ."""
    # Types match your real outputs; values are arbitrary but valid
    return {
        "cfg": {
            "dataset": "cifar10",
            "subset": 64,
            "epochs": 1,
            "batch_size": 64,
            "seed": 42,
            "model": "tinycnn",
            "optimizer": "adam",
            "lr": 1e-3,
            "wd": 0.0,
            "ema": False,
            "out_dir": "runs/dummy",
            "dropout": 0.3,
            "run_id": "dummy_run",
            "_study": "dummy_study",
            "_variant": "baseline",
        },
        "out": {
            "seed": 42,
            "val/acc": 0.123,  # float
            "val/loss": 2.345,  # float
            "params": 7738,  # int
            "dataset": "cifar10",
            "model_used": "TinyCNN",
            "run_id": "dummy_run",
            "run_dir": "runs/dummy/dummy_run",
            "ckpt": "runs/dummy/dummy_run/ckpts/ckpt.pt",
            "spect_stats": None,  # allowed to be null in your real files
            "loss_log": "runs/dummy/dummy_run/loss.jsonl",
            "_elapsed_sec": 0.001,  # float
        },
        "_i": 0,
    }


# Small numeric tolerance for floats (even on CPU tiny diffs can appear with
# different BLAS or Python minor versions)
EPS = 1e-8


def _read_jsonl(path: pathlib.Path):
    """A very compact jsonl reader."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _validate_schema(rec: Dict[str, Any]):
    """
    Validates the resulting jsonl contents agaisnt this EXTACTLY for certain required fields. (requires upkeep).
    Some fields/new fields ingored (might miss stuff).
    """

    print("OUTPUT LOOKS LIKE:", rec)

    # top level keys:
    assert set(rec.keys()) == TOP_REQUIRED, f"Top-level keys mismatch: {set(rec.keys())}"
    cfg = rec["cfg"]
    out = rec["out"]

    assert CFG_REQUIRED.issubset(cfg.keys()), f"cfg keys missing: {CFG_REQUIRED - set(cfg.keys())}"  # this here will bite in recent refactors.
    assert OUT_REQUIRED.issubset(out.keys()), f"out keys missing: {OUT_REQUIRED - set(out.keys())}"

    # Spot-check types on some critical fields
    assert isinstance(cfg["seed"], int)
    assert isinstance(cfg["epochs"], int)
    assert isinstance(cfg["subset"], int)
    assert isinstance(cfg["lr"], (int, float))
    assert isinstance(cfg["ema"], bool)
    assert isinstance(cfg["run_id"], str)

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

    # Normalize out paths
    out = rec["out"]
    for k in list(out.keys()):
        if k in VOLATILE_OUT:
            # Remove truly volatile fields
            out.pop(k, None)
        elif isinstance(out[k], str):
            out[k] = _normalize_paths(out[k])

    # Normalize cfg.out_dir path
    cfg = rec["cfg"]
    if isinstance(cfg.get("out_dir"), str):
        cfg["out_dir"] = _normalize_paths(cfg["out_dir"])

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
    We keep the run very small to be CI-friendly.

    This test will probably fail if you've done anything to ablate.py OR train.py recently,
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
            "ablation_harness.ablate",
            "--config",
            str(study_yaml),
            "--out",
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
    Ensures sanity of _validate_schema given dummy aprox (at present).
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
