"""
Tests CLI.
"""


def test_trainer_run_smoke(tmp_path):
    """Quick trainer smoke run."""
    from ablation_harness import train

    out = train.run(
        {
            "study_name": "tinycnn_tester",
            "seed": 0,
            "epochs": 1,
            "out_dir": tmp_path.as_posix(),
            "data": {
                "dataset": "cifar10",
                "subset": 256,
                "batch_size": 128,
                "num_workers": 0,
            },
            "model": {
                "name": "tinycnn",
                "hidden": 64,
                "dropout": 0.0,
            },
            "optim": {
                "optimizer": "adam",
                "lr": 0.001,
                "wd": 0.0,
            },
        }
    )
    assert "val/acc" in out and 0.0 <= out["val/acc"] <= 1.0


def _load_jsonl(path):
    """Loads jsonl."""
    import json

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def test_cli_run_smoke_study(tmp_path):
    """Test the study schema actually works."""
    import subprocess
    import sys

    import yaml

    study_name = "moons_study"
    cfg = {
        "schema": "study/v1",
        "study_name": study_name,
        "metric": "val/acc",
        "goal": "max",
        "epochs": 1,
        "seed": 0,
        "data": {"dataset": "moons", "batch_size": 64},
        "model": {"name": "mlp", "hidden": 32},
        "optim": {"optimizer": "adam", "lr": 0.001, "wd": 0.0},
        "variants": [
            {"overrides": {"model": {"hidden": 16}, "optim": {"lr": 0.1}}},
            {"overrides": {"model": {"hidden": 64}, "optim": {"lr": 0.001}}},
        ],
    }
    cfg_path = tmp_path / "study.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    out_dir = tmp_path / "test_out"
    cmd = [
        sys.executable,
        "-m",
        "ablation_harness.cli",
        "run",
        "--config",
        str(cfg_path),
        "--out_dir",
        str(out_dir),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, f"stderr:\n{r.stderr}\nstdout:\n{r.stdout}"

    # New executor writes the global ledger at <out_dir>/results.jsonl
    results = out_dir / "results.jsonl"
    assert results.exists()
    rows = _load_jsonl(results)
    # should have 2 runs (baseline+1 variant) or 2 variants (depends on your planner).
    assert len(rows) >= 2
    # plan.json should exist under <out_dir>/<study_name>/plan.json
    plan = out_dir / study_name / "plan.json"
    assert plan.exists()


def test_cli_dry_run_plan_only(tmp_path):
    """Test in dry run, only plan created."""
    import subprocess
    import sys

    import yaml

    study_name = "dryrun_study"
    cfg = {
        "schema": "study/v1",
        "study_name": study_name,
        "metric": "val/acc",
        "goal": "max",
        "epochs": 1,
        "seed": 0,
        "data": {"dataset": "moons", "batch_size": 64},
        "model": {"name": "mlp", "hidden": 32},
        "optim": {"optimizer": "adam", "lr": 0.001, "wd": 0.0},
        "variants": [
            {"overrides": {"model": {"hidden": 16}, "optim": {"lr": 0.001}}},
            {"overrides": {"model": {"hidden": 64}, "optim": {"lr": 0.1}}},
        ],
    }
    cfg_path = tmp_path / "dry.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    out_dir = tmp_path / "dry_out"
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "ablation_harness.cli",
            "run",
            "--config",
            str(cfg_path),
            "--out_dir",
            str(out_dir),
            "--dry_run",
            "--clean_run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, f"stderr:\n{r.stderr}\nstdout:\n{r.stdout}"

    # Plan exists
    plan = out_dir / study_name / "plan.json"
    assert plan.exists()

    # No run directories should be created
    study_dir = out_dir / study_name
    run_dirs = [p for p in study_dir.iterdir() if p.is_dir() and p.name != ".trash"]
    # only .trash or none are allowed
    assert all(p.name in {".trash"} for p in run_dirs) or len(run_dirs) == 0

    # If your executor writes a results.jsonl even for dry runs, ensure every row is marked dry_run
    results = out_dir / "results.jsonl"
    if results.exists():
        rows = _load_jsonl(results)
        assert all(row.get("out", {}).get("dry_run") for row in rows)


def test_study_seed_override(tmp_path):
    """Forced seed actually works."""
    import subprocess
    import sys

    import yaml

    study_name = "seed_override"
    cfg = {
        "schema": "study/v1",
        "study_name": study_name,
        "metric": "val/acc",
        "goal": "max",
        "epochs": 1,
        "seed": 0,
        "data": {"dataset": "moons", "batch_size": 64},
        "model": {"name": "mlp", "hidden": 32},
        "optim": {"optimizer": "adam", "lr": 0.001, "wd": 0.0},
        "variants": [
            {"overrides": {"model": {"hidden": 16}, "optim": {"lr": 0.001}}},
            {"overrides": {"model": {"hidden": 64}, "optim": {"lr": 0.1}}},
        ],
        "seeds": [0, 1, 2],
    }

    cfg_path = tmp_path / "study.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    out_dir = tmp_path / "out"
    forced = 500
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "ablation_harness.cli",
            "run",
            "--config",
            str(cfg_path),
            "--out_dir",
            str(out_dir),
            "--seed",
            str(forced),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, f"stderr:\n{r.stderr}\nstdout:\n{r.stdout}"
    rows = _load_jsonl(out_dir / "results.jsonl")
    seeds_used = {row.get("cfg", {}).get("seed") for row in rows}
    assert seeds_used == {forced}, f"Unexpected seeds used: {seeds_used}"
