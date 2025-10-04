import os
import subprocess
import sys
from pathlib import Path

import yaml


def test_trainer_run_smoke():
    from ablation_harness import trainer

    out = trainer.run(
        {
            "model": "mlp",
            "hidden": 32,
            "dropout": 0.0,
            "lr": 1e-3,
            "wd": 0.0,
            "epochs": 1,
            "batch_size": 64,
            "seed": 0,
            "dataset": "moons",
        }
    )
    assert "val/acc" in out and 0.0 <= out["val/acc"] <= 1.0


def test_ablate_cli_smoke(tmp_path):
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

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    out_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        "-m",
        "ablation_harness.ablate",
        "--config",
        str(cfg_path),
        "--out_dir",
        str(out_dir),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, f"stderr:\n{r.stderr}\nstdout:\n{r.stdout}"
    assert (out_dir / "results.jsonl").exists()
    assert (out_dir / "summary.csv").exists()


def test_dry_run_makes_no_files(tmp_path, monkeypatch):
    from pathlib import Path

    out_dir = tmp_path / "ablate-dry"

    cfg = tmp_path / "b.yaml"
    cfg_dict = {
        "outdir": Path(tmp_path).as_posix(),
        "seed": 1337,
        "dataset": {"name": "synthetic", "root": "."},
    }
    cfg.write_text(yaml.safe_dump(cfg_dict, sort_keys=False))
    import json
    import subprocess

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ablation_harness.ablate",
            "--config",
            str(cfg),
            "--dry-run",
            "--out_dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"

    plan = json.loads(proc.stdout)
    assert not os.path.exists(plan["out_dir"])  # nothing created

    for r in plan["runs"]:
        for p in r["planned_artifacts"]:
            assert not Path(p).exists()


def test_study_seed_override(tmp_path, monkeypatch):

    out_dir = tmp_path / "ablate-seeded"

    cfg_path = tmp_path / "b.yaml"

    cfg = {
        "schema": "study/test",
        "study_name": "moons_test",
        "metric": "val/acc",
        "goal": "max",
        "baseline": {"model": "mlp", "seed": 0, "dataset": "moons", "lr": 0.001},
        "seeds": [0, 1, 2, 3],
        "variants": None,
    }
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    import json
    import subprocess

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ablation_harness.ablate",
            "--config",
            str(cfg_path),
            "--out_dir",
            str(out_dir),
            "--seed",
            str(500),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"

    def load_jsonl(path: str | Path):
        path = Path(path)
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                rows.append(json.loads(s))
        return rows

    results_path = out_dir / "results.jsonl"

    assert results_path.exists()

    rows = load_jsonl(results_path)

    forced = 500
    seeds_used = {row["cfg"].get("seed") for row in rows}
    assert seeds_used == {forced}, f"Unexpected seeds found: {seeds_used}"

    # 2) Optional: ensure seed wasn't expanded as part of the grid
    #    (i.e., no duplicated configs except for seed)
    def cfg_without_seed(cfg: dict):
        return tuple(sorted((k, v) for k, v in cfg.items() if k != "seed" and not k.startswith("_")))

    uniq_no_seed = {cfg_without_seed(row["cfg"]) for row in rows}
    assert len(uniq_no_seed) == len(rows), "Looks like seed was expanded in the grid."

    # 3) Optional: sanity—every cfg explicitly contains seed
    assert all("seed" in row["cfg"] for row in rows)


def test_sweep_seed_override(tmp_path, monkeypatch):

    out_dir = tmp_path / "ablate-seeded"

    cfg_path = tmp_path / "b.yaml"

    cfg = {
        "base": {"model": "mlp", "seed": 0, "dataset": "moons"},
        "grid": {"lr": [0.01, 0.001], "dropout": [0.0, 0.5], "seed": [0, 1, 2]},
        "metric": "val/acc",
        "goal": "max",
    }
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    import json
    import subprocess

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ablation_harness.ablate",
            "--config",
            str(cfg_path),
            "--out_dir",
            str(out_dir),
            "--seed",
            str(500),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"

    def load_jsonl(path: str | Path):
        path = Path(path)
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                rows.append(json.loads(s))
        return rows

    results_path = out_dir / "results.jsonl"

    assert results_path.exists()

    rows = load_jsonl(results_path)

    forced = 500
    seeds_used = {row["cfg"].get("seed") for row in rows}
    assert seeds_used == {forced}, f"Unexpected seeds found: {seeds_used}"

    # 2) Optional: ensure seed wasn't expanded as part of the grid
    #    (i.e., no duplicated configs except for seed)
    def cfg_without_seed(cfg: dict):
        return tuple(sorted((k, v) for k, v in cfg.items() if k != "seed" and not k.startswith("_")))

    uniq_no_seed = {cfg_without_seed(row["cfg"]) for row in rows}
    assert len(uniq_no_seed) == len(rows), "Looks like seed was expanded in the grid."

    # 3) Optional: sanity—every cfg explicitly contains seed
    assert all("seed" in row["cfg"] for row in rows)
