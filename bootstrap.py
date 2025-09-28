import pathlib
import textwrap

root = pathlib.Path(".")


def w(path, content):
    path = root / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")


# keep dirs
for d in [
    "src/ablation_harness",
    "scripts",
    "experiments",
    "tests",
    "runs",
    "reports/plots",
    ".github/workflows",
]:
    (root / d).mkdir(parents=True, exist_ok=True)
for keep in ["runs/.gitkeep", "reports/.gitkeep"]:
    w(keep, "")

w(
    ".gitignore",
    """
.venv/
__pycache__/
.pytest_cache/
dist/
*.egg-info/
.ipynb_checkpoints/
runs/**
!runs/.gitkeep
reports/**
!reports/.gitkeep
""",
)

w(
    "README.md",
    """
# Ablation Harness + Repro Template

One-command ablations with multi-seed determinism, JSONL results, Markdown report, and CI.

Quickstart:
    python -m pip install -U pip
    pip install -e .[dev]
    pre-commit install
    python -m ablation_harness.ablate -c experiments/baseline.yaml
""",
)

w("LICENSE", "MIT License — (c) 2025 L")

w(
    "pyproject.toml",
    """
[project]
name = "ablation-harness"
version = "0.1.0"
description = "Ablation harness + reproducibility template"
readme = "README.md"
requires-python = ">=3.10"
dependencies = ["numpy>=1.26","omegaconf>=2.3","matplotlib>=3.8"]

[project.optional-dependencies]
dev = ["pytest>=8.2","pytest-cov>=5.0","pre-commit>=3.7","ruff>=0.5","black>=24.8","isort>=5.13"]

[tool.pytest.ini_options]
addopts = "-q"
testpaths = ["tests"]

[tool.black]
line-length = 100

[tool.isort]
profile = "black"

[tool.ruff]
line-length = 100
select = ["E","F","I","C90"]
""",
)

w(
    ".pre-commit-config.yaml",
    """
repos:
  - repo: https://github.com/psf/black
    rev: 24.8.0
    hooks: [{id: black}]
  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.5.7
    hooks: [{id: ruff}]
  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks: [{id: isort}]
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: end-of-file-fixer
      - id: trailing-whitespace
""",
)

w(
    ".github/workflows/ci.yml",
    """
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.12' }
      - name: Install
        run: |
          python -m pip install -U pip
          pip install -e .[dev]
          pre-commit install
      - name: Lint
        run: pre-commit run --all-files
      - name: Tests
        run: pytest -q
""",
)

w(
    "experiments/baseline.yaml",
    """
seeds: [0, 1, 2, 3]
experiments:
  - name: baseline
    n_samples: 1000
  - name: bigger
    n_samples: 3000
""",
)

w("src/ablation_harness/__init__.py", "__all__ = []")

w(
    "src/ablation_harness/seed_utils.py",
    """
import os, random
import numpy as np

def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
""",
)

w(
    "src/ablation_harness/ablate.py",
    r"""
import argparse, time, json, pathlib
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
from omegaconf import OmegaConf
from .seed_utils import set_seed
import matplotlib.pyplot as plt

@dataclass
class Experiment:
    name: str
    n_samples: int = 1000

def run_one(exp: Experiment, seed: int) -> Dict[str, Any]:
    set_seed(seed)
    x = np.random.randn(exp.n_samples)
    loss = float(np.mean(np.abs(x)))
    return {"config_name": exp.name, "seed": seed, "metrics": {"dummy_loss": loss}}

def load_cfg(path: str):
    cfg = OmegaConf.load(path)
    exps = [Experiment(**e) for e in cfg.experiments]
    seeds = list(cfg.seeds)
    return exps, seeds

def aggregate(rows: List[Dict[str, Any]]):
    by = {}
    for r in rows:
        by.setdefault(r["config_name"], []).append(r["metrics"]["dummy_loss"])
    table = []
    for k, v in by.items():
        import numpy as np
        arr = np.array(v, float)
        table.append((k, float(arr.mean()), float(arr.std(ddof=1)), len(arr)))
    table.sort()
    return table

def write_report(table, out_dir: pathlib.Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    md = ["# Ablation Report", "", "| config | mean | std | n |", "|---|---:|---:|---:|"]
    for name, mean, std, n in table:
        md.append(f"| {name} | {mean:.6f} | {std:.6f} | {n} |")
    (out_dir / "ablate.md").write_text("\\n".join(md) + "\\n", encoding="utf-8")

def plot_seed_variance(rows: List[Dict[str, Any]], out_dir: pathlib.Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    by = {}
    for r in rows:
        by.setdefault(r["config_name"], []).append(r["metrics"]["dummy_loss"])
    for name, vals in by.items():
        plt.figure()
        plt.title(f"Seed variance: {name}")
        plt.plot(vals, marker="o")
        plt.xlabel("seed idx")
        plt.ylabel("dummy_loss")
        plt.tight_layout()
        (out_dir / f"seed_variance_{name}.png").parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / f"seed_variance_{name}.png", dpi=150)
        plt.close()

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True, help="Path to experiments YAML")
    p.add_argument("--out", default=None, help="Runs dir (default: runs/<ts>)")
    args = p.parse_args(argv)

    exps, seeds = load_cfg(args.config)
    ts = time.strftime("%Y%m%d-%H%M%S")
    runs_dir = pathlib.Path(args.out) if args.out else pathlib.Path("runs")/f"ablate-{ts}"
    reports_dir = pathlib.Path("reports")
    plots_dir = reports_dir / "plots"
    runs_dir.mkdir(parents=True, exist_ok=True)

    results_path = runs_dir / "results.jsonl"
    rows = []
    t0 = time.time()
    for e in exps:
        for s in seeds:
            r = run_one(e, s)
            r["elapsed_sec"] = round(time.time() - t0, 4)
            r["timestamp"] = ts
            rows.append(r)

    with results_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\\n")

    table = aggregate(rows)
    write_report(table, reports_dir)
    plot_seed_variance(rows, plots_dir)
    print(f"Wrote {results_path} and reports/ablate.md")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
""",
)

w(
    "tests/test_seed_utils.py",
    """
from ablation_harness.seed_utils import set_seed
import numpy as np

def test_set_seed_repro():
    set_seed(123)
    a = np.random.randn(5)
    set_seed(123)
    b = np.random.randn(5)
    assert (a == b).all()
""",
)

w(
    "tests/test_ablate_smoke.py",
    """
import pathlib, json
from ablation_harness import ablate as A

def test_ablate_smoke(tmp_path: pathlib.Path):
    cfg = tmp_path/'exp.yaml'
    cfg.write_text('seeds: [0,1]\\nexperiments: [{name: x, n_samples: 10}]\\n')
    out = tmp_path/'runs'
    A.main(['-c', str(cfg), '--out', str(out)])
    lines = next(out.glob('**/results.jsonl')).read_text().strip().splitlines()
    assert len(lines) == 2
    j = json.loads(lines[0])
    assert "metrics" in j and "dummy_loss" in j["metrics"]
""",
)

print("Bootstrap complete.")
