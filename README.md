# Ablation Harness + Repro Template

One-command ablations with multi-seed determinism, JSONL results, Markdown report, and CI.

Quickstart:
    python -m pip install -U pip wheel
    pip install -e ".[dev,torch-cpu]"
    pre-commit install
    python -m ablation_harness.ablate --config experiments/baseline.yaml

![CI](https://github.com/l-mid/ablation-harness/actions/workflows/ci.yml/badge.svg?branch=main)

### Wkly Reports (for 4 wks):
- [Week 1 Baseline — TinyCNN on CIFAR-10](docs/reports/wk1_baseline.md)
