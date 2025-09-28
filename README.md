# Ablation Harness + Repro Template

One-command ablations with multi-seed determinism, JSONL results, Markdown report, and CI.

Quickstart:
    python -m pip install -U pip
    pip install -e .[dev]
    pre-commit install
    python -m ablation_harness.ablate -c experiments/baseline.yaml
