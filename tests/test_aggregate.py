import json
import subprocess
import sys
from pathlib import Path


def _write_jsonl(p: Path, *, rows=5):
    """Write JSONL file."""
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "cfg": {
                "dataset": "cifar10",
                "subset": 256,
                "epochs": 1,
                "batch_size": 128,
                "seed": 0,
                "model": "tinycnn",
                "optimizer": "adam",
                "lr": 0.0003,
                "wd": 0.0,
                "ema": "false",
                "_study": "tinycnn_cifar10_wk2",
                "_variant": "varient_1",
            },
            "out": {
                "seed": 0,
                "val/acc": 0.1,
                "val/loss": 2.30482744102478,
                "params": 7738,
                "dataset": "cifar10",
                "model_used": "TinyCNN",
                "run_id": "generic_any_id",
                "run_dir": "runs/logs\\generic_any_id",
                "ckpt": "runs/logs\\generic_any_id\\ckpts.pt",
                "spect_stats": "null",
                "loss_log": "runs/logs\\generic_any_id\\loss.jsonl",
                "_elapsed_sec": 5.551,
            },
            "_i": 0,
        }
        for _ in range(rows)
    ]
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _load_jsonl(path: str | Path):
    """Load JSONL file."""
    path = Path(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def test_aggregate_markdown_report_written(tmp_path):

    jsonl_path = tmp_path / "jsonl_dir/results.json"

    _write_jsonl(jsonl_path)
    _ = _load_jsonl(jsonl_path)  # rows
    # print(rows)

    md_path = tmp_path / "test_ablation_agg"
    p = Path(md_path)

    """Useage: (example)
        python -m scripts.aggregate \
        runs/wk2_tinycnn/results.jsonl \
        --metric val/acc --goal max \
        --cols optimizer lr wd ema \
        --timing _elapsed_sec \
        --out reports/wk2_ablation.md"""

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.aggregate",
            "--metric",
            "val/acc",
            "--cols",
            "optimizer lr wd ema",
            "--goal",
            "max",
            "--timing",
            "_elapsed_sec",
            str(jsonl_path),
            "--out",
            str(md_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"

    assert p.exists() and p.stat().st_size > 0
