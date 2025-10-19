"""A test for a plot for comparing cfg diffs."""

import json
import subprocess
import sys
from pathlib import Path


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


def test_ablation_plot_smoke(tmp_path, write_jsonl_rows):

    jsonl_path = tmp_path / "jsonl_dir/results.json"

    write_jsonl_rows(jsonl_path)  # full study/v1 fixture, due to evaluating cfg differences
    _ = _load_jsonl(jsonl_path)  # rows
    # print(rows)

    graph_path = tmp_path / "test_ablation_graph"
    p = Path(graph_path)

    """
    Useage:

    python -m ablation_harness.plot_ablation
    runs/wk2_tinycnn/results.jsonl
    --metric val/acc
    --goal max
    --label-keys optim.optimizer optim.lr ema.enabled
    --out runs/wk2_tinycnn/plots
    """

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ablation_harness.plot_ablation",
            str(jsonl_path),
            "--metric",
            "val/acc",
            "--label-keys",
            "optim.optimizer optim.lr ema.enabled",
            "--goal",
            "max",
            "--out",
            str(graph_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"

    assert p.exists()  # and p.stat().st_size > 0        interesting error
