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


def test_walltime_plot_smoke(tmp_path, write_jsonl_rows):

    jsonl_path = tmp_path / "jsonl_dir/results.json"

    write_jsonl_rows(jsonl_path)
    _ = _load_jsonl(jsonl_path)  # rows
    # print(rows)

    graph_path = tmp_path / "test_walltime_graph"
    p = Path(graph_path)

    """
    Useage:
        python -m ablation_harness.plot_walltime
        runs/wk2_tinycnn/results.jsonl
        --label-keys optimizer lr ema
        --out runs/wk2_tinycnn/plots
    """

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ablation_harness.plot_walltime",
            str(jsonl_path),
            "--label-keys",
            "optim.optimizer optim.lr optim.wd ema.enabled",
            "--out",
            str(graph_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"

    out_path = graph_path / "walltime_bar.png"  # default output of this plotter

    assert out_path.exists() and out_path.stat().st_size > 0
