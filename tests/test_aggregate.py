import json
import subprocess
import sys
from pathlib import Path

from scripts.aggregate import aggregate_once


def _write_jsonl_complete(p: Path, *, rows=5):
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


def test_aggregate_markdown_report_written_smoke(tmp_path):

    jsonl_path = tmp_path / "jsonl_dir/results.json"

    _write_jsonl_complete(jsonl_path)
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


# -------------- Behaviour Tests ---------------------


def _jsonl_line(name, metric=None, cfg_extra=None, out_extra=None):
    cfg = {"name": name}
    if cfg_extra:
        cfg.update(cfg_extra)
    out = {}
    if metric is not None:
        out["val/acc"] = metric
    if out_extra:
        out.update(out_extra)
    return json.dumps({"cfg": cfg, "out": out})


def _write_jsonl(path: Path, rows):
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _table_order(md_text: str, column_name="name"):
    """
    Parse the markdown table produced by to_markdown() and return the sequence of 'name' values per row.
    Table shape is: | config | <cfg_cols...> | metric | timing? |
    We locate the 3rd line onward (data rows), then pick the 'name' column by header index.
    """
    lines = [ln.strip() for ln in md_text.splitlines() if ln.strip()]
    # Header lines start with '| ... |' and the line above is a small preamble
    table_start = next(i for i, ln in enumerate(lines) if ln.startswith("| config |"))
    header = [h.strip() for h in lines[table_start].strip("| ").split("|")]
    sep = lines[table_start + 1]
    assert set(sep.replace(" ", "")) == {"|", "-"}  # markdown separator line

    # Find the index of the requested column
    try:
        j = header.index(column_name)
    except ValueError:
        raise AssertionError(f"Column {column_name!r} not found in header: {header}")

    # Data rows start after the separator
    data_rows = lines[table_start + 2 :]
    values = []
    for r in data_rows:
        cells = [c.strip() for c in r.strip("| ").split("|")]
        # guard: ensure we have enough columns (config + name + metric + maybe timing)
        assert len(cells) >= j + 1
        values.append(cells[j])
    return values


def _parse_markdown_table(md_text: str):
    lines = [ln for ln in md_text.splitlines() if ln.strip()]
    # find header line
    start = next(i for i, ln in enumerate(lines) if ln.strip().startswith("|"))
    header = [h.strip() for h in lines[start].strip("|").split("|")]
    # skip the separator line (---)
    data = []
    for ln in lines[start + 2 :]:
        if not ln.strip().startswith("|"):
            break
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        # pad if needed
        while len(cells) < len(header):
            cells.append("")
        data.append(dict(zip(header, cells)))
    return header, data


def test_max_sorting_and_stable_ties(tmp_path: Path):
    """
    Tests that even when skipped, rows apear in correct order.
    """

    src = tmp_path / "results.jsonl"
    out_md = tmp_path / "report.md"

    # Craft rows: two ties at 0.8 in order B then C, a best A=0.9, a worst D=0.1, and one to be skipped (None)
    rows = [
        _jsonl_line("B", 0.8, out_extra={"_elapsed_sec": 1.0}),
        _jsonl_line("C", 0.8, out_extra={"_elapsed_sec": 1.0}),
        _jsonl_line("A", 0.9, out_extra={"_elapsed_sec": 1.0}),
        _jsonl_line("D", 0.1, out_extra={"_elapsed_sec": 1.0}),
        _jsonl_line("SKIP", None),
    ]
    _write_jsonl(src, rows)

    aggregate_once(
        src=src,
        out_md=out_md,
        metric_key="val/acc",
        goal="max",
        cfg_cols=["name"],
        timing_key="_elapsed_sec",
    )
    text = out_md.read_text(encoding="utf-8")

    # Expect A first (best), then the tied B then C (stable order), then D
    assert _table_order(text) == ["1", "A", "0.900", "1.0", "2", "B", "0.800", "1.0", "3", "C", "0.800", "1.0", "4", "D", "0.100", "1.0"][1::4]  # row 1  # row 2  # row 3
    # Explanation: we selected the 'name' column; above assertion shows expected sequence ["A","B","C","D"]


def test_min_sorting_reverses_order(tmp_path: Path):
    """Tests min sorting reverses the order in the sorter."""

    src = tmp_path / "results.jsonl"
    out_md = tmp_path / "report.md"

    rows = [
        _jsonl_line("B", 0.8, out_extra={"_elapsed_sec": 2.0}),
        _jsonl_line("C", 0.8, out_extra={"_elapsed_sec": 2.0}),
        _jsonl_line("A", 0.9, out_extra={"_elapsed_sec": 2.0}),
        _jsonl_line("D", 0.1, out_extra={"_elapsed_sec": 2.0}),
    ]
    _write_jsonl(src, rows)

    aggregate_once(
        src=src,
        out_md=out_md,
        metric_key="val/acc",
        goal="min",
        cfg_cols=["name"],
        timing_key="_elapsed_sec",
    )
    text = out_md.read_text(encoding="utf-8")
    # For min: smallest first → D, then the ties B then C (stable), then A
    assert _table_order(text) == ["D", "B", "C", "A"]


def test_skips_missing_metric_and_writes_placeholder(tmp_path: Path):
    """
    Test the aggregator skips properly in the case of None.
    """

    src = tmp_path / "results.jsonl"
    out_md = tmp_path / "report.md"

    # All rows missing 'val/acc' → should emit the placeholder file
    rows = [
        _jsonl_line("X", None),
        _jsonl_line("Y", None),
    ]
    _write_jsonl(src, rows)

    aggregate_once(
        src=src,
        out_md=out_md,
        metric_key="val/acc",
        goal="max",
        cfg_cols=["name"],
        timing_key="_elapsed_sec",
    )
    assert out_md.exists()
    assert out_md.read_text(encoding="utf-8").strip() == "_No complete runs found yet._"


def test_bool_column_on_off_format(tmp_path: Path):
    """Tests bool colum for expected behaviour."""

    src = tmp_path / "results.jsonl"
    out_md = tmp_path / "report.md"

    src.write_text(
        "\n".join(
            [
                '{"cfg":{"name":"E","ema":true},"out":{"val/acc":0.5,"_elapsed_sec":3.0}}',
                '{"cfg":{"name":"F","ema":false},"out":{"val/acc":0.4,"_elapsed_sec":3.0}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    aggregate_once(
        src=src,
        out_md=out_md,
        metric_key="val/acc",
        goal="max",
        cfg_cols=["name", "ema"],
        timing_key="_elapsed_sec",
    )

    text = out_md.read_text(encoding="utf-8")
    header, rows = _parse_markdown_table(text)

    # Quick sanity: expected columns are present
    for col in ("name", "ema", "val/acc", "_elapsed_sec"):
        assert col in header

    # Check row order (goal=max → E then F)
    assert [r["name"] for r in rows] == ["E", "F"]

    # Check boolean rendering specifically
    name_to_ema = {r["name"]: r["ema"] for r in rows}
    assert name_to_ema == {"E": "on", "F": "off"}


def test_timing_column_numeric_format(tmp_path: Path):
    """
    Test timing formatting works as expected in elapsed seconds.
    """

    src = tmp_path / "results.jsonl"
    out_md = tmp_path / "report.md"

    rows = [
        _jsonl_line("G", 0.2, out_extra={"_elapsed_sec": 12.3456}),
        _jsonl_line("H", 0.3, out_extra={"_elapsed_sec": 0}),
    ]
    _write_jsonl(src, rows)

    aggregate_once(
        src=src,
        out_md=out_md,
        metric_key="val/acc",
        goal="max",
        cfg_cols=["name"],
        timing_key="_elapsed_sec",
    )
    text = out_md.read_text(encoding="utf-8")
    # timing is formatted to one decimal place
    assert "12.3" in text
    assert "0.0" in text
