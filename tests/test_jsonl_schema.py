"""
Not imported/used in practice.
"""

import json
import pathlib
import tempfile

REQUIRED = {
    "run_id": str,
    "step": int,
    "split": str,
    "loss": float,
    "timestamp": str,
    "seed": int,
    "cfg_hash": str,
}


def validate_row(row: dict):
    assert set(row.keys()) == set(REQUIRED.keys())
    for k, t in REQUIRED.items():
        assert isinstance(row[k], t), f"{k} must be {t}"


def test_jsonl_row_schema():
    row = {
        "run_id": "r1",
        "step": 0,
        "split": "train",
        "loss": 1.234,
        "timestamp": "2025-10-01T15:00:00Z",
        "seed": 1337,
        "cfg_hash": "abc123",
    }
    validate_row(row)
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d, "train.jsonl")
        p.write_text(json.dumps(row) + "\n")
        back = json.loads(p.read_text().splitlines()[0])
        validate_row(back)
