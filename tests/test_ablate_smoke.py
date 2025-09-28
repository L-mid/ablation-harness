import json
import pathlib

from ablation_harness import ablate as A


def test_ablate_smoke(tmp_path: pathlib.Path):
    cfg = tmp_path / "exp.yaml"
    cfg.write_text("seeds: [0,1]\nexperiments: [{name: x, n_samples: 10}]\n")
    out = tmp_path / "runs"
    A.main(["-c", str(cfg), "--out", str(out)])
    lines = next(out.glob("**/results.jsonl")).read_text().strip().splitlines()
    assert len(lines) == 2
    j = json.loads(lines[0])
    assert "metrics" in j and "dummy_loss" in j["metrics"]
    pairs = {(json.loads(ln)["config_name"], json.loads(ln)["seed"]) for ln in lines}
    assert len(pairs) == 2  # uniqueness by (config, seed)
