import importlib
import json
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from .fs_utils import quarantine_then_delete, robust_rmtree
from .io_jsonl import append_jsonl


@dataclass
class RunResult:
    i: int
    cfg: Dict[str, Any]
    out: Dict[str, Any]


@dataclass
class ExecSummary:
    jsonl_path: str
    n_ok: int
    n_err: int
    results: List[RunResult]


def _get_name_of(cfg: Dict[str, Any]) -> str:
    """Find study_name or sweep_name."""
    return cfg.get("study_name") or cfg.get("sweep_name") or "study"


def _run_id_of(cfg: Dict[str, Any]) -> str:
    """Gets the run_id of this plan."""
    rid = cfg.get("run_id")
    if not rid:
        raise ValueError("Planner must assign a unique run_id per run.")
    return rid


def _ensure_plan(out_dir: Path, study: str, runs: List[Dict[str, Any]]) -> Path:
    """Ensures plan.json exists."""
    study_dir = out_dir / study
    study_dir.mkdir(parents=True, exist_ok=True)
    plan_path = study_dir / "plan.json"
    payload = {
        "study_name": study,
        "n_runs": len(runs),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runs": [{"run_id": r.get("run_id"), "cfg": r} for r in runs],
    }
    plan_path.write_text(json.dumps(payload, indent=2))
    return plan_path


def run_many(  # noqa C901
    runs: List[Dict[str, Any]],
    trainer_mod: str,
    out_dir: str,
    metric: str = "val/acc",
    goal: str = "max",
    concurrency: int = 1,
    dry_run: bool = False,
    resume_failed: bool = False,
    clean_run: bool = False,
    resume: bool = False,
    max_fail: int = 1,
) -> ExecSummary:
    """High level running orchestrator."""
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Assume all runs belong to the same study; if mixed, we’ll group by study
    # (simple path: write one plan per study encountered)
    runs_by_study: Dict[str, List[Dict[str, Any]]] = {}
    for r in runs:
        runs_by_study.setdefault(_get_name_of(r), []).append(r)
    for study, rs in runs_by_study.items():
        _ensure_plan(out_root, study, rs)

    jsonl_path = out_root / "results.jsonl"
    mod = importlib.import_module(trainer_mod)
    run_fn = getattr(mod, "run")

    results: List[RunResult] = []
    n_ok = n_err = 0

    for i, cfg in enumerate(runs):
        t0 = time.time()
        out: Dict[str, Any]
        study = _get_name_of(cfg)
        run_id = _run_id_of(cfg)
        study_dir = out_root / study
        run_dir = study_dir / run_id

        # Make study dir early; set cfg for trainer

        study_dir.mkdir(parents=True, exist_ok=True)
        cfg["out_dir"] = str(study_dir)  # trainer will create subdirs under this using run_id
        cfg["run_id"] = run_id  # ensure present (planner should have set it)

        # Existence rule
        if run_dir.exists():
            if clean_run:
                print("[executor] Clean run flag detected.")
                # Try direct remove; if denied, quarantine and delete
                if not robust_rmtree(run_dir):
                    print("[executor.py] Delete failed. Trying quarantine_then_delete...")
                    qdir, deleted = quarantine_then_delete(run_dir)
                    if not deleted:
                        print("[executor.py] Quarentine and delete failed. appending: DELETE_ME.txt to dir")
                        # Leave a marker so you can clean later
                        (qdir / "DELETE_ME.txt").write_text("Windows lock prevented immediate removal.")
                    # after this point, we consider it clean (either deleted or quarantined)
            elif resume:
                pass  # let trainer resume from ckpts/last.pt
            else:
                out = {
                    "error": f"run_dir exists for {run_id}; use --clean_run or --resume",
                    "run_dir": str(run_dir),
                }
                n_err += 1
                out["_elapsed_sec"] = round(time.time() - t0, 3)
                append_jsonl(str(jsonl_path), {"cfg": cfg, "out": out, "_i": i})

                results.append(RunResult(i, cfg, out))
                # optional early-stop
                if n_err >= max_fail:
                    break
                print(f"[{i+1}/{len(runs)}] {run_id} -> SKIPPED (exists)")
                continue

        # Dry-run?
        if dry_run:
            out = {"dry_run": True, "run_dir": str(run_dir)}
            n_ok += 1
        else:
            try:
                out = run_fn(cfg)  # runner.
            except Exception as e:
                out = {"error": str(e), "trace": traceback.format_exc(limit=4)}
                n_err += 1
            else:
                n_ok += 1

        out["_elapsed_sec"] = round(time.time() - t0, 3)
        rec = {"cfg": cfg, "out": out, "_i": i}
        append_jsonl(str(jsonl_path), rec)
        results.append(RunResult(i, cfg, out))

        disp_name = cfg.get("_variant", run_id)
        print(f"[{i+1}/{len(runs)}] {disp_name} -> {out.get(metric,'NA')} ({out.get('_elapsed_sec','?')}s)")

        if n_err >= max_fail:
            break

    return ExecSummary(str(jsonl_path), n_ok, n_err, results)
