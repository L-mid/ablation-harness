import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunLayout:
    root: Path
    ckpts: Path
    logs: Path
    plots: Path
    results: Path
    run_id: str


def resolve_run_layout(base, run_id: str, clean: bool = False) -> RunLayout:
    root = (Path(base) / run_id).resolve()
    if clean and root.exists():
        shutil.rmtree(root)
    (root / "ckpts").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "plots").mkdir(parents=True, exist_ok=True)
    return RunLayout(
        root=root,
        ckpts=root / "ckpts",
        logs=root / "logs",
        plots=root / "plots",
        results=root / "loss.jsonl",
        run_id=run_id,
    )
