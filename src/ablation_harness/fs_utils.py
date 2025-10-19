from __future__ import annotations

import os
import shutil
import stat
import tempfile
import time
import uuid
from pathlib import Path


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomically writes and saves bytes (used in checkpointing)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)  # atomic on same filesystem


def resolve_run_layout(base: Path, run_id: str) -> dict[str, Path]:
    """Ensures all paths are fixed and none windows-style."""
    root = (base / run_id).resolve()
    return {
        "root": root,
        "ckpts": root / "ckpts",
        "logs": root / "logs",
        "plots": root / "plots",
        "results": root / "results.jsonl",
    }


def _win_long(p: Path) -> str:
    """Long path prefix to avoid MAX_PATH issues."""
    s = str(p)
    if os.name == "nt" and not s.startswith("\\\\?\\"):
        # long-path prefix to avoid MAX_PATH issues
        return "\\\\?\\" + s
    return s


def _on_rm_error(func, path, exc_info):
    """Make path writable then retry func(path)."""
    try:
        os.chmod(path, stat.S_IWRITE)
    except Exception:
        pass
    try:
        func(path)
    except Exception:
        pass


def exists_nonempty(p: Path) -> bool:
    try:
        # Some re-creations are empty dirs; treat empty as deleted-success for our purposes.
        return p.exists() and any(p.iterdir())
    except PermissionError:
        # If listing fails, assume nonempty (conservative)
        return True


def robust_rmtree(path: Path, retries: int = 6, backoff: float = 0.25) -> bool:  # noqa C901
    """Best effort attempt to delete the path provided."""
    print(f"[rmtree] start: {path}")
    path = Path(path)
    if not path.exists():
        print("[rmtree] does not exist -> OK")
        return True

    ok = False
    last_err = None
    for i in range(retries):
        try:
            shutil.rmtree(_win_long(path), onerror=_on_rm_error)
            ok = True
            print(f"[rmtree] removed on try {i+1}")
            break
        except (PermissionError, OSError) as e:
            last_err = e
            print(f"[rmtree] try {i+1} failed: {e!r}")
            time.sleep(backoff * (2**i))

    if not ok and last_err:
        print("[rmtree] final sweep clearing read-only attrs …")
        try:
            for root, dirs, files in os.walk(_win_long(path)):
                for n in files:
                    try:
                        os.chmod(os.path.join(root, n), stat.S_IWRITE)
                    except Exception:
                        pass
            shutil.rmtree(_win_long(path), onerror=_on_rm_error)
            ok = True
            print("[rmtree] removed on final sweep")
        except Exception as e:
            ok = False
            print(f"[rmtree] final sweep failed: {e!r}")

    # Post-check: consider SUCCESS if dir is gone OR exists but empty.
    if ok and (not path.exists() or not exists_nonempty(path)):
        if path.exists():
            # show what’s inside (helps catch hidden resurrected files)
            items = list(path.iterdir())
            print(f"[rmtree] directory re-created EMPTY by watcher: {items}")
            # Attempt to remove empty dir
            try:
                path.rmdir()
                print("[rmtree] removed empty dir on final rmdir()")
            except Exception as e:
                print(f"[rmtree] empty dir removal failed (ok anyway): {e!r}")
        print("[rmtree] DONE")
        return True

    # Debug: show contents so you can see the culprit (.tmp/.ini/etc.)
    try:
        contents = list(path.iterdir())
    except Exception as e:
        contents = [f"<ls failed: {e!r}>"]
    print(f"[rmtree] NOT removed (ok={ok}, exists={path.exists()}, contents={contents})")
    return False


def quarantine_then_delete(run_dir: Path) -> tuple[Path, bool]:
    """
    Atomically move run_dir to a sibling .trash and then try to delete it.
    Returns (trash_dir, deleted_now).
    """
    run_dir = Path(run_dir)
    base = run_dir.parent
    trash_root = base / ".trash"
    trash_root.mkdir(parents=True, exist_ok=True)
    qdir = trash_root / f"{run_dir.name}__{uuid.uuid4().hex[:8]}"

    try:
        run_dir.replace(qdir)  # atomic-ish move; if this fails, dir didn't exist or was locked.
    except Exception as e:
        print(f"[quarantine] rename failed: {e!r}")
        return run_dir, False

    # Now try to delete the quarantined copy; even if it fails, we’ve freed up the original name.
    deleted = robust_rmtree(qdir)
    return qdir, deleted
