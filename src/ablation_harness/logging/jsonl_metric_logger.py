import json
import math
from pathlib import Path

import numpy as np

try:
    import torch
except Exception:
    torch = None


class MetricLogger:
    def __init__(self, path: str | Path, fmt: str = "jsonl"):
        self.path = Path(path)
        self.fmt = fmt
        self._csv_header_written = False
        self._fh = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._pending_step: int | None = None
        self._pending_out: dict = {}

    def __enter__(self):
        mode = "a"
        self._fh = self.path.open(mode, encoding="utf-8", newline="")
        return self

    def __exit__(self, *exc):
        self.flush()
        if self._fh:
            self._fh.flush()
            self._fh.close()

    def _coerce(self, v):
        # Keep this conservative: JSON-serializable + stable.
        if torch is not None and isinstance(v, torch.Tensor):
            if v.ndim == 0:
                v = v.item()
            else:
                v = v.detach().cpu().tolist()
        if isinstance(v, np.ndarray):
            v = v.tolist()

        if isinstance(v, (int, float)):
            if math.isfinite(v):
                return float(v)
            return None  # avoid NaN/inf in logs
        return v

    def flush(self):
        if self._fh is None:
            return
        if self._pending_step is None:
            return

        if self.fmt == "jsonl":
            rec = {"_i": int(self._pending_step), "out": dict(self._pending_out)}
            json.dump(rec, self._fh, ensure_ascii=False, separators=(",", ":"))
            self._fh.write("\n")
            self._fh.flush()
        else:
            import csv

            row = {"_i": int(self._pending_step), **self._pending_out}
            writer = csv.DictWriter(self._fh, fieldnames=list(row.keys()))
            if not self._csv_header_written:
                writer.writeheader()
                self._csv_header_written = True
            writer.writerow(row)
            self._fh.flush()

        # clear
        self._pending_step = None
        self._pending_out = {}

    def log(self, step: int, **metrics):
        """
        Coalesces repeated calls at the same step into ONE jsonl line.
        If the step changes, flush the previous step first.
        """
        step = int(step)
        if self._pending_step is None:
            self._pending_step = step
        elif step != self._pending_step:
            self.flush()
            self._pending_step = step

        # merge (last write wins per-key, but still same step line)
        for k, v in metrics.items():
            self._pending_out[k] = self._coerce(v)
