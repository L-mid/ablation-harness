import json
import math
from pathlib import Path


class MetricLogger:
    def __init__(self, path: str | Path, fmt: str = "jsonl"):
        self.path = Path(path)
        self.fmt = fmt
        self._csv_header_written = False
        self._fh = None
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __enter__(self):
        mode = "a"
        self._fh = self.path.open(mode, encoding="utf-8", newline="")
        return self

    def __exit__(self, *exc):
        if self._fh:
            self._fh.flush()
            self._fh.close()

    def log(self, step: int, **metrics):
        """metrics examples:
        {"train/loss": 2.30} or {"val/loss": 1.95, "val/acc": 0.41}"""
        if self.fmt == "jsonl":
            rec = {"_i": int(step), "out": {k: (float(v) if isinstance(v, (int, float)) and math.isfinite(v) else v) for k, v in metrics.items()}}
            json.dump(rec, self._fh, ensure_ascii=False, separators=(",", ":"))
            self._fh.write("\n")
            self._fh.flush()
        else:
            # Optional CSV mode if you prefer:
            import csv

            row = {"_i": int(step), **metrics}
            writer = csv.DictWriter(self._fh, fieldnames=list(row.keys()))
            if not self._csv_header_written:
                writer.writeheader()
                self._csv_header_written = True
            writer.writerow(row)
            self._fh.flush()
