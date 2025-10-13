# ablation_harness/logger.py
from __future__ import annotations

import atexit
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence


def _ensure_dir(p: str) -> str:
    """Ensures there is a dir for a path."""
    os.makedirs(p, exist_ok=True)
    return p


def _as_dict(cfg: Any) -> Dict[str, Any]:
    """Try to convert cfg (dict/dataclass/OmegaConf) into a plain dict"""
    try:
        import dataclasses

        if dataclasses.is_dataclass(cfg):
            return dataclasses.asdict(cfg)  # type: ignore
    except Exception:
        pass
    try:
        # OmegaConf / omegaconf.DictConfig
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            return OmegaConf.to_container(cfg, resolve=True)  # type: ignore
    except Exception:
        pass
    if isinstance(cfg, dict):
        return cfg
    # Fallback: best effort
    try:
        return json.loads(json.dumps(cfg, default=lambda o: getattr(o, "__dict__", str(o))))
    except Exception:
        return {"cfg_repr": repr(cfg)}


# ---- Rank-zero guard (safe even if you're not in DDP yet) -------------------
def is_rank_zero() -> bool:
    return int(os.getenv("RANK", "0")) == 0


def rank_zero_only(fn):
    def wrapper(*args, **kwargs):
        if is_rank_zero():
            return fn(*args, **kwargs)

    return wrapper


# ---- Logger protocol -----------------------------------------------
class Logger(Protocol):
    """Logging Template abstract."""

    def on_run_start(self, cfg: Any) -> None: ...
    def on_epoch_start(self, epoch: int) -> None: ...
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None: ...
    def log_text(self, key: str, text: str, step: Optional[int] = None) -> None: ...
    def log_artifact(self, path: str, name: Optional[str] = None) -> None: ...
    def on_epoch_end(self, epoch: int) -> None: ...
    def on_run_end(self) -> None: ...


# ---- Null logger -----------------------------------------------------
class NullLogger:
    """No-op Logger."""

    def on_run_start(self, cfg):
        pass

    def on_epoch_start(self, epoch):
        pass

    def log_metrics(self, metrics, step=None):
        pass

    def log_text(self, key, text, step=None):
        pass

    def log_artifact(self, path, name=None):
        pass

    def on_epoch_end(self, epoch):
        pass

    def on_run_end(self):
        pass


# ---- TensorBoard -----------------------------------------------------
class TBLogger:
    """TensorBoard class."""

    def __init__(self, log_dir: str, flush_secs: int = 10):
        """Inits Tensorboard."""
        self.log_dir = _ensure_dir(log_dir)
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception as e:
            raise RuntimeError("TensorBoard not available: pip install tensorboard") from e
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=flush_secs)
        atexit.register(self.on_run_end)  # closes properly.

    @rank_zero_only
    def on_run_start(self, cfg: Any) -> None:
        """Logs on run start."""
        cfg_dict = _as_dict(cfg)
        self.writer.add_text("config/json", f"```json\n{json.dumps(cfg_dict, indent=2)}\n```", global_step=0)

    def on_epoch_start(self, epoch: int) -> None:
        """Runs on epoch start."""
        pass

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Logs metric ints and floats into TensorBoard."""
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(k, float(v), global_step=0 if step is None else step)

    @rank_zero_only
    def log_text(self, key: str, text: str, step: Optional[int] = None) -> None:
        """Logs text explicitly."""
        self.writer.add_text(key, text, global_step=0 if step is None else step)

    @rank_zero_only
    def log_artifact(self, path: str, name: Optional[str] = None) -> None:
        """
        TensorBoard helper to log artifacts, in a wandb style.
            Handles both text-like and images.
        """

        # TB doesn't have artifacts; attach as file text/image if needed.
        # Minimal: if it's a PNG, show it; if it's text-like, add as text.
        ext = os.path.splitext(path)[1].lower()
        if ext in {".png", ".jpg", ".jpeg"}:
            try:
                import PIL.Image as Image

                img = Image.open(path)
                import numpy as np

                # try to add image first
                self.writer.add_image(name or os.path.basename(path), np.array(img), global_step=0, dataformats="HWC")  # assume end channels
            except Exception:
                pass
        else:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()[:50000]
                self.writer.add_text(name or os.path.basename(path), f"```\n{content}\n```", global_step=0)
            except Exception:
                pass

    def on_epoch_end(self, epoch: int) -> None:
        """Runs on epoch end."""
        pass

    @rank_zero_only
    def on_run_end(self) -> None:
        """Flushes and closes TensorBoard correctly at the run's end."""
        try:
            self.writer.flush()
            self.writer.close()
        except Exception:
            pass


# ---- Weights & Biases -------------------------------------------------------
class WandbLogger:
    """Weights & Biases logging class."""

    def __init__(  # inits wandb.
        self,
        project: str,
        entity: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        notes: Optional[str] = None,
        mode: Optional[str] = None,  # "online" | "offline" | "disabled"
        dir: Optional[str] = None,
    ):
        self._wandb = None
        self._run = None
        self._init_kwargs = dict(project=project, entity=entity, name=run_name, tags=tags, notes=notes, dir=dir)
        if mode:
            os.environ.setdefault("WANDB_MODE", mode)  # "offline" is great as a default
        atexit.register(self.on_run_end)

    @rank_zero_only
    def on_run_start(self, cfg: Any) -> None:
        """Import wandb and init it on run's start."""
        try:
            import wandb  # type: ignore

            self._wandb = wandb
        except Exception as e:
            raise RuntimeError("wandb not available: pip install wandb") from e
        cfg_dict = _as_dict(cfg)
        self._run = self._wandb.init(**{k: v for k, v in self._init_kwargs.items() if v is not None})
        # Save config without overwriting user-changed values
        self._wandb.config.update(cfg_dict, allow_val_change=True)

    def on_epoch_start(self, epoch: int) -> None:
        """Run on epoch start."""
        pass

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log a dict to wandb. Keeps an explicit step column."""
        if self._wandb is None:
            return
        payload = dict(metrics)
        if step is not None:
            payload["trainer/step"] = step  # keep an explicit step column
        self._wandb.log(payload, step=step)

    @rank_zero_only
    def log_text(self, key: str, text: str, step: Optional[int] = None) -> None:
        """Explicitly log text to wandb. Makes a table per key."""
        if self._wandb is None:
            return
        table = self._wandb.Table(columns=["text"])
        table.add_data(text)
        self._wandb.log({key: table}, step=step)

    @rank_zero_only
    def log_artifact(self, path: str, name: Optional[str] = None) -> None:
        """Log artifacts (images/files) to wandb. Text here unknown."""
        if self._wandb is None:
            return
        art = self._wandb.Artifact(name or os.path.basename(path), type="file")
        art.add_file(path)
        self._wandb.log_artifact(art)

    def on_epoch_end(self, epoch: int) -> None:
        """Run on epoch end."""
        pass

    @rank_zero_only
    def on_run_end(self) -> None:
        """Finish wandb properly at the end of run."""
        try:
            if self._wandb is not None:
                self._wandb.finish()
        except Exception:
            pass


# ---- Composite --------------------------------------------------------------
class MultiLogger:
    """Handle BOTH TensorBoard and Wandb at once."""

    def __init__(self, loggers: Iterable[Logger]) -> None:
        """
        Init all loggers provided from cfg/initalization:
            Runs all sequentally with the same commands.
            (relies on the Protocall class being unified in all loggers)
        """
        self.loggers: list["Logger"] = list(loggers)

    def on_run_start(self, cfg):
        for logger in self.loggers:
            logger.on_run_start(cfg)

    def on_epoch_start(self, epoch):
        for logger in self.loggers:
            logger.on_epoch_start(epoch)

    def log_metrics(self, metrics, step=None):
        for logger in self.loggers:
            logger.log_metrics(metrics, step)

    def log_text(self, key, text, step=None):
        for logger in self.loggers:
            logger.log_text(key, text, step)

    def log_artifact(self, path, name=None):
        for logger in self.loggers:
            logger.log_artifact(path, name)

    def on_epoch_end(self, epoch):
        for logger in self.loggers:
            logger.on_epoch_end(epoch)

    def on_run_end(self):
        for logger in self.loggers:
            logger.on_run_end()


# ---- Factory ---------------------------------------------------------------
def build_logger(cfg: Any) -> Logger:
    """
    ## Builds loggers wanted from cfg provided.

    Expected cfg structure (minimal):

    cfg.logging.enable: bool
    cfg.logging.backends: List[str]  # any of {"wandb","tensorboard"}
    cfg.logging.dir: str             # base dir for local logs (e.g., runs/${run_id})
    cfg.run_id: str                  # used to name logs

    Optional:
      cfg.logging.wandb.project/entity/run_name/tags/notes/mode
      cfg.logging.tensorboard.flush_secs
    """

    # try to find the logging dict in cfg
    cfgd = _as_dict(cfg)
    lc = _as_dict(cfg).get("logging", {})

    # if can't find OR set to be disabled:
    if not lc or not lc.get("enable", True):
        return NullLogger()

    def _pick_run_id(d: dict) -> str:
        """Shuffles through cfg to find 'run_id'"""
        return lc.get("run_id") or d.get("run_id") or (d.get("baseline") or {}).get("run_id") or (d.get("experiment") or {}).get("run_id") or "run"

    run_id = str(_pick_run_id(cfgd))
    base_dir = lc.get("dir", "runs")

    # fetch backends, default to tensorboard (happens after no-op check)
    backends = lc.get("backends", ["tensorboard"])

    # the base run dir is from logger.dir's input ('runs/') + run_id choice
    run_dir = _ensure_dir(os.path.join(base_dir, run_id))
    loggers: List[Logger] = []  # available loggers native to here

    if "tensorboard" in backends:
        tb_dir = os.path.join(run_dir, "tb")  # explicit dir for tb
        loggers.append(TBLogger(log_dir=tb_dir, flush_secs=int(lc.get("tensorboard", {}).get("flush_secs", 10))))  # added to []

    if "wandb" in backends:
        wb = lc.get("wandb", {})  # wb = get wandb slice from cfg, explictly (under logging)
        loggers.append(  # append to []
            WandbLogger(  # fetched from wb cfg (meaning run_id etc explict to there)
                project=wb.get("project", "ablation-harness"),
                entity=wb.get("entity"),
                run_name=wb.get("run_name", _as_dict(cfg).get("run_id") or "none_designated"),
                tags=wb.get("tags"),
                notes=wb.get("notes"),
                mode=wb.get("mode", "online"),  # offline by default: sync later
                dir=run_dir,
            )
        )

    if not loggers:
        return NullLogger()
    return MultiLogger(loggers)
