import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

DEFAULT_OUT = Path("C:/ML/runs") if os.name == "nt" else Path.home() / "ml_runs"


# ---- Subsystems ----


@dataclass
class DataCfg:
    dataset: Literal["moons", "cifar10"] = "moons"
    subset: Optional[int] = None
    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = False


@dataclass
class ModelCfg:
    name: Literal["mlp", "tinycnn"] = "mlp"
    hidden: int = 64
    dropout: float = 0.0


@dataclass
class OptimCfg:
    optimizer: Literal["adam", "sgd"] = "adam"
    lr: float = 1e-3
    wd: float = 0.0
    momentum: float = 0.9


@dataclass
class SchedCfg:
    name: Literal["cosine", "none"] = "cosine"


@dataclass
class EMACfg:
    enabled: bool = False
    decay: float = 0.9999


@dataclass
class SpectralDiagCfg:  # currently unused!
    enabled: bool = False
    every_n_epochs: int = 1
    topk: int = 5


@dataclass
class WandbCfg:
    project: str = "ablation-harness"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    mode: Literal["online", "offline", "disabled"] = "disabled"


@dataclass
class TensorBoardCfg:
    flush_secs: int = 10


@dataclass
class LoggingCfg:
    enable: bool = True
    backends: List[Literal["tensorboard", "wandb"]] = field(default_factory=lambda: ["tensorboard"])
    wandb: WandbCfg = field(default_factory=WandbCfg)
    tensorboard: TensorBoardCfg = field(default_factory=TensorBoardCfg)
    log_every_n_steps: int = 10


# ---- Spec (what users write / planner merges) ----


@dataclass
class StudySpec:
    study_name: str = "study"
    seed: int = 0
    epochs: int = 4
    out_dir: str = str(DEFAULT_OUT)  # absolute recommended
    run_id: str = "run"  # filled by planner; can be ignored in YAML

    device: Optional[str] = None  # "cpu"/"cuda"/None(auto)
    deterministic: bool = True
    clean_run: bool = False

    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    optim: OptimCfg = field(default_factory=OptimCfg)
    sched: SchedCfg = field(default_factory=SchedCfg)
    ema: EMACfg = field(default_factory=EMACfg)
    logging: LoggingCfg = field(default_factory=LoggingCfg)
    spectral_diag: SpectralDiagCfg = field(default_factory=SpectralDiagCfg)


# ---- Runtime (what trainer actually needs) ----


@dataclass
class RuntimeConfig:
    # minimal, flat, fast to access
    study_name: str
    run_id: str
    out_dir: str
    seed: int
    epochs: int
    device: str
    deterministic: bool
    clean_run: bool

    # hot-path fields needed to build loaders/model/optim
    dataset: str
    subset: Optional[int]
    batch_size: int
    num_workers: int
    pin_memory: bool

    model_name: str
    hidden: int
    dropout: float

    opt_name: str
    lr: float
    wd: float
    momentum: float

    sched_name: str
    ema_enabled: bool
    ema_decay: float

    task: str | None = None  # "diffusion" or None/classification
    total_steps: int = 10_000  # used for steps-based loops
    eval_every: int = 5_000  # evaluate every N train steps
    grad_clip: float = 1.0
    amp: bool = False

    beta_schedule: str = "linear"  # "linear" | "cosine" | "learned" (later)
    eval_sampler: str = "ddpm"
    eval_nfe: int = 50
    eval_n_samples: int = 10_000
    fid_stats: str | None = None

    data_shuffle: bool = True
