import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

DEFAULT_OUT = Path("C:/ML/runs") if os.name == "nt" else Path.home() / "ml_runs"


# ---- Subsystems ----


@dataclass
class LossConfig:
    weighting: str = "constant"  # "constant" or "minsnr"
    minsnr_gamma: float = 5.0


@dataclass
class CurvatureConfig:  # Hutchinson trace probe config
    enabled: bool = True
    method: str = "hutchinson"
    probes: int = 8
    log_prefix: str = "curvature/hutch"


@dataclass
class TrainCfg:
    total_steps: int = 10_000
    grad_clip: float = 1.0
    amp: bool = False


@dataclass
class EvalGridCfg:
    enabled: bool = True
    every: int = 1_000
    sampler: Literal["ddpm", "ddim"] = "ddim"
    nfe: int = 20
    n_samples: int = 36
    batch_size: int = 64
    save_images: bool = True
    sample_seed: int = 0  # fixed seed for apples-to-apples grids


@dataclass
class EvalKidCfg:
    enabled: bool = True
    every: int = 4_000
    sampler: Literal["ddpm", "ddim"] = "ddim"
    nfe: int = 20
    n_samples: int = 1_024
    repeats: int = 3  # average across repeats for stability
    batch_size: int = 64
    feature_cache: Optional[str] = None  # optional path to cached real features


@dataclass
class EvalFidMilestoneCfg:
    enabled: bool = True
    every: int = 20_000
    sampler: Literal["ddpm", "ddim"] = "ddpm"
    nfe: int = 50
    n_samples: int = 5_000
    batch_size: int = 64
    fid_stats: Optional[str] = None  # e.g., "cifar10_inception_stats.npz"
    run_if_kid_improved_pct: float = 3.0  # gate heavy eval on KID improvement


@dataclass
class EvalFinalCfg:
    enabled: bool = True
    at_end: bool = True  # run once at training end
    sampler: Literal["ddpm", "ddim"] = "ddpm"
    nfe: int = 50
    n_samples: int = 50_000  # report-grade
    batch_size: int = 64
    fid_stats: Optional[str] = None
    save_images: bool = False


@dataclass
class EvalReconCfg:
    """
    Val reconstruction diagnositic for diffusion:
        sample t, form x_t from x0, predict eps, reconstruct x0_hat, log metrics + images.
    """

    enabled: bool = False
    every: int = 4

    # how much val to average over each time run recon.
    n_batches: int = 4

    # timestep selection
    t_mode: Literal["uniform", "fixed"] = "uniform"
    t_values: Optional[List[int]] = None  # used if t_mode="fixed"

    # metrics
    metrics: List[Literal["mse", "psnr", "l1"]] = field(default_factory=lambda: ["mse", "psnr"])
    max_val: float = 2.0  # image dynamic range for PSNR if x in [-1, 1]

    # how many images to visualize (pairs: x0 vs x0_hat)
    n_images: int = 16
    save_images: bool = True

    # logging
    log_prefix: str = "val/recon"


# --- Top-level eval config (keeps legacy fields, adds structured tasks) ---


@dataclass
class EvalsCfg:
    # top-level toggles under `eval:` (your YAML has only `quick` today)
    quick: bool = False

    # structured eval tasks (all optional so you can omit them in YAML)
    grid: EvalGridCfg = field(default_factory=EvalGridCfg)
    kid: EvalKidCfg = field(default_factory=EvalKidCfg)
    fid_milestone: EvalFidMilestoneCfg = field(default_factory=EvalFidMilestoneCfg)
    final: EvalFinalCfg = field(default_factory=EvalFinalCfg)
    recon: EvalReconCfg = field(default_factory=EvalReconCfg)


@dataclass
class DiffusionCfg:
    enabled: bool = False  # sets rt.task = "diffusion" when True
    beta_schedule: Literal["linear", "cosine", "learned"] = "linear"


@dataclass
class DataCfg:
    dataset: Literal["moons", "cifar10"] = "moons"
    subset: Optional[int] = None
    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = False
    shuffle: bool = True


@dataclass
class ModelCfg:
    name: Literal["mlp", "tinycnn"] = "mlp"
    in_channels: int = 3
    out_channels: int = 3
    base_channels: int = 32
    channel_mults: List[int] = field(default_factory=lambda: [1, 2, 2, 2])
    num_res_blocks: int = 2
    dropout: float = 0.0
    time_embedding: int = 512
    gn_groups: int = 32

    # legacy, for classification
    hidden: int = 64


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
    decay: float = 0.999  # 0.999–0.9999 are common for vision; smaller for short runs
    device: str = None  # keep EMA on CPU to save VRAM (optional)
    pin_mem: bool = False  # only if device is CUDA and you want pinned mem copies
    include_buffers: bool = False  # usually False; BN buffers are already moving avgs
    warmup_steps: int = 0


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
    total_steps: int = 100
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
    train: TrainCfg = field(default_factory=TrainCfg)
    eval: EvalsCfg = field(default_factory=EvalsCfg)
    loss: LossConfig = field(default_factory=LossConfig)
    curvature: CurvatureConfig = field(default_factory=CurvatureConfig)
    diffusion: DiffusionCfg = field(default_factory=DiffusionCfg)


# ---- Runtime (for compression in trainer if wanted) ----


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
    pin_mem: bool
    include_buffers: bool
    warmup_steps: int

    task: str | None = None  # "diffusion" or None/classification
    total_steps: int = 10_000  # used for steps-based loops
    eval_every: int = 5_000  # evaluate every N train steps
    grad_clip: float = 1.0
    amp: bool = False

    beta_schedule: str = "linear"  # "linear" | "cosine"
    eval_sampler: str = "ddpm"
    eval_nfe: int = 50
    eval_n_samples: int = 10_000
    fid_stats: str | None = None

    shuffle: bool = True
