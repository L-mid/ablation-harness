import os
import random
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: int, single_thread: bool = True, deterministic: bool = True, cudnn_benchmark: Optional[bool] = None) -> torch.Generator:
    """
    Seed all RNGs and (optionally) force deterministic algorithms.
    Call this ONCE at the very start of a run.
    """
    # Python / NumPy
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if single_thread:
        torch.set_num_threads(1)
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    if deterministic:
        # Make PyTorch pick deterministic kernals
        torch.use_deterministic_algorithms(True)
        # cuDNN knobs (harmless on CPU)
        torch.backends.cudnn.deterministic = True
        # Benchmark should be off for determinisum you explicitly want speed
        torch.backends.cudnn.benchmark = False if cudnn_benchmark is None else cudnn_benchmark

        # For CUDA matmul determinisum on some stacks; set **before** CUDA context creation ideally
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        torch.backends.cudnn.deterministic = False
        # Let user choose whether to enable autotuner when not strict-deterministic
        if cudnn_benchmark is not None:
            torch.backends.cudnn.benchmark = cudnn_benchmark

    g = torch.Generator()
    g.manual_seed(seed)

    return g


def seed_worker(base_seed: int):
    """Use in DataLoader so each worker has a distinct, reproducible seed."""

    def _init(worker_id: int):
        worker_seed = (base_seed + worker_id) % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    worker_seed = _init(base_seed)
    return worker_seed


# unused for now; makes a generator (e.g. for dataloaders)
def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g
