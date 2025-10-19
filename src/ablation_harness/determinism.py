import warnings

import torch


def detect_flaky_ops(device: str | None = None) -> list[str]:
    "Warns on a few set nondeterministic ops, forces torch.use_deterministic_algorithims and fails with clear error/warning."

    msgs = []
    if torch.backends.cudnn.benchmark:
        msgs.append("cudnn.benchmark=True can cause nondeterminism")
    if not torch.backends.cudnn.deterministic:
        msgs.append("cudnn.deterministic=False (enable for deterministic kernels)")
    # try deterministic guard to force error on known nondet ops
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:
        msgs.append(f"determinism guard failed: {e!r}")
    if device and "cuda" in device and not torch.cuda.is_available():
        msgs.append("CUDA requested but not available")
    for m in msgs:
        warnings.warn(f"[flaky-detector] {m}")
    return msgs
