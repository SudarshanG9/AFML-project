"""Trace every final measurement to exact inputs and environment."""
import hashlib
import platform
from pathlib import Path

import torch


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def environment_info() -> dict[str, str]:
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "cuda_runtime": str(torch.version.cuda),
        "mps_available": str(torch.backends.mps.is_available()),
        "cuda_available": str(torch.cuda.is_available()),
    }
