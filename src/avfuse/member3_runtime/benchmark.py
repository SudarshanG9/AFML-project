"""Real hardware timing helper; CUDA peak memory only when supported."""
import statistics
import time
from collections.abc import Callable
from typing import Any
import torch


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def benchmark_callable(
    fn: Callable[[], Any], device: torch.device,
    warmup: int = 10, repeats: int = 50,
) -> dict[str, float | int | None]:
    if repeats < 1 or warmup < 0:
        raise ValueError("Need repeats >= 1 and warmup >= 0")
    durations = []
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        sync_device(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(repeats):
            sync_device(device)
            start = time.perf_counter()
            fn()
            sync_device(device)
            durations.append((time.perf_counter() - start) * 1000.0)
    sorted_durations = sorted(durations)
    p95_idx = max(0, int(0.95 * (len(sorted_durations)-1)))
    peak = (int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None)
    return {
        "repeats": repeats,
        "p50_ms": float(statistics.median(durations)),
        "p95_ms": float(sorted_durations[p95_idx]),
        "peak_memory_bytes": peak,
    }
