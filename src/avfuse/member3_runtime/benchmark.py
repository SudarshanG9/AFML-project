"""Generic, device-aware callable benchmark engine (Segment B).

This module times a zero-argument callable. It knows nothing about models,
precision modes or research configurations: callers decide *what* to time and
*whether the result is scientifically meaningful*. Anything the callable does
(including quantize/dequantize conversion) is included in the measurement; the
engine never instruments or estimates conversion overhead on its own. To measure
conversion separately, benchmark a callable that performs only the conversion.

Measurement protocol (per ``benchmark_callable`` call)
-----------------------------------------------------
1. Validate arguments and resolve the device. An unavailable accelerator raises
   ``DeviceUnavailableError`` before the callable runs; there is no CPU fallback.
2. Run ``warmup`` untimed calls, then synchronize the device. Warm-up calls never
   contribute timing samples.
3. (CUDA) reset peak-memory statistics, so the reported peak covers only the timed
   repetitions (memory already resident, such as model weights and inputs, is
   included in the absolute peak and reported separately as a baseline).
4. For each of ``repeats`` repetitions: synchronize, read ``perf_counter_ns``, call,
   synchronize, read ``perf_counter_ns``. Samples are stored in milliseconds.
5. (CUDA) synchronize and read the peak-memory counters.

Percentiles
-----------
``compute_percentile`` supports the three methods named in the benchmark schema:
``linear`` (interpolate between ranks, same as ``numpy.percentile`` default),
``lower`` (``sorted[floor(q*(n-1))]``) and ``nearest_rank`` (``sorted[ceil(q*n)-1]``).
All are defined for any ``n >= 1``; with one sample every percentile equals it.
No smoothing, trimming or outlier removal is applied.

Memory honesty
--------------
* CUDA: ``torch.cuda.max_memory_allocated`` / ``max_memory_reserved``.
* CPU: not measured. ``tracemalloc`` does not see PyTorch tensor storage and
  process RSS is a process-wide high-water mark, not a per-benchmark peak.
* MPS: not measured. PyTorch exposes current-allocation queries but no
  resettable peak counter, so no defensible peak can be reported.
Unavailable memory is reported as ``None`` with ``memory_method == "not_measured"``,
never as zero.

Simulated INT8 runs in floating point; timings from this engine must never be
presented as INT8 hardware speedups.
"""
from __future__ import annotations

import contextlib
import math
import statistics
import time
from collections.abc import Callable, Sequence
from fractions import Fraction
from numbers import Integral, Real
from typing import Any

import torch

PERCENTILE_METHODS = ("linear", "lower", "nearest_rank")
DEFAULT_PERCENTILE_METHOD = "linear"

SYNC_CUDA = "cuda_synchronize"
SYNC_MPS = "mps_synchronize"
SYNC_NOT_REQUIRED_CPU = "not_required_cpu"
SYNC_DISABLED = "disabled"

MEMORY_CUDA = "cuda_max_memory_allocated"
MEMORY_NOT_MEASURED = "not_measured"

_CPU_MEMORY_NOTE = (
    "Peak memory is not measured on CPU: tracemalloc does not track PyTorch tensor "
    "storage and process RSS is a process-wide high-water mark, not a per-benchmark peak."
)
_MPS_MEMORY_NOTE = (
    "Peak memory is not measured on MPS: PyTorch exposes no resettable peak-memory "
    "counter for MPS, so no defensible peak value is available."
)


class BenchmarkError(Exception):
    """Base class for benchmark-engine errors that are not plain bad arguments."""


class DeviceUnavailableError(BenchmarkError, RuntimeError):
    """The requested accelerator is not usable. The engine never falls back to CPU."""


class MemoryMeasurementUnavailableError(BenchmarkError, RuntimeError):
    """Peak memory was required (``require_memory=True``) but cannot be measured."""


def _now_ns() -> int:
    return time.perf_counter_ns()


# ----------------------------------------------------------------- validation
def _validate_count(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer >= {minimum}, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


def _mps_is_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def resolve_device(device: torch.device | str) -> torch.device:
    """Validate ``device`` and return a normalized ``torch.device``.

    * ``cpu`` -> ``cpu``.
    * ``cuda`` (no index) -> ``cuda:<current device>`` so synchronization and memory
      counters always address the device that actually runs the work.
    * ``mps`` -> ``mps``.

    Raises ``TypeError`` for a wrong argument type, ``ValueError`` for malformed or
    unsupported device specifications and ``DeviceUnavailableError`` when the
    accelerator is not usable.
    """
    if isinstance(device, torch.device):
        dev = device
    elif isinstance(device, str):
        try:
            dev = torch.device(device)
        except (RuntimeError, ValueError) as exc:
            raise ValueError(
                f"Invalid device {device!r}. Use 'cpu', 'cuda', 'cuda:N' or 'mps'."
            ) from exc
    else:
        raise TypeError(
            f"device must be a torch.device or str, got {type(device).__name__}"
        )

    if dev.type == "cpu":
        if dev.index not in (None, 0):
            raise ValueError(f"Invalid CPU device {str(dev)!r}; use 'cpu'.")
        return torch.device("cpu")

    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise DeviceUnavailableError(
                f"Device {str(dev)!r} was requested but torch.cuda.is_available() is False "
                "(CPU-only PyTorch build, no NVIDIA driver, or no visible GPU). "
                "Install a CUDA-enabled PyTorch build, or request device 'cpu' explicitly. "
                "The benchmark engine never falls back to CPU silently."
            )
        count = torch.cuda.device_count()
        index = dev.index if dev.index is not None else torch.cuda.current_device()
        if not 0 <= index < count:
            raise DeviceUnavailableError(
                f"CUDA device index {index} is not available; "
                f"torch.cuda.device_count() == {count}. "
                f"Choose an index in [0, {count - 1}]."
            )
        return torch.device("cuda", index)

    if dev.type == "mps":
        if dev.index not in (None, 0):
            raise ValueError(f"Invalid MPS device {str(dev)!r}; use 'mps'.")
        if not _mps_is_available():
            built = getattr(getattr(torch.backends, "mps", None), "is_built", lambda: False)()
            reason = (
                "this PyTorch build has MPS support but no MPS device is usable "
                "(requires Apple silicon and macOS 12.3+)"
                if built
                else "this PyTorch build was not compiled with MPS support"
            )
            raise DeviceUnavailableError(
                f"Device 'mps' was requested but is unavailable: {reason}. "
                "Request device 'cpu' explicitly if that is what you intend. "
                "The benchmark engine never falls back to CPU silently."
            )
        if not hasattr(getattr(torch, "mps", None), "synchronize"):
            raise DeviceUnavailableError(
                "torch.mps.synchronize is missing in this PyTorch build; MPS timings "
                "would be asynchronous and therefore invalid. Upgrade PyTorch."
            )
        return torch.device("mps")

    raise ValueError(
        f"Unsupported device type {dev.type!r}. Supported: 'cpu', 'cuda', 'cuda:N', 'mps'."
    )


# ------------------------------------------------------------ synchronization
def sync_device(device: torch.device) -> None:
    """Block until all queued work on ``device`` has finished.

    CPU is synchronous, so this is a no-op and never touches CUDA/MPS.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
    elif device.type != "cpu":
        raise ValueError(f"Cannot synchronize unsupported device type {device.type!r}")


def synchronization_label(device: torch.device, enabled: bool = True) -> str:
    """Schema value (``measurement.synchronization``) describing per-call syncing."""
    if device.type == "cpu":
        return SYNC_NOT_REQUIRED_CPU
    if not enabled:
        return SYNC_DISABLED
    return SYNC_CUDA if device.type == "cuda" else SYNC_MPS


# ----------------------------------------------------------------- statistics
def compute_percentile(
    samples: Sequence[float], q: float, method: str = DEFAULT_PERCENTILE_METHOD
) -> float:
    """Percentile ``q`` (0-100) of ``samples`` using a documented ``method``.

    Index arithmetic is exact (``fractions.Fraction``) so ``lower`` and
    ``nearest_rank`` do not suffer float rounding such as ``0.95 * 20 -> 18.999...``.
    """
    if method not in PERCENTILE_METHODS:
        raise ValueError(f"Unknown percentile method {method!r}; use one of {PERCENTILE_METHODS}")
    if isinstance(q, bool) or not isinstance(q, Real) or not 0 <= q <= 100:
        raise ValueError(f"q must be a number in [0, 100], got {q!r}")
    values = sorted(float(s) for s in samples)
    n = len(values)
    if n == 0:
        raise ValueError("Cannot compute a percentile of zero samples")
    if any(not math.isfinite(v) for v in values):
        raise ValueError("Samples must be finite numbers")

    frac = Fraction(repr(float(q))) / 100
    if method == "nearest_rank":
        rank = math.ceil(frac * n)
        return values[max(rank, 1) - 1]
    pos = frac * (n - 1)
    lo = math.floor(pos)
    if method == "lower":
        return values[lo]
    hi = min(lo + 1, n - 1)
    weight = float(pos - lo)
    return values[lo] + (values[hi] - values[lo]) * weight


def summarize_latencies(
    samples_ms: Sequence[float], percentile_method: str = DEFAULT_PERCENTILE_METHOD
) -> dict[str, float]:
    """Summary statistics of raw millisecond samples (no smoothing or trimming)."""
    if len(samples_ms) == 0:
        raise ValueError("Cannot summarize zero timing samples")
    return {
        "mean_ms": math.fsum(samples_ms) / len(samples_ms),
        "median_ms": float(statistics.median(samples_ms)),
        "p50_ms": compute_percentile(samples_ms, 50, percentile_method),
        "p95_ms": compute_percentile(samples_ms, 95, percentile_method),
        "min_ms": float(min(samples_ms)),
        "max_ms": float(max(samples_ms)),
    }


# ------------------------------------------------------------------ benchmark
def benchmark_callable(
    fn: Callable[[], Any],
    device: torch.device | str,
    warmup: int = 10,
    repeats: int = 50,
    *,
    percentile_method: str = DEFAULT_PERCENTILE_METHOD,
    synchronize: bool = True,
    measure_memory: bool = True,
    require_memory: bool = False,
    inference_mode: bool = True,
) -> dict[str, Any]:
    """Time ``fn()`` on ``device`` and return latency (and, where supported, memory).

    ``fn`` must be a zero-argument callable whose setup (input creation, model
    construction, device transfers) is done by the caller beforehand, so only the
    work of interest is timed. Its return value is discarded. Exceptions raised by
    ``fn`` propagate unchanged.

    Args:
        warmup: untimed calls before measurement (``>= 0``; ``int``, not ``bool``).
        repeats: timed calls (``>= 1``; ``int``, not ``bool``).
        percentile_method: ``linear`` (default, matches the config template),
            ``lower`` or ``nearest_rank``. Recorded in the result.
        synchronize: synchronize the accelerator around every timed call. Setting
            this to ``False`` on CUDA/MPS measures only host-side launch time and is
            labelled ``"disabled"``. The post-warm-up and pre-memory-read
            synchronizations are always performed.
        measure_memory: attempt peak-memory measurement (CUDA only).
        require_memory: raise ``MemoryMeasurementUnavailableError`` instead of
            returning ``None`` when memory cannot be measured on ``device``.
        inference_mode: run under ``torch.inference_mode()``.

    Raises:
        TypeError: ``fn`` not callable, non-integer counts, or non-device ``device``.
        ValueError: out-of-range counts, unknown device or percentile names.
        DeviceUnavailableError: requested accelerator unusable (no CPU fallback).
        MemoryMeasurementUnavailableError: ``require_memory`` cannot be honoured.

    Returns:
        A dict. Legacy keys (``repeats``, ``p50_ms``, ``p95_ms``,
        ``peak_memory_bytes``) are preserved. Additional keys: ``device``,
        ``warmup``, ``timing_samples_ms``, ``mean_ms``, ``median_ms``, ``min_ms``,
        ``max_ms``, ``percentile_method``, ``synchronization``,
        ``peak_memory_reserved_bytes``, ``memory_baseline_bytes``, ``memory_method``,
        ``memory_note``. All times are milliseconds; memory values are bytes or
        ``None`` when unavailable (never zero as a placeholder).
    """
    if not callable(fn):
        raise TypeError(f"fn must be callable, got {type(fn).__name__}")
    warmup = _validate_count("warmup", warmup, 0)
    repeats = _validate_count("repeats", repeats, 1)
    if percentile_method not in PERCENTILE_METHODS:
        raise ValueError(
            f"Unknown percentile method {percentile_method!r}; use one of {PERCENTILE_METHODS}"
        )
    if require_memory and not measure_memory:
        raise ValueError("require_memory=True conflicts with measure_memory=False")

    dev = resolve_device(device)

    memory_note: str | None
    if not measure_memory:
        memory_note = "Peak memory measurement was disabled by the caller."
    elif dev.type == "cuda":
        memory_note = None
    elif dev.type == "mps":
        memory_note = _MPS_MEMORY_NOTE
    else:
        memory_note = _CPU_MEMORY_NOTE
    track_memory = measure_memory and dev.type == "cuda"
    if require_memory and not track_memory:
        raise MemoryMeasurementUnavailableError(
            f"Peak memory cannot be measured on device {str(dev)!r}. {memory_note} "
            "Use a CUDA device or pass require_memory=False."
        )

    sync_each = synchronize
    samples_ms: list[float] = []
    peak_allocated: int | None = None
    peak_reserved: int | None = None
    baseline: int | None = None

    context = torch.inference_mode() if inference_mode else contextlib.nullcontext()
    with context:
        for _ in range(warmup):
            fn()
        sync_device(dev)  # warm-up work must not leak into the first timed sample
        if track_memory:
            baseline = int(torch.cuda.memory_allocated(dev))
            torch.cuda.reset_peak_memory_stats(dev)
        for _ in range(repeats):
            if sync_each:
                sync_device(dev)
            start = _now_ns()
            fn()
            if sync_each:
                sync_device(dev)
            stop = _now_ns()
            samples_ms.append((stop - start) / 1e6)
        sync_device(dev)
        if track_memory:
            peak_allocated = int(torch.cuda.max_memory_allocated(dev))
            peak_reserved = int(torch.cuda.max_memory_reserved(dev))

    result: dict[str, Any] = {
        "repeats": repeats,
        "p50_ms": 0.0,
        "p95_ms": 0.0,
        "peak_memory_bytes": peak_allocated,
        "device": str(dev),
        "warmup": warmup,
        "timing_samples_ms": samples_ms,
    }
    result.update(summarize_latencies(samples_ms, percentile_method))
    result.update(
        {
            "percentile_method": percentile_method,
            "synchronization": synchronization_label(dev, sync_each),
            "peak_memory_reserved_bytes": peak_reserved,
            "memory_baseline_bytes": baseline,
            "memory_method": MEMORY_CUDA if track_memory else MEMORY_NOT_MEASURED,
            "memory_note": memory_note,
        }
    )
    return result
