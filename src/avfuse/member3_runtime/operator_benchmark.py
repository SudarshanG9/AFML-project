"""Operator-level fusion-attention microbenchmark (Segment C, Day 4).

This module times Member 1's ``fusion_attention()`` **in isolation**. All timing,
synchronization, percentile and CUDA-memory logic lives in Segment B's
``benchmark_callable``; nothing is re-implemented here.

What is (and is not) inside the timed region
--------------------------------------------
The timed callable is exactly::

    fusion_attention(q, k, v, mode=mode, backend=api_backend, attn_mask=attn_mask)

Tensor creation, validation, device transfers, dataset loading, media decoding and
encoder execution all happen **before** timing starts. The fake-quantization work that
``fusion_attention`` performs inside a non-``reference`` precision mode *is* part of the
measured operator latency (it is not separated out here). Simulated INT8 runs in
floating point: these timings are **not** INT8 kernel timings and say nothing about INT8
hardware acceleration. No conversion-overhead number is produced by this module.

Backends
--------
Public labels map to Member 1's API names:

* ``"sdpa"``         -> ``backend="sdpa"`` (the performance path).
* ``"dense_oracle"`` -> ``backend="dense"`` (a correctness oracle that materializes the
  full score matrix; not a substitute for SDPA performance).

There is no custom tiled kernel; ``"custom_tiled"`` is rejected. SDPA may internally
choose a flash, memory-efficient or math kernel; this module does not record which one.

Synthetic versus real inputs
----------------------------
Every result carries ``input_kind`` (``"synthetic"`` or ``"real"``). Synthetic tensors
are for smoke tests only and are never research evidence. ``"real"`` requires a
``shape_source`` string saying where the shapes came from (for example the Member 1 Q/K/V
shape-report handoff). The declaration is made by the caller; this module cannot verify it.

Score-matrix estimate (H1)
--------------------------
``score_matrix_bytes`` is ``B * H * Nq * Nk * bytes_per_element`` from Member 1's
``score_matrix_bytes(q, k)``. It is a **naive dense score-matrix size**. It is NOT a
measurement of what fused SDPA allocates (fused kernels avoid materializing it).

Results are in-memory dictionaries, not benchmark schema v0.2 records; artifact writing
belongs to Segment D.
"""
from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import torch
from torch import Tensor

from avfuse.member1_fusion.attention import fusion_attention, score_matrix_bytes
from avfuse.member3_runtime.benchmark import (
    DEFAULT_PERCENTILE_METHOD,
    benchmark_callable,
    resolve_device,
)
from avfuse.shared.contracts import PRECISION_MODES

SCOPE_OPERATOR = "operator"

INPUT_SYNTHETIC = "synthetic"
INPUT_REAL = "real"
INPUT_KINDS = (INPUT_SYNTHETIC, INPUT_REAL)

BACKEND_SDPA = "sdpa"
BACKEND_DENSE_ORACLE = "dense_oracle"
#: Public benchmark label -> ``fusion_attention(backend=...)`` name (Member 1's API).
BACKEND_LABEL_TO_API: dict[str, str] = {
    BACKEND_SDPA: "sdpa",
    BACKEND_DENSE_ORACLE: "dense",
}
SUPPORTED_BACKENDS = tuple(BACKEND_LABEL_TO_API)

SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)

#: Dense oracle safety limit on the *score matrix* size. The dense path also holds the
#: softmax weights and (with a mask) a masked copy, so peak use is a few times this value.
DEFAULT_DENSE_MAX_SCORE_BYTES = 128 * 1024 * 1024

EVIDENCE_SYNTHETIC = "synthetic_not_research_evidence"
EVIDENCE_REAL_DECLARED = "real_shapes_caller_declared"

SCORE_ESTIMATE_FORMULA = "batch_size * num_heads * q_tokens * k_tokens * bytes_per_element"
SCORE_ESTIMATE_NOTE = (
    "Naive dense score-matrix size estimate (B*H*Nq*Nk*bytes_per_element). It is NOT "
    "the memory allocated by fused SDPA, which does not necessarily materialize the "
    "full score matrix."
)
LATENCY_SCOPE_NOTE = (
    "Operator-only latency: fusion_attention() on pre-built tensors. Excludes data "
    "loading, media decoding, encoders, detection head, file I/O and host/device "
    "transfers. Not cached-feature model latency and not raw-video end-to-end latency."
)
FAKE_QUANT_NOTE = (
    "Fake-quantization performed inside fusion_attention() for this precision mode is "
    "included in the measured latency; it is NOT an isolated conversion-overhead "
    "measurement. Simulated INT8 keeps floating-point tensors: this is not real INT8 "
    "storage, not INT8 kernel execution and not evidence of INT8 hardware speedup."
)
REFERENCE_PRECISION_NOTE = (
    "reference mode performs no fake quantization inside fusion_attention()."
)
REAL_INPUT_NOTE = (
    "input_kind='real' is declared by the caller via shape_source and is not verified by "
    "this benchmark."
)
SDPA_NOTE = (
    "SDPA kernel selection (flash / memory-efficient / math) is not recorded by this "
    "benchmark."
)
DENSE_NOTE = (
    "dense_oracle is a correctness oracle that materializes the score matrix; its "
    "timings are not a substitute for SDPA performance."
)


class OperatorInputError(ValueError):
    """Q/K/V/mask/device/dtype arguments are invalid for an operator benchmark."""


class DenseOracleLimitError(OperatorInputError):
    """The dense oracle was requested for a workload above the safety limit."""


# ------------------------------------------------------------------ workloads
def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer >= 1, got {value!r}")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return int(value)


@dataclass(frozen=True)
class OperatorWorkload:
    """Explicit attention workload dimensions. Nothing here has defaults on purpose:
    callers must state every dimension (real values come from Member 1)."""

    batch_size: int
    num_heads: int
    q_tokens: int
    k_tokens: int
    head_dim: int
    v_dim: int | None = None  # defaults to head_dim when None
    name: str | None = None

    def __post_init__(self) -> None:
        for field in ("batch_size", "num_heads", "q_tokens", "k_tokens", "head_dim"):
            _positive_int(field, getattr(self, field))
        if self.v_dim is not None:
            _positive_int("v_dim", self.v_dim)

    @property
    def resolved_v_dim(self) -> int:
        return self.head_dim if self.v_dim is None else self.v_dim


@dataclass(frozen=True)
class OperatorInputs:
    """Pre-built tensors for one workload (already on the benchmark device)."""

    q: Tensor
    k: Tensor
    v: Tensor
    attn_mask: Tensor | None = None
    name: str | None = None


def build_batch_head_sweep(
    *,
    batch_sizes: Sequence[int],
    head_counts: Sequence[int],
    q_tokens: int,
    k_tokens: int,
    head_dim: int,
    v_dim: int | None = None,
) -> list[OperatorWorkload]:
    """Vary batch size and head count over fixed token/head dimensions.

    All five dimensions must be passed explicitly; none is defaulted or invented.
    Order is batch-major (for each batch size, every head count), so it is deterministic.
    """
    if not len(batch_sizes) or not len(head_counts):
        raise ValueError("batch_sizes and head_counts must each contain at least one value")
    return [
        OperatorWorkload(
            batch_size=b,
            num_heads=h,
            q_tokens=q_tokens,
            k_tokens=k_tokens,
            head_dim=head_dim,
            v_dim=v_dim,
            name=f"B{b}_H{h}_Nq{q_tokens}_Nk{k_tokens}_D{head_dim}",
        )
        for b, h in itertools.product(batch_sizes, head_counts)
    ]


def make_synthetic_inputs(
    workload: OperatorWorkload,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> OperatorInputs:
    """Deterministic **synthetic** Q/K/V. Smoke-test use only; never research evidence.

    Tensors are drawn on CPU with a private generator and then moved to ``device``.
    Call this *before* benchmarking: creation and transfer are never timed.
    """
    if dtype not in SUPPORTED_DTYPES:
        raise OperatorInputError(f"dtype {dtype} unsupported; use one of {SUPPORTED_DTYPES}")
    dev = resolve_device(device)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    w = workload

    def draw(*shape: int) -> Tensor:
        return torch.randn(*shape, generator=gen, dtype=torch.float32).to(dtype).to(dev)

    return OperatorInputs(
        q=draw(w.batch_size, w.num_heads, w.q_tokens, w.head_dim),
        k=draw(w.batch_size, w.num_heads, w.k_tokens, w.head_dim),
        v=draw(w.batch_size, w.num_heads, w.k_tokens, w.resolved_v_dim),
        name=w.name,
    )


# ----------------------------------------------------------------- validation
def _device_matches(tensor: Tensor, expected: torch.device) -> bool:
    if tensor.device.type != expected.type:
        return False
    return expected.index is None or tensor.device.index == expected.index


def validate_operator_inputs(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Tensor | None,
    device: torch.device,
) -> None:
    """Raise ``OperatorInputError`` (with an actionable message) if inputs are unusable.

    ``device`` must already be resolved. Tensors are never moved implicitly.
    """
    for name, t in (("q", q), ("k", k), ("v", v)):
        if not isinstance(t, Tensor):
            raise OperatorInputError(f"{name} must be a torch.Tensor, got {type(t).__name__}")
        if t.dim() != 4:
            raise OperatorInputError(
                f"{name} must be rank-4 [batch, heads, tokens, dim]; got shape {tuple(t.shape)} "
                f"(rank {t.dim()})."
            )
        if any(s < 1 for s in t.shape):
            raise OperatorInputError(f"{name} has an empty dimension: shape {tuple(t.shape)}.")
        if t.dtype not in SUPPORTED_DTYPES:
            raise OperatorInputError(
                f"{name} has unsupported dtype {t.dtype}; supported: "
                f"{[str(d) for d in SUPPORTED_DTYPES]}."
            )
        if not _device_matches(t, device):
            raise OperatorInputError(
                f"{name} is on device {t.device} but the benchmark device is {device}. "
                "Tensors are never moved implicitly: move them before calling (outside "
                "any timed region)."
            )
    if not (q.dtype == k.dtype == v.dtype):
        raise OperatorInputError(
            f"q, k and v must share one dtype; got q={q.dtype}, k={k.dtype}, v={v.dtype}."
        )
    if not (q.shape[0] == k.shape[0] == v.shape[0]):
        raise OperatorInputError(
            f"Batch sizes differ: q={q.shape[0]}, k={k.shape[0]}, v={v.shape[0]}."
        )
    if not (q.shape[1] == k.shape[1] == v.shape[1]):
        raise OperatorInputError(
            f"Head counts differ: q={q.shape[1]}, k={k.shape[1]}, v={v.shape[1]}."
        )
    if q.shape[3] != k.shape[3]:
        raise OperatorInputError(
            f"Q and K head dimensions differ: q={q.shape[3]}, k={k.shape[3]}."
        )
    if k.shape[2] != v.shape[2]:
        raise OperatorInputError(
            f"K and V sequence lengths differ: k={k.shape[2]}, v={v.shape[2]}."
        )

    if attn_mask is None:
        return
    if not isinstance(attn_mask, Tensor):
        raise OperatorInputError(f"attn_mask must be a Tensor, got {type(attn_mask).__name__}")
    if not _device_matches(attn_mask, device):
        raise OperatorInputError(
            f"attn_mask is on device {attn_mask.device} but the benchmark device is {device}."
        )
    if attn_mask.dtype != torch.bool and attn_mask.dtype != q.dtype:
        raise OperatorInputError(
            f"attn_mask dtype must be torch.bool (True = attend) or match the Q/K/V dtype "
            f"({q.dtype}); got {attn_mask.dtype}."
        )
    score_shape = (q.shape[0], q.shape[1], q.shape[2], k.shape[2])
    if attn_mask.dim() < 2 or attn_mask.dim() > 4:
        raise OperatorInputError(
            f"attn_mask must have rank 2-4 and broadcast to {score_shape}; got shape "
            f"{tuple(attn_mask.shape)}."
        )
    try:
        broadcast = tuple(torch.broadcast_shapes(tuple(attn_mask.shape), score_shape))
    except RuntimeError:
        broadcast = None
    if broadcast != score_shape:
        raise OperatorInputError(
            f"attn_mask shape {tuple(attn_mask.shape)} is not broadcastable to the "
            f"attention score shape [batch, heads, Nq, Nk] = {score_shape}."
        )


def _validate_dense_limit(q: Tensor, k: Tensor, backend: str, limit: int) -> int:
    estimate = score_matrix_bytes(q, k)
    if backend == BACKEND_DENSE_ORACLE and estimate > limit:
        raise DenseOracleLimitError(
            f"dense_oracle refused: the dense score matrix would be {estimate} bytes "
            f"({estimate / 2**20:.1f} MiB), above the safety limit of {limit} bytes "
            f"({limit / 2**20:.1f} MiB). Use backend='sdpa' for this workload, shrink the "
            "workload, or explicitly raise dense_max_score_bytes if you accept the "
            "memory risk (the dense path needs several times the score-matrix size)."
        )
    return estimate


def _validate_backend_mode(backend: str, mode: str) -> None:
    if backend not in BACKEND_LABEL_TO_API:
        extra = " (no custom tiled kernel exists)" if backend == "custom_tiled" else ""
        raise ValueError(
            f"Unsupported operator backend {backend!r}{extra}; use one of {SUPPORTED_BACKENDS}."
        )
    if mode not in PRECISION_MODES:
        raise ValueError(f"Unknown precision mode {mode!r}; use one of {PRECISION_MODES}.")


def _validate_input_kind(input_kind: str, shape_source: str | None) -> None:
    if input_kind not in INPUT_KINDS:
        raise ValueError(f"input_kind must be one of {INPUT_KINDS}, got {input_kind!r}.")
    if input_kind == INPUT_REAL and not (isinstance(shape_source, str) and shape_source.strip()):
        raise ValueError(
            "input_kind='real' requires a non-empty shape_source describing where the "
            "tensor shapes came from (e.g. the Member 1 Q/K/V shape report)."
        )


def _validate_dense_limit_value(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"dense_max_score_bytes must be a positive integer, got {value!r}")
    return int(value)


# ------------------------------------------------------------- score estimate
def score_matrix_estimate(q: Tensor, k: Tensor) -> dict[str, Any]:
    """H1 score-matrix estimate for the actual ``q``/``k`` tensors (an estimate only)."""
    batch, heads, n_q, _ = q.shape
    n_k = k.shape[-2]
    bytes_per_element = q.element_size()
    expected = batch * heads * n_q * n_k * bytes_per_element
    reported = score_matrix_bytes(q, k)  # Member 1's function is the source of truth
    if reported != expected:
        raise RuntimeError(
            f"Member 1 score_matrix_bytes()={reported} disagrees with "
            f"{SCORE_ESTIMATE_FORMULA}={expected}; the interface has changed - report to Member 1."
        )
    return {
        "batch_size": int(batch),
        "num_heads": int(heads),
        "q_tokens": int(n_q),
        "k_tokens": int(n_k),
        "dtype": str(q.dtype),
        "bytes_per_element": int(bytes_per_element),
        "formula": SCORE_ESTIMATE_FORMULA,
        "bytes": int(reported),
        "mib": reported / 2**20,
        "is_estimate_only": True,
        "note": SCORE_ESTIMATE_NOTE,
    }


# ------------------------------------------------------------------ benchmark
def make_attention_callable(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    mode: str,
    backend: str,
    attn_mask: Tensor | None = None,
) -> Callable[[], Tensor]:
    """The zero-argument callable that is timed: ``fusion_attention`` and nothing else."""
    api_backend = BACKEND_LABEL_TO_API[backend]

    def run() -> Tensor:
        return fusion_attention(q, k, v, mode=mode, backend=api_backend, attn_mask=attn_mask)

    return run


def benchmark_attention_operator(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    mode: str,
    backend: str,
    device: torch.device | str,
    input_kind: str,
    shape_source: str | None = None,
    attn_mask: Tensor | None = None,
    warmup: int = 10,
    repeats: int = 50,
    percentile_method: str = DEFAULT_PERCENTILE_METHOD,
    measure_memory: bool = True,
    require_memory: bool = False,
    dense_max_score_bytes: int = DEFAULT_DENSE_MAX_SCORE_BYTES,
    workload_name: str | None = None,
) -> dict[str, Any]:
    """Benchmark ``fusion_attention`` in isolation for one workload/backend/mode.

    ``input_kind`` is mandatory (``"synthetic"`` or ``"real"``); ``"real"`` also needs
    ``shape_source``. Inputs must already be on ``device``. Validation and the dense
    safety check run before any warm-up or timing.

    Returns an in-memory result dict (see module docstring): workload metadata, the
    score-matrix estimate, caveat notes, and Segment B's measurement fields merged in
    (``timing_samples_ms``, ``p50_ms``, ``p95_ms``, ``percentile_method``,
    ``synchronization``, peak-memory fields, ...). It is not a schema v0.2 record.
    """
    _validate_input_kind(input_kind, shape_source)
    _validate_backend_mode(backend, mode)
    limit = _validate_dense_limit_value(dense_max_score_bytes)
    dev = resolve_device(device)  # raises DeviceUnavailableError; never falls back
    validate_operator_inputs(q, k, v, attn_mask, dev)
    _validate_dense_limit(q, k, backend, limit)
    estimate = score_matrix_estimate(q, k)

    run = make_attention_callable(q, k, v, mode=mode, backend=backend, attn_mask=attn_mask)
    measurement = benchmark_callable(
        run,
        dev,
        warmup=warmup,
        repeats=repeats,
        percentile_method=percentile_method,
        measure_memory=measure_memory,
        require_memory=require_memory,
    )

    simulated = mode != "reference"
    notes = [LATENCY_SCOPE_NOTE]
    if simulated:
        notes.append(FAKE_QUANT_NOTE)
    notes.append(DENSE_NOTE if backend == BACKEND_DENSE_ORACLE else SDPA_NOTE)
    if input_kind == INPUT_SYNTHETIC:
        notes.append("Synthetic inputs: smoke-test value only, not a research finding.")
    else:
        notes.append(REAL_INPUT_NOTE)

    result: dict[str, Any] = {
        "scope": SCOPE_OPERATOR,
        "input_kind": input_kind,
        "evidence_status": (
            EVIDENCE_SYNTHETIC if input_kind == INPUT_SYNTHETIC else EVIDENCE_REAL_DECLARED
        ),
        "shape_source": shape_source,
        "workload_name": workload_name,
        "backend": backend,
        "api_backend": BACKEND_LABEL_TO_API[backend],
        "precision_mode": mode,
        "precision_simulated_int8": simulated,
        "fake_quant_included_in_latency": simulated,
        "precision_note": FAKE_QUANT_NOTE if simulated else REFERENCE_PRECISION_NOTE,
        "q_shape": list(q.shape),
        "k_shape": list(k.shape),
        "v_shape": list(v.shape),
        "dtype": str(q.dtype),
        "batch_size": int(q.shape[0]),
        "num_heads": int(q.shape[1]),
        "has_attn_mask": attn_mask is not None,
        "attn_mask_shape": None if attn_mask is None else list(attn_mask.shape),
        "attn_mask_dtype": None if attn_mask is None else str(attn_mask.dtype),
        "score_matrix_bytes": estimate["bytes"],
        "score_matrix_estimate": estimate,
        "score_matrix_note": SCORE_ESTIMATE_NOTE,
        "notes": notes,
    }
    result.update(measurement)  # device, warmup, repeats, samples, p50/p95, memory fields
    return result


def synthetic_cases(
    workloads: Iterable[OperatorWorkload],
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> list[OperatorInputs]:
    """Build **synthetic** inputs for each workload (outside any timed region)."""
    return [
        make_synthetic_inputs(w, device=device, dtype=dtype, seed=seed) for w in workloads
    ]


def run_operator_sweep(
    cases: Iterable[OperatorInputs],
    *,
    input_kind: str,
    shape_source: str | None = None,
    device: torch.device | str,
    modes: Sequence[str] = PRECISION_MODES,
    backends: Sequence[str] = (BACKEND_SDPA,),
    warmup: int = 10,
    repeats: int = 50,
    percentile_method: str = DEFAULT_PERCENTILE_METHOD,
    measure_memory: bool = True,
    require_memory: bool = False,
    dense_max_score_bytes: int = DEFAULT_DENSE_MAX_SCORE_BYTES,
) -> list[dict[str, Any]]:
    """Benchmark every (case, backend, mode) combination.

    Order is deterministic: cases in the given order, then backends, then modes. Every
    case/backend/mode is validated (including the dense limit) **before** anything is
    timed, so a bad configuration fails fast instead of part-way through a sweep.
    A dense request above the limit raises; it is never silently skipped.
    """
    cases = list(cases)
    backends = list(backends)
    modes = list(modes)
    if not cases or not backends or not modes:
        raise ValueError("cases, backends and modes must each contain at least one entry")
    _validate_input_kind(input_kind, shape_source)
    limit = _validate_dense_limit_value(dense_max_score_bytes)
    dev = resolve_device(device)
    for backend, mode in itertools.product(backends, modes):
        _validate_backend_mode(backend, mode)
    for case in cases:
        validate_operator_inputs(case.q, case.k, case.v, case.attn_mask, dev)
        for backend in backends:
            _validate_dense_limit(case.q, case.k, backend, limit)

    results = []
    for case in cases:
        for backend in backends:
            for mode in modes:
                results.append(
                    benchmark_attention_operator(
                        case.q,
                        case.k,
                        case.v,
                        mode=mode,
                        backend=backend,
                        device=dev,
                        input_kind=input_kind,
                        shape_source=shape_source,
                        attn_mask=case.attn_mask,
                        warmup=warmup,
                        repeats=repeats,
                        percentile_method=percentile_method,
                        measure_memory=measure_memory,
                        require_memory=require_memory,
                        dense_max_score_bytes=limit,
                        workload_name=case.name,
                    )
                )
    return results
