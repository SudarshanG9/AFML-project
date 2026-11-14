"""Cached-feature model benchmarking (scope ``cached_feature_model``).

Times ``model.predict_from_cached_features(features, mode)`` on one fixed cached sample
using Segment B's ``benchmark_callable`` and records the result through Segment D's
``RunContext``. This is detector inference on cached encoder tokens: it is NOT raw-video
end-to-end latency (no media decoding, no frozen-encoder extraction).

Honesty rules enforced here
* Only modes that executed successfully produce a record; unsupported modes produce none.
* A strict run (default) with any unsupported requested mode times nothing and writes nothing.
* Conversion overhead is never derived by subtracting latencies. It is stored only when a
  dedicated conversion-only callable is supplied; otherwise it is ``null``.
* ``backend`` is ``"none"`` unless the caller states a verified backend.
* A requested accelerator that is unavailable raises (Segment B); there is no CPU fallback.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from avfuse.shared.contracts import PRECISION_MODES

from . import provenance as prov
from .artifacts import RunContext
from .benchmark import DEFAULT_PERCENTILE_METHOD, benchmark_callable, resolve_device
from .runner import (
    SweepResult,
    infer_batch_size,
    run_precision_sweep,
)

SCOPE = "cached_feature_model"
INPUT_BOUNDARY = "cached_encoder_tokens"
SAMPLE_HASH_ALGORITHM = "sha256-tensor-bytes-v1"

_BASE_NOTE = (
    "Cached-feature detector inference only (predict_from_cached_features); excludes raw-media "
    "decoding and frozen-encoder extraction. Not raw-video end-to-end latency."
)
_NO_CONVERSION_NOTE = (
    " Conversion overhead is not isolated (null): any quantization/conversion work inside the "
    "model forward is included in the total model latency."
)


class CheckpointIdentityError(RuntimeError):
    """The model's checkpoint hash disagrees with the run's recorded checkpoint hash."""


class ModelDeviceMismatchError(RuntimeError):
    """Model device differs from the benchmark device (no silent device change)."""


@dataclass(frozen=True)
class ConversionBenchmark:
    """A dedicated, conversion-only callable supplied by Member 1.

    ``description`` must state exactly what the callable includes.
    ``included_in_latency`` says whether that cost is already inside the model latency.
    """

    fn: Callable[[], Any]
    description: str
    included_in_latency: bool = True


@dataclass
class ModelBenchmarkResult:
    status: str                                  # complete | partial | blocked
    sweep: SweepResult
    records: list[dict] = field(default_factory=list)
    benchmarked_modes: tuple[str, ...] = ()
    written: dict | None = None
    input_identity_sha256: str | None = None

    @property
    def four_mode_complete(self) -> bool:
        return set(self.benchmarked_modes) == set(PRECISION_MODES)


def sample_identity(features: Mapping[str, Tensor]) -> dict[str, Any]:
    """Hash record identifying the exact benchmarked tensors (dtype, shape and bytes)."""
    digest = hashlib.sha256()
    shapes = {}
    for key in ("audio_tokens", "video_tokens"):
        t = features[key].detach().cpu().contiguous()
        digest.update(f"{key}|{t.dtype}|{tuple(t.shape)}|".encode())
        digest.update(t.numpy().tobytes())
        shapes[key] = list(t.shape)
    return prov.make_hash_record(
        prov.KIND_CACHE_CONTENT, SAMPLE_HASH_ALGORITHM, digest.hexdigest(),
        "benchmarked_sample_tensors", name="benchmark_sample", shapes=shapes,
        note="Identifies the exact cached tensors that were timed, not the whole feature cache.")


def _checkpoint_record(ctx: RunContext, model: Any, checkpoint_sha256: str | None) -> dict | None:
    actual = checkpoint_sha256 or getattr(model, "checkpoint_sha256", None)
    recorded = ctx.checkpoint
    if recorded is not None and actual is not None and recorded["sha256"] != actual:
        raise CheckpointIdentityError(
            f"Run context checkpoint hash {recorded['sha256']} != model checkpoint hash {actual}")
    if recorded is not None:
        return recorded
    if actual is None:
        return None
    return prov.make_hash_record(prov.KIND_CHECKPOINT, prov.ALGORITHM_FILE, actual,
                                 prov.SCOPE_FILE_BYTES, name="model_reported_checkpoint")


def _assert_eval(model: Any) -> None:
    for candidate in (model, getattr(model, "model", None)):
        if candidate is not None and getattr(candidate, "training", False) is True:
            raise RuntimeError("Model is in training mode; call eval() before benchmarking")

def _assert_device(model: Any, device: torch.device | str) -> None:
    model_device = getattr(model, "device", None)

    if model_device is None:
        return

    want_spec = torch.device(device)
    have_spec = torch.device(model_device)

    mismatch = have_spec.type != want_spec.type

    if (
        have_spec.type == want_spec.type == "cuda"
        and have_spec.index is not None
        and want_spec.index is not None
        and have_spec.index != want_spec.index
    ):
        mismatch = True

    if mismatch:
        raise ModelDeviceMismatchError(
            f"Model is on {have_spec} but the benchmark device is "
            f"{want_spec}; move the model explicitly"
        )

    want = resolve_device(device)
    have = resolve_device(model_device)

    if have != want:
        raise ModelDeviceMismatchError(
            f"Model is on {have} but the benchmark device is {want}; "
            "move the model explicitly"
        )

def benchmark_precision_modes(
    model: Any,
    features: Mapping[str, Tensor],
    *,
    ctx: RunContext,
    device: torch.device | str = "cpu",
    modes: Sequence[str] = PRECISION_MODES,
    warmup: int = 10,
    repeats: int = 50,
    percentile_method: str = DEFAULT_PERCENTILE_METHOD,
    measure_memory: bool = True,
    require_memory: bool = False,
    synchronize: bool = True,
    execution_dtype: str = "float32",
    backend: str = "none",
    allow_partial: bool = False,
    checkpoint_sha256: str | None = None,
    input_identity: dict[str, Any] | None = None,
    conversion_benchmarks: Mapping[str, ConversionBenchmark] | None = None,
    input_duration_seconds: float | None = None,
    write_records: bool = True,
) -> ModelBenchmarkResult:
    """Benchmark every requested, supported mode on the same fixed cached sample.

    Strict by default: if a requested mode is unsupported the call returns status
    ``blocked`` without timing or writing anything. With ``allow_partial=True`` only the
    supported modes are timed and the result is labelled ``partial``.
    Ordinary runtime errors propagate; ``DeviceUnavailableError`` is never converted to CPU.
    """
    _assert_eval(model)
    _assert_device(model, device)
    conversion_benchmarks = dict(conversion_benchmarks or {})
    if "reference" in conversion_benchmarks:
        raise ValueError("reference mode has no precision conversion to benchmark")

    # One checked pass per mode (also validates the score contract and finds unsupported modes).
    sweep = run_precision_sweep(model, features, modes, allow_partial=allow_partial)
    if sweep.status == "blocked" or not sweep.completed_modes:
        return ModelBenchmarkResult(status="blocked", sweep=sweep)

    checkpoint = _checkpoint_record(ctx, model, checkpoint_sha256)
    identity = input_identity if input_identity is not None else sample_identity(features)
    batch = infer_batch_size(features)
    shapes = {k: list(features[k].shape) for k in ("audio_tokens", "video_tokens")}
    records: list[dict] = []
    done: list[str] = []
    for mode in sweep.completed_modes:  # same sample object for every mode
        measurement = benchmark_callable(
            lambda mode=mode: model.predict_from_cached_features(features, mode),
            device, warmup=warmup, repeats=repeats, percentile_method=percentile_method,
            measure_memory=measure_memory, require_memory=require_memory,
            synchronize=synchronize)
        conv = conversion_benchmarks.get(mode)
        overhead = included = None
        notes = _BASE_NOTE
        if conv is not None:
            conv_m = benchmark_callable(
                conv.fn, device, warmup=warmup, repeats=repeats,
                percentile_method=percentile_method, measure_memory=False, synchronize=synchronize)
            overhead, included = float(conv_m["p50_ms"]), bool(conv.included_in_latency)
            notes += f" Conversion overhead (p50) from a dedicated callable: {conv.description}."
        elif mode != "reference":
            notes += _NO_CONVERSION_NOTE
        records.append(ctx.build_record(
            measurement, config=mode, scope=SCOPE, backend=backend, batch_size=batch,
            execution_dtype=execution_dtype, input_boundary=INPUT_BOUNDARY,
            input_shapes=shapes, input_duration_seconds=input_duration_seconds,
            checkpoint=checkpoint, input_identity=identity,
            conversion_overhead_ms=overhead, conversion_overhead_included_in_latency=included,
            notes=notes,
            extensions={"model_benchmark": {
                "timed_call": "predict_from_cached_features",
                "same_cached_sample_for_all_modes": True,
                "backend_note": (
                    "backend 'none': the full-model attention backend is not claimed; "
                    "operator-level sdpa is a separate scope" if backend == "none" else
                    f"caller-verified full-model backend: {backend}"
                )}}))
        done.append(mode)

    status = "complete" if sweep.status == "complete" else "partial"
    written = None
    if write_records:
        written = ctx.write_benchmark_records(records)
    return ModelBenchmarkResult(
        status=status, sweep=sweep, records=records, benchmarked_modes=tuple(done),
        written=written, input_identity_sha256=identity.get("sha256"))
