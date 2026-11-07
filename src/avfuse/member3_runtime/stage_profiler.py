"""Generic pipeline stage profiler (Segment C, Day 3).

This module is deliberately **decoupled from Member 2's and Member 1's code**. It never
imports their packages and never re-implements their pipeline. The caller binds each
stage to a zero-argument callable that already holds its prepared input (or supplies an
externally obtained measurement), and the profiler times each one independently through
Segment B's ``benchmark_callable``.

Stage vocabulary
----------------
The names follow the actual LAV-DF pipeline on Member 2's ``feature/member2-data`` branch
(``lavdf.py``, ``manifest.py``, ``splits.py``, ``scripts/m2_lavdf.py``,
``scripts/m2_build_manifest.py``) plus the encoder/model stages. See ``STAGE_DESCRIPTIONS``.
Listing a stage name is **not** evidence that it was measured: each record states
whether a callable was supplied and whether a measurement completed.

Composite stages
----------------
Member 2's current code has no separate callable boundaries for several operations
(``extract_clip`` runs the video and audio ffmpeg calls back to back; ``extract_record``
adds ``.npy`` serialization; ``m2_build_manifest.py`` runs validation, row construction,
audit and writing in one script; ``m2_lavdf.py select`` has no function boundary at all).
Such operations may be profiled only as a clearly named **combined** stage (see
``COMPOSITE_STAGES``). A combined measurement:

* is reported under its own name, never under the stages it covers;
* leaves the covered individual stages ``not_provided`` (their latency is NOT inferred
  or apportioned from the aggregate);
* never counts toward ``complete``.

Honesty rules
-------------
* A stage with no callable and no external measurement is ``not_provided`` and carries
  no timing keys (never 0 ms, never ``None`` standing in for a measurement).
* A failing callable raises ``StageExecutionError``; it is never converted into a timing.
* Independent stage statistics are NOT end-to-end latency and are never summed;
  ``end_to_end_measured`` is always ``False`` (no end-to-end callable is supported yet).
* ``input_kind`` is caller-declared and unverified. ``unspecified`` is the default;
  ``synthetic`` is test-only; ``real`` requires a non-empty ``source`` description.
* Preprocessing stages default to ``warmup=0, repeats=1`` so one-time operations are never
  repeated by accident; only the cached-feature stages (``fusion``, ``head``) default to
  repeated measurement. Override per stage with ``StageSettings``.
* Operations with side effects or caches (for example ``extract_record`` skips decoding
  when its ``.npy`` outputs already exist) must be bound to fresh state by the caller;
  the profiler cannot detect a cache hit.

Results are in-memory dicts; artifact/provenance writing belongs to Segment D.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from avfuse.member3_runtime.benchmark import (
    DEFAULT_PERCENTILE_METHOD,
    PERCENTILE_METHODS,
    benchmark_callable,
    resolve_device,
)

# ------------------------------------------------------------------- stage names
GROUP_PREPROCESSING = "preprocessing_once"
GROUP_CACHED_MODEL = "cached_feature_model"

#: Individual (non-composite) stages in pipeline order, with their scope group.
INDIVIDUAL_STAGES: dict[str, str] = {
    "metadata_indexing": GROUP_PREPROCESSING,
    "split_sampling": GROUP_PREPROCESSING,
    "selection_list_generation": GROUP_PREPROCESSING,
    "dataset_transfer": GROUP_PREPROCESSING,
    "archive_extraction": GROUP_PREPROCESSING,
    "video_decode": GROUP_PREPROCESSING,
    "audio_decode": GROUP_PREPROCESSING,
    "raw_array_serialization": GROUP_PREPROCESSING,
    "clips_csv_serialization": GROUP_PREPROCESSING,
    "clip_validation": GROUP_PREPROCESSING,
    "manifest_row_construction": GROUP_PREPROCESSING,
    "split_audit": GROUP_PREPROCESSING,
    "manifest_serialization": GROUP_PREPROCESSING,
    "feature_encoding": GROUP_PREPROCESSING,
    "fusion": GROUP_CACHED_MODEL,
    "head": GROUP_CACHED_MODEL,
}
#: Optional: only measured when an actual transfer / extraction operation is supplied.
OPTIONAL_STAGES = ("dataset_transfer", "archive_extraction")
REQUIRED_STAGES = tuple(s for s in INDIVIDUAL_STAGES if s not in OPTIONAL_STAGES)

#: Combined stages: name -> the individual stages whose work they contain.
COMPOSITE_STAGES: dict[str, tuple[str, ...]] = {
    "combined_selection_command": (
        "metadata_indexing", "split_sampling", "selection_list_generation",
    ),
    "combined_audio_video_decode": ("video_decode", "audio_decode"),
    "combined_decode_and_array_serialization": (
        "video_decode", "audio_decode", "raw_array_serialization",
    ),
    "combined_manifest_build": (
        "clip_validation", "manifest_row_construction", "split_audit", "manifest_serialization",
    ),
}

#: Deterministic reporting order (pipeline order; each composite sits after what it covers).
STAGE_ORDER = (
    "metadata_indexing",
    "split_sampling",
    "selection_list_generation",
    "combined_selection_command",
    "dataset_transfer",
    "archive_extraction",
    "video_decode",
    "audio_decode",
    "combined_audio_video_decode",
    "raw_array_serialization",
    "combined_decode_and_array_serialization",
    "clips_csv_serialization",
    "clip_validation",
    "manifest_row_construction",
    "split_audit",
    "manifest_serialization",
    "combined_manifest_build",
    "feature_encoding",
    "fusion",
    "head",
)
assert set(STAGE_ORDER) == set(INDIVIDUAL_STAGES) | set(COMPOSITE_STAGES)

STAGE_SCOPE_GROUP: dict[str, str] = {
    **INDIVIDUAL_STAGES,
    **{name: GROUP_PREPROCESSING for name in COMPOSITE_STAGES},
}

#: What each stage corresponds to on Member 2's branch (documentation only; never imported).
STAGE_DESCRIPTIONS: dict[str, str] = {
    "metadata_indexing": "lavdf.index_lavdf(root): read metadata.json, build one record per video.",
    "split_sampling": "lavdf.sample_per_split(records, caps, seed): per-split/category sampling.",
    "selection_list_generation": (
        "The list-writing branch of `m2_lavdf.py select`. It writes a file list; it does NOT "
        "download anything. Inline in main() - no standalone callable exists today."
    ),
    "dataset_transfer": (
        "External network download of the dataset (not performed by Member 2's code)."
    ),
    "archive_extraction": (
        "External local archive extraction, e.g. `tar -x -T <list>`: disk I/O, not network."
    ),
    "video_decode": (
        "The video ffmpeg call inside lavdf.extract_clip(); no separate callable today."
    ),
    "audio_decode": (
        "The audio ffmpeg call inside lavdf.extract_clip(); no separate callable today."
    ),
    "raw_array_serialization": (
        "The np.save of frames.npy / audio.npy inside lavdf.extract_record(); "
        "no separate callable today."
    ),
    "clips_csv_serialization": "lavdf.write_clips_csv(rows, path).",
    "clip_validation": "lavdf.long_enough() and lavdf.window_covers_fake() filtering.",
    "manifest_row_construction": "manifest.manifest_row_from_clip(clip).",
    "split_audit": "splits.split_audit(rows): source/identity leakage audit.",
    "manifest_serialization": (
        "manifest.write_manifest(rows, path). Note: it also runs verify_manifest() "
        "(schema + leakage validation) before writing; the two are not separable."
    ),
    "feature_encoding": (
        "Frozen-encoder feature generation (encoder cache). Pending: depends on Member 1's "
        "checkpoint; preprocess.py on Member 2's branch is still a placeholder."
    ),
    "fusion": "Cached-feature fusion operation (Member 1). Pending model handoff.",
    "head": "Detection head on the fused representation (Member 1). Pending model handoff.",
    "combined_selection_command": (
        "The whole `m2_lavdf.py select` command (indexing + sampling + list writing; also "
        "process start-up and printing if timed as a subprocess)."
    ),
    "combined_audio_video_decode": "lavdf.extract_clip(): video then audio ffmpeg, sequentially.",
    "combined_decode_and_array_serialization": (
        "lavdf.extract_record(): existence check, extract_clip(), np.save x2 and mmap shape "
        "reads. It SKIPS decoding if the .npy outputs already exist."
    ),
    "combined_manifest_build": (
        "The whole `m2_build_manifest.py` flow (also reads clips.csv, which has no stage of "
        "its own)."
    ),
}

# ------------------------------------------------------------------- statuses
STATUS_MEASURED = "measured"
STATUS_EXTERNALLY_MEASURED = "externally_measured"
STATUS_NOT_PROVIDED = "not_provided"
ORIGIN_BENCHMARK = "benchmark_callable"
ORIGIN_EXTERNAL = "external"

INPUT_SYNTHETIC = "synthetic"
INPUT_REAL = "real"
INPUT_UNSPECIFIED = "unspecified"
INPUT_MIXED = "mixed"
STAGE_INPUT_KINDS = (INPUT_SYNTHETIC, INPUT_REAL, INPUT_UNSPECIFIED)

EVIDENCE_SYNTHETIC = "synthetic_not_research_evidence"
EVIDENCE_UNSPECIFIED = "unspecified_not_research_evidence"
EVIDENCE_MIXED = "mixed_inputs_not_research_evidence"
EVIDENCE_REAL_DECLARED = "real_inputs_caller_declared"
EVIDENCE_NONE_MEASURED = "no_stage_measured"

#: Keys that only ever appear on records that actually carry a measurement.
LATENCY_KEYS = (
    "p50_ms", "p95_ms", "timing_samples_ms", "mean_ms", "median_ms", "min_ms", "max_ms",
)

END_TO_END_CAVEAT = (
    "Stage statistics were measured independently, each on its own prepared input. Their "
    "medians must not be summed or reported as end-to-end latency; no end-to-end run was "
    "measured by this profile."
)
SCOPE_CAVEAT = (
    "preprocessing_once stages (dataset preparation, decoding, serialization, manifest, "
    "feature_encoding) are one-time costs; fusion/head are cached_feature_model costs. "
    "Never report cached-feature latency as raw-video end-to-end latency, and never mix "
    "the groups into one latency number."
)
INPUT_CAVEAT = (
    "input_kind is declared by the caller and is not verified by the profiler. Synthetic "
    "and unspecified profiles are software validation only, not research findings."
)
COMPOSITE_CAVEAT = (
    "Combined stages are aggregates: the stages they cover remain not_provided and no "
    "per-stage latency is inferred from them."
)


class StageExecutionError(RuntimeError):
    """A stage callable (or its timing setup) failed. No profile is produced."""

    def __init__(self, stage: str, cause: BaseException, completed: list[dict[str, Any]]):
        super().__init__(f"stage {stage!r} failed: {type(cause).__name__}: {cause}")
        self.stage = stage
        self.completed_stages = completed


# ------------------------------------------------------------- caller-side specs
def _check_count(label: str, value: Any, minimum: int, context: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{context}: {label} must be an int >= {minimum}, got {value!r}.")


@dataclass(frozen=True)
class StageSettings:
    """Per-stage warm-up / repeat counts. ``StageSettings(warmup=0, repeats=1)`` is the
    one-time setting used for preprocessing operations."""

    warmup: int
    repeats: int

    def __post_init__(self) -> None:
        _check_count("warmup", self.warmup, 0, "StageSettings")
        _check_count("repeats", self.repeats, 1, "StageSettings")


ONE_TIME_SETTINGS = StageSettings(warmup=0, repeats=1)
REPEATABLE_SETTINGS = StageSettings(warmup=10, repeats=50)


@dataclass(frozen=True)
class StageInput:
    """Caller-declared input metadata for a stage. ``real`` requires a ``source``."""

    kind: str = INPUT_UNSPECIFIED
    source: str | None = None

    def __post_init__(self) -> None:
        _check_input(self.kind, self.source, "StageInput")


@dataclass(frozen=True)
class ExternalMeasurement:
    """A duration measured outside this profiler (e.g. a recorded Colab run, or a ``tar``
    or download timed by the shell). ``source`` says where it came from and is required.
    No percentiles are derived and no synchronization is claimed."""

    durations_ms: Sequence[float]
    source: str

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("ExternalMeasurement requires a non-empty source description.")
        values = list(self.durations_ms)
        if not values:
            raise ValueError("ExternalMeasurement requires at least one duration.")
        for v in values:
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"External duration must be a finite number, got {v!r}.")
            if v <= 0:
                raise ValueError(
                    f"External duration must be > 0 ms (got {v!r}); a missing measurement "
                    "must be omitted, not recorded as zero."
                )
        object.__setattr__(self, "durations_ms", tuple(float(v) for v in values))


def _check_input(kind: str, source: str | None, context: str) -> None:
    if kind not in STAGE_INPUT_KINDS:
        raise ValueError(f"{context}: input kind must be one of {STAGE_INPUT_KINDS}, got {kind!r}.")
    if kind == INPUT_REAL and not (isinstance(source, str) and source.strip()):
        raise ValueError(
            f"{context}: input kind 'real' requires a non-empty source describing the real "
            "data (caller-declared; the profiler cannot verify it)."
        )


def _check_stage_name(name: str, context: str) -> None:
    if name not in STAGE_ORDER:
        raise ValueError(f"Unknown stage {name!r} in {context}; valid stages: {STAGE_ORDER}.")


# ------------------------------------------------------------------- profiler
def profile_pipeline_stages(
    stages: Mapping[str, Callable[[], Any]],
    *,
    device: torch.device | str,
    external_measurements: Mapping[str, ExternalMeasurement] | None = None,
    input_kind: str = INPUT_UNSPECIFIED,
    input_source: str | None = None,
    stage_inputs: Mapping[str, StageInput] | None = None,
    stage_settings: Mapping[str, StageSettings] | None = None,
    repeatable_settings: StageSettings = REPEATABLE_SETTINGS,
    stage_devices: Mapping[str, torch.device | str] | None = None,
    percentile_method: str = DEFAULT_PERCENTILE_METHOD,
    measure_memory: bool = True,
) -> dict[str, Any]:
    """Profile each supplied stage independently and report what is missing.

    Args:
        stages: stage name -> zero-argument callable bound to its prepared input. Valid
            names are ``STAGE_ORDER`` (individual and combined stages).
        device: default device for synchronization/memory (``cpu`` for CPU-only stages).
        external_measurements: stage name -> ``ExternalMeasurement`` for durations
            measured elsewhere. A stage may not be given both a callable and one.
        input_kind / input_source: default caller-declared input metadata for all stages.
            ``real`` needs ``input_source``. Default ``unspecified`` (non-research).
        stage_inputs: per-stage ``StageInput`` overrides.
        stage_settings: per-stage ``StageSettings``. Without an override, preprocessing
            stages use ``ONE_TIME_SETTINGS`` (warmup=0, repeats=1) and the
            ``cached_feature_model`` stages (fusion, head) use ``repeatable_settings``
            (default warmup=10, repeats=50).
        stage_devices: per-stage device override.

    Raises:
        StageExecutionError: a stage callable raised (original exception is ``__cause__``).
        ValueError / TypeError: invalid names, specs or settings - all checked before any
            stage runs. ``DeviceUnavailableError`` is likewise raised before any stage.
    """
    external = dict(external_measurements or {})
    stage_inputs = dict(stage_inputs or {})
    stage_settings = dict(stage_settings or {})
    stage_devices = dict(stage_devices or {})
    stages = dict(stages)

    # ---- validate everything up front: nothing runs until all of this passes
    _check_input(input_kind, input_source, "input_kind/input_source")
    if percentile_method not in PERCENTILE_METHODS:
        raise ValueError(
            f"Unknown percentile method {percentile_method!r}; use one of {PERCENTILE_METHODS}"
        )
    if not isinstance(repeatable_settings, StageSettings):
        raise TypeError("repeatable_settings must be a StageSettings instance")
    for name, fn in stages.items():
        _check_stage_name(name, "stages")
        if not callable(fn):
            raise TypeError(f"Stage {name!r} must be a zero-argument callable, got {fn!r}")
    for name, ext in external.items():
        _check_stage_name(name, "external_measurements")
        if not isinstance(ext, ExternalMeasurement):
            raise TypeError(f"external_measurements[{name!r}] must be an ExternalMeasurement")
        if name in stages:
            raise ValueError(
                f"Stage {name!r} has both a callable and an external measurement; supply one."
            )
    supplied = set(stages) | set(external)
    for map_name, mapping, allowed in (
        ("stage_inputs", stage_inputs, supplied),
        ("stage_settings", stage_settings, set(stages)),
        ("stage_devices", stage_devices, set(stages)),
    ):
        for name in mapping:
            _check_stage_name(name, map_name)
            if name not in allowed:
                what = "callable" if allowed is not supplied else "callable or external measurement"
                raise ValueError(f"{map_name} refers to stage {name!r}, which has no {what}.")
    for name, spec in stage_inputs.items():
        if not isinstance(spec, StageInput):
            raise TypeError(f"stage_inputs[{name!r}] must be a StageInput instance")
    for name, spec in stage_settings.items():
        if not isinstance(spec, StageSettings):
            raise TypeError(f"stage_settings[{name!r}] must be a StageSettings instance")
    default_device = resolve_device(device)
    resolved_devices = {n: resolve_device(d) for n, d in stage_devices.items()}

    def input_for(name: str) -> tuple[str, str | None]:
        spec = stage_inputs.get(name)
        return (spec.kind, spec.source) if spec else (input_kind, input_source)

    # ---- measure (each supplied stage independently, in pipeline order)
    records: list[dict[str, Any]] = []
    for name in STAGE_ORDER:
        is_composite = name in COMPOSITE_STAGES
        base = {
            "stage": name,
            "stage_kind": "composite" if is_composite else "individual",
            "scope_group": STAGE_SCOPE_GROUP[name],
            "required": name in REQUIRED_STAGES,
        }
        if is_composite:
            base["covers"] = list(COMPOSITE_STAGES[name])
        if name in stages:
            if name in stage_settings:
                settings, settings_source = stage_settings[name], "stage_settings"
            elif STAGE_SCOPE_GROUP[name] == GROUP_CACHED_MODEL:
                settings, settings_source = repeatable_settings, "default_repeatable"
            else:
                settings, settings_source = ONE_TIME_SETTINGS, "default_one_time"
            kind, source = input_for(name)
            try:
                measurement = benchmark_callable(
                    stages[name],
                    resolved_devices.get(name, default_device),
                    warmup=settings.warmup,
                    repeats=settings.repeats,
                    percentile_method=percentile_method,
                    measure_memory=measure_memory,
                )
            except Exception as exc:  # a failure is never turned into a timing
                raise StageExecutionError(name, exc, list(records)) from exc
            record = {
                **base,
                "status": STATUS_MEASURED,
                "callable_supplied": True,
                "measurement_completed": True,
                "measurement_origin": ORIGIN_BENCHMARK,
                "settings_source": settings_source,
                "input_kind": kind,
                "input_source": source,
            }
            record.update(measurement)
            records.append(record)
        elif name in external:
            ext = external[name]
            kind, source = input_for(name)
            records.append(
                {
                    **base,
                    "status": STATUS_EXTERNALLY_MEASURED,
                    "callable_supplied": False,
                    "measurement_completed": True,
                    "measurement_origin": ORIGIN_EXTERNAL,
                    "external_source": ext.source,
                    "input_kind": kind,
                    "input_source": source,
                    "repeats": len(ext.durations_ms),
                    "timing_samples_ms": list(ext.durations_ms),
                    "synchronization": "not_applicable_external",
                    "note": (
                        "Duration supplied by the caller from an external measurement; not "
                        "timed or synchronized by this profiler."
                    ),
                }
            )
        elif not is_composite:  # composites are listed only when supplied
            records.append(
                {
                    **base,
                    "status": STATUS_NOT_PROVIDED,
                    "callable_supplied": False,
                    "measurement_completed": False,
                    "measurement_origin": None,
                    "input_kind": None,
                    "input_source": None,
                    "note": "No callable or external measurement supplied; nothing was "
                    "measured (this is not 0 ms).",
                }
            )

    # ---- annotate individual stages that only a composite measurement touches
    measured_composites = [r["stage"] for r in records
                           if r["stage_kind"] == "composite" and r["measurement_completed"]]
    for r in records:
        if r["status"] == STATUS_NOT_PROVIDED:
            covering = [c for c in measured_composites if r["stage"] in COMPOSITE_STAGES[c]]
            if covering:
                r["covered_only_by_composite"] = covering
                r["note"] += (
                    " A combined measurement covers this stage, but its latency cannot be "
                    "separated from it."
                )

    individual = [r for r in records if r["stage_kind"] == "individual"]
    measured = [r["stage"] for r in individual if r["measurement_completed"]]
    missing = [r["stage"] for r in individual if r["required"] and not r["measurement_completed"]]
    optional_missing = [
        r["stage"] for r in individual if not r["required"] and not r["measurement_completed"]
    ]

    scope_summary: dict[str, dict[str, Any]] = {}
    for group in (GROUP_PREPROCESSING, GROUP_CACHED_MODEL):
        req = [s for s in REQUIRED_STAGES if STAGE_SCOPE_GROUP[s] == group]
        scope_summary[group] = {
            "required_stages": req,
            "measured_stages": [s for s in req if s in measured],
            "missing_stages": [s for s in req if s in missing],
            "complete": all(s in measured for s in req),
        }

    done = [r for r in records if r["measurement_completed"]]
    kinds = {r["input_kind"] for r in done}
    if not kinds:
        overall_kind, evidence = input_kind, EVIDENCE_NONE_MEASURED
    elif len(kinds) > 1:
        overall_kind, evidence = INPUT_MIXED, EVIDENCE_MIXED
    else:
        overall_kind = next(iter(kinds))
        evidence = {
            INPUT_SYNTHETIC: EVIDENCE_SYNTHETIC,
            INPUT_UNSPECIFIED: EVIDENCE_UNSPECIFIED,
            INPUT_REAL: EVIDENCE_REAL_DECLARED,
        }[overall_kind]

    complete = not missing
    caveats = [END_TO_END_CAVEAT, SCOPE_CAVEAT, INPUT_CAVEAT]
    if measured_composites:
        caveats.append(COMPOSITE_CAVEAT)
    return {
        "profile_type": "pipeline_stage_profile",
        "input_kind": overall_kind,
        "evidence_status": evidence,
        "device": str(default_device),
        "stage_order": [r["stage"] for r in records],
        "stages": records,
        "defined_stages": list(INDIVIDUAL_STAGES),
        "required_stages": list(REQUIRED_STAGES),
        "optional_stages": list(OPTIONAL_STAGES),
        "measured_stages": measured,
        "missing_stages": missing,
        "optional_not_provided": optional_missing,
        "composite_measurements": measured_composites,
        "scope_summary": scope_summary,
        "complete": complete,
        "eligible_as_real_pipeline_profile": complete and evidence == EVIDENCE_REAL_DECLARED,
        "end_to_end_measured": False,
        "caveats": caveats,
    }
