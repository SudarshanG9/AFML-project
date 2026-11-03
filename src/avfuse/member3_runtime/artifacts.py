"""Run identity, benchmark-record construction/validation, artifact writing and integrity.

Segment D (provenance and artifact management). Nothing in this module runs a benchmark,
a model or any dataset processing; it stores and verifies what other segments measured.

Run directory layout (``<runs_root>/<run_id>/``; ``artifacts/`` is git-ignored)::

    resolved_config.json          snapshot of the resolved configuration (if snapshot_config)
    environment_report.json       environment_report.collect_environment_report() output
    provenance.json               hashes, git state, environment summary
    benchmark_records.jsonl       schema v0.2 records, one per line
    timing_samples/<record_id>.json   only when timing_samples_storage == "separate_file"
    stage_profile.json            optional avfuse_stage_profile artifact
    run_manifest.json             artifact list: path, type, size, SHA-256, time, required
    integrity/artifact_hashes.json    anchor: SHA-256 of run_manifest.json + hash list

Self-hash policy: ``run_manifest.json`` lists every artifact except itself and
``integrity/artifact_hashes.json``. The manifest's own SHA-256 is stored in the anchor file,
which no other file covers, so there is no circular dependency. The anchor detects accidental
corruption and inconsistent edits; it is **not** tamper-proof against someone who can rewrite
both files.

Concurrency: unique run IDs keep separate runs apart and no-clobber writes use an atomic
hard-link where the filesystem supports it, but a single run directory has **one writer**.
There is no file locking and no safe concurrent appending to one run.
"""
from __future__ import annotations

import copy
import json
import math
import os
import re
import statistics
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

from avfuse.member3_runtime import provenance as prov
from avfuse.member3_runtime.benchmark import compute_percentile
from avfuse.member3_runtime.environment import collect_environment_report, write_report
from avfuse.member3_runtime.stage_profiler import LATENCY_KEYS

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_SCHEMA_PATH = REPO_ROOT / "contracts" / "schemas" / "benchmark.schema.json"
CONFIG_SCHEMA_PATH = REPO_ROOT / "contracts" / "schemas" / "benchmark_config.schema.json"
SCHEMA_VERSION = "0.2"

MANIFEST_NAME = "run_manifest.json"
ANCHOR_NAME = "integrity/artifact_hashes.json"
CONFIG_NAME = "resolved_config.json"
ENVIRONMENT_NAME = "environment_report.json"
PROVENANCE_NAME = "provenance.json"
RECORDS_NAME = "benchmark_records.jsonl"
STAGE_PROFILE_NAME = "stage_profile.json"
SAMPLES_DIR = "timing_samples"
RESERVED_NAMES = frozenset({MANIFEST_NAME, ANCHOR_NAME})

ARTIFACT_TYPES = {
    "config": "resolved_config",
    "environment": "avfuse_environment_report",
    "provenance": "avfuse_run_provenance",
    "records": "avfuse_benchmark_records",
    "samples": "avfuse_timing_samples",
    "stage_profile": "avfuse_stage_profile",
    "manifest": "avfuse_run_manifest",
    "anchor": "avfuse_artifact_hashes",
}
ARTIFACT_VERSION = "1.0"

PRECISION_CONFIGS = ("reference", "low_precision", "restore_value_output", "restore_query")
STORAGE_MODES = ("inline", "separate_file")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_UTC_Z = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$")
_FORBIDDEN_EXTENSION_KEYS = frozenset({
    "label", "labels", "y_true", "y_pred", "ground_truth", "target", "targets",
    "prediction", "predictions",
})


class ArtifactError(Exception):
    """Base class for artifact-management failures."""


class ArtifactExistsError(ArtifactError, FileExistsError):
    """A file or run directory exists and overwrite is not enabled."""


class UnsafePathError(ArtifactError, ValueError):
    """A path is absolute, escapes the run directory, or is otherwise unsafe."""


class RunFinalizedError(ArtifactError):
    """The run manifest was already written; no further artifacts may be registered."""


class BenchmarkValidationError(ArtifactError, ValueError):
    """One or more benchmark records are invalid; ``errors`` lists every problem found."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = list(errors)
        shown = "\n  - ".join(self.errors[:20])
        more = f"\n  ... and {len(self.errors) - 20} more" if len(self.errors) > 20 else ""
        super().__init__(f"{len(self.errors)} benchmark validation error(s):\n  - {shown}{more}")


class DuplicateRecordError(BenchmarkValidationError):
    """Two records describe the same measurement series."""


# ------------------------------------------------------------------------------ time and ids
def utc_timestamp(moment: datetime | None = None) -> str:
    """Timezone-aware UTC time as ISO 8601 with millisecond precision and a trailing ``Z``."""
    if moment is None:
        moment = datetime.now(timezone.utc)
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("Naive datetimes are not accepted; pass a timezone-aware datetime")
    utc = moment.astimezone(timezone.utc)
    return utc.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_utc_timestamp(text: str) -> datetime:
    """Parse a ``...Z`` timestamp; raises ``ValueError`` for anything else."""
    if not isinstance(text, str) or not _UTC_Z.match(text):
        raise ValueError(f"Not an ISO 8601 UTC timestamp with trailing Z: {text!r}")
    return datetime.fromisoformat(text[:-1] + "+00:00")


def validate_run_id(run_id: str) -> str:
    """Run IDs are filesystem-safe: letters, digits, ``.``, ``_``, ``-``; max 128 chars."""
    if not isinstance(run_id, str) or not _SAFE_ID.match(run_id) or ".." in run_id:
        raise ValueError(
            f"Unsafe run_id {run_id!r}: use 1-128 characters from [A-Za-z0-9._-], "
            "starting with a letter or digit, without '..'"
        )
    return run_id


def generate_run_id(moment: datetime | None = None) -> str:
    """``run-<UTC time>-<16 random hex>``. Unique, but NOT a content hash."""
    stamp = (moment or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return f"run-{stamp:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:16]}"


# ------------------------------------------------------------------------------ safe paths
def validate_relative_path(rel: str) -> str:
    """Return ``rel`` if it is a safe, ``/``-separated path relative to the run directory."""
    if not isinstance(rel, str) or not rel:
        raise UnsafePathError("Artifact path must be a non-empty string")
    if rel.startswith("/") or "\\" in rel or re.match(r"^[A-Za-z]:", rel):
        raise UnsafePathError(f"Artifact path must be relative and '/'-separated: {rel!r}")
    if any(ord(c) < 32 for c in rel):
        raise UnsafePathError(f"Artifact path contains control characters: {rel!r}")
    parts = rel.split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise UnsafePathError(f"Artifact path has empty, '.' or '..' components: {rel!r}")
    return rel


def resolve_within(run_dir: Path, rel: str) -> Path:
    """Join ``rel`` onto ``run_dir``, refusing symlinks and any escape from ``run_dir``."""
    validate_relative_path(rel)
    root = run_dir.resolve()
    current = run_dir
    for part in PurePosixPath(rel).parts:
        current = current / part
        if current.is_symlink():
            raise UnsafePathError(f"Symbolic links are not allowed inside a run: {rel!r}")
    resolved = (run_dir / rel).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise UnsafePathError(f"Path escapes the run directory: {rel!r}") from None
    return run_dir / rel


# -------------------------------------------------------------------------- atomic writing
def _fsync_directory(directory: Path) -> None:
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _commit_temp(tmp: Path, dest: Path, overwrite: bool) -> None:
    """Move ``tmp`` onto ``dest``: atomic replace, or atomic no-clobber via hard link."""
    if overwrite:
        os.replace(tmp, dest)
    else:
        try:
            os.link(tmp, dest)
        except FileExistsError:
            raise ArtifactExistsError(
                f"Refusing to overwrite existing file: {dest.name!r}"
            ) from None
        except OSError:
            # Hard links unsupported here: fall back to check-then-replace (small race window).
            if dest.exists() or dest.is_symlink():
                raise ArtifactExistsError(
                    f"Refusing to overwrite existing file: {dest.name!r}"
                ) from None
            os.replace(tmp, dest)
    _fsync_directory(dest.parent)


def _prepare_destination(dest: Path, overwrite: bool) -> Path:
    if dest.is_symlink():
        raise UnsafePathError(f"Refusing to write through a symbolic link: {dest.name!r}")
    if dest.is_dir():
        raise ArtifactError(f"Destination is a directory: {dest.name!r}")
    if dest.exists() and not overwrite:
        raise ArtifactExistsError(f"Refusing to overwrite existing file: {dest.name!r}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    return dest.with_name(f".{dest.name}.{uuid.uuid4().hex}.tmp")


def atomic_write_bytes(dest: str | Path, data: bytes, *, overwrite: bool = False) -> dict[str, Any]:
    """Write ``data`` to a temporary sibling, fsync, then atomically move it into place.

    On any failure the destination is left untouched and the temporary file is removed.
    Returns ``{"size_bytes", "sha256"}`` computed from the bytes written.
    """
    dest = Path(dest)
    tmp = _prepare_destination(dest, overwrite)
    try:
        with tmp.open("xb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        _commit_temp(tmp, dest, overwrite)
    finally:
        tmp.unlink(missing_ok=True)
    import hashlib

    return {"size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def serialize_json(value: Any, *, indent: int | None = 2) -> bytes:
    """Stable human-readable JSON: sorted keys, UTF-8, NaN/Infinity rejected, trailing newline."""
    prov.canonical_json_bytes(value)  # rejects non-finite numbers, non-string keys, odd types
    return (json.dumps(value, sort_keys=True, indent=indent, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def write_json_artifact(
    dest: str | Path, value: Any, *, overwrite: bool = False, indent: int | None = 2
) -> dict[str, Any]:
    """Atomically write ``value`` as UTF-8 JSON. See :func:`atomic_write_bytes`."""
    return atomic_write_bytes(dest, serialize_json(value, indent=indent), overwrite=overwrite)


def serialize_jsonl_line(record: dict[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


# ---------------------------------------------------------------------- record validation
@lru_cache(maxsize=4)
def _validator(schema_path: str) -> Draft202012Validator:
    path = Path(schema_path)
    if not path.is_file():
        raise ArtifactError(
            f"Benchmark schema not found: {path.name!r}. Pass schema_path= when running "
            "outside the repository checkout."
        )
    schema = json.loads(path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


def _finite_nonneg(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0)


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def series_key(record: dict[str, Any]) -> dict[str, Any]:
    """Fields that identify one measurement series within an experiment."""
    workload = record.get("workload") or {}
    precision = record.get("precision") or {}
    return {
        "experiment_id": record.get("experiment_id"), "config": record.get("config"),
        "scope": record.get("scope"), "device": record.get("device"),
        "backend": record.get("backend"), "batch_size": record.get("batch_size"),
        "input_boundary": workload.get("input_boundary"),
        "input_shapes": workload.get("input_shapes"),
        "execution_dtype": precision.get("execution_dtype"),
    }


def record_id_for(record: dict[str, Any]) -> str:
    """Deterministic, filesystem-safe identifier of a record's series (used for sample files)."""
    digest = prov.sha256_json(series_key(record))[:16]
    return f"{record['scope']}-{record['config']}-{digest}"


def _scan_keys(value: Any, path: str, out: list[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in _FORBIDDEN_EXTENSION_KEYS:
                out.append(f"{path}/{key}: dataset labels/predictions must not appear "
                           "in benchmark records")
            _scan_keys(item, f"{path}/{key}", out)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_keys(item, f"{path}[{index}]", out)


def _check_record(
    record: dict[str, Any], where: str, samples: list[float] | None, errors: list[str]
) -> None:
    """Cross-field rules the JSON Schema cannot (or cannot clearly) express."""
    try:
        parse_utc_timestamp(record["timestamp_utc"])
    except (ValueError, KeyError) as exc:
        errors.append(f"{where}/timestamp_utc: {exc}")
    if record["p50_ms"] > record["p95_ms"]:
        errors.append(f"{where}: p50_ms ({record['p50_ms']}) exceeds p95_ms ({record['p95_ms']})")

    method = record["measurement"]["percentile_method"]
    if samples is not None:
        if len(samples) != record["repeats"]:
            errors.append(
                f"{where}: {len(samples)} timing samples but repeats == {record['repeats']}")
        elif not all(_finite_nonneg(s) for s in samples):
            errors.append(f"{where}: timing samples must be finite and non-negative")
        else:
            expected = {
                "mean_ms": math.fsum(samples) / len(samples),
                "median_ms": float(statistics.median(samples)),
                "p50_ms": compute_percentile(samples, 50, method),
                "p95_ms": compute_percentile(samples, 95, method),
            }
            for key, value in expected.items():
                if not _close(float(record[key]), value):
                    errors.append(
                        f"{where}: {key}={record[key]} does not match {value} recomputed from "
                        f"the samples (percentile_method={method!r})"
                    )
    for key in ("mean_ms", "median_ms", "p50_ms", "p95_ms"):
        if not _finite_nonneg(record[key]):
            errors.append(f"{where}/{key}: must be finite and non-negative")

    device_type = record["device"].split(":")[0]
    sync = record["measurement"]["synchronization"]
    allowed_sync = {"cpu": {"not_required_cpu", "disabled"},
                    "cuda": {"cuda_synchronize", "disabled"},
                    "mps": {"mps_synchronize", "disabled"}}[device_type]
    if sync not in allowed_sync:
        errors.append(f"{where}: synchronization {sync!r} is inconsistent with device "
                      f"{record['device']!r}")
    if record["environment"]["device_metadata"]["type"] != device_type:
        errors.append(f"{where}: environment.device_metadata.type does not match device")

    peak, method_mem = record["peak_memory_bytes"], record["memory_method"]
    if (peak is None) != (method_mem == "not_measured"):
        errors.append(f"{where}: peak_memory_bytes is null exactly when memory_method is "
                      "'not_measured'")
    if method_mem == "cuda_max_memory_allocated" and device_type != "cuda":
        errors.append(f"{where}: CUDA memory method on non-CUDA device {record['device']!r}")
    scope, backend = record["scope"], record["backend"]
    boundary = record["workload"]["input_boundary"]
    if scope == "preprocessing_once" and (boundary != "raw_media" or backend != "none"):
        errors.append(f"{where}: preprocessing_once requires raw_media input and backend 'none'")
    if scope in ("operator", "cached_feature_model") and boundary == "raw_media":
        errors.append(f"{where}: scope {scope!r} cannot use raw_media input")
    if scope == "operator" and backend == "none":
        errors.append(f"{where}: operator scope requires an attention backend")

    precision = record["precision"]
    expected_mode = "native_float" if record["config"] == "reference" else "simulated_int8"
    if precision["mode"] != expected_mode:
        errors.append(f"{where}: config {record['config']!r} requires precision.mode "
                      f"{expected_mode!r}")
    if precision["real_int8_kernel"] is not False:
        errors.append(f"{where}: precision.real_int8_kernel must always be false")
    if precision["mode"] == "simulated_int8" and "simulat" not in precision["disclosure"].lower():
        errors.append(f"{where}: simulated_int8 disclosure must state that INT8 is simulated")
    _scan_keys(record.get("extensions", {}), f"{where}/extensions", errors)


def load_timing_samples(run_dir: str | Path, ref: str) -> list[float]:
    """Load and sanity-check a timing-sample artifact referenced by ``timing_samples_ref``."""
    path = resolve_within(Path(run_dir), ref)
    if not path.is_file():
        raise ArtifactError(f"Timing-sample file does not exist: {ref!r}")
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"Timing-sample file is unreadable: {ref!r}: {exc}") from exc
    if (not isinstance(content, dict)
            or content.get("artifact_type") != ARTIFACT_TYPES["samples"]
            or not isinstance(content.get("timing_samples_ms"), list)):
        raise ArtifactError(f"Not a valid timing-sample artifact: {ref!r}")
    return content["timing_samples_ms"]


def validate_benchmark_records(
    records: list[dict[str, Any]], *, run_id: str | None = None,
    run_dir: str | Path | None = None, schema_path: str | Path | None = None,
) -> None:
    """Validate a whole batch (schema v0.2 + cross-field rules); raise before anything is written.

    Exactly one sample representation is required. Inline samples are recomputed against the
    stated statistics; external references require ``run_dir`` and are checked against the file.
    Raises :class:`BenchmarkValidationError` listing *every* problem (or
    :class:`DuplicateRecordError` when only duplicates are found).
    """
    validator = _validator(str(schema_path or BENCHMARK_SCHEMA_PATH))
    errors: list[str] = []
    duplicates: list[str] = []
    seen: dict[str, int] = {}
    run_ids: set[str] = set()
    for index, record in enumerate(records):
        where = f"record[{index}]"
        if not isinstance(record, dict):
            errors.append(f"{where}: must be an object")
            continue
        schema_errors = sorted(validator.iter_errors(record), key=lambda e: list(map(str, e.path)))
        if schema_errors:
            for err in schema_errors:
                loc = "/".join(str(p) for p in err.absolute_path) or "(record)"
                errors.append(f"{where}/{loc}: {err.message[:300]}")
            continue  # cross-field checks assume schema validity
        run_ids.add(record["run_id"])
        if run_id is not None and record["run_id"] != run_id:
            errors.append(f"{where}: run_id {record['run_id']!r} != expected {run_id!r}")
        has_inline_samples = "timing_samples_ms" in record
        has_sample_reference = "timing_samples_ref" in record
        if has_inline_samples == has_sample_reference:
            errors.append(
                f"{where}: provide exactly one of timing_samples_ms or timing_samples_ref"
            )
        samples = record.get("timing_samples_ms") if has_inline_samples else None
        if has_sample_reference and not has_inline_samples:
            if run_dir is None:
                validate_relative_path(record["timing_samples_ref"])
                errors.append(
                    f"{where}: run_dir is required to resolve timing_samples_ref"
                )
            else:
                try:
                    samples = load_timing_samples(run_dir, record["timing_samples_ref"])
                except (ArtifactError, UnsafePathError) as exc:
                    errors.append(f"{where}: {exc}")
        _check_record(record, where, samples, errors)
        key = prov.sha256_json(series_key(record))
        if key in seen:
            duplicates.append(f"{where}: duplicates record[{seen[key]}] (same measurement series)")
        else:
            seen[key] = index
    if len(run_ids) > 1:
        errors.append(f"records mix several run_ids: {sorted(run_ids)}")
    if errors or duplicates:
        if errors:
            raise BenchmarkValidationError(errors + duplicates)
        raise DuplicateRecordError(duplicates)


# ----------------------------------------------------------------------- record building
def precision_block(config: str, execution_dtype: Any) -> dict[str, Any]:
    """Schema ``precision`` object for one of the four configurations."""
    if config not in PRECISION_CONFIGS:
        raise ValueError(f"Unknown config {config!r}; use one of {PRECISION_CONFIGS}")
    dtype = str(execution_dtype).removeprefix("torch.")
    if config == "reference":
        return {
            "mode": "native_float", "execution_dtype": dtype, "simulated": False,
            "real_int8_kernel": False,
            "disclosure": f"Executed natively in {dtype}; no quantization emulation.",
        }
    return {
        "mode": "simulated_int8", "execution_dtype": dtype, "simulated": True,
        "real_int8_kernel": False,
        "disclosure": (
            f"Simulated (fake-quantized) INT8 carried out in {dtype} floating-point arithmetic; "
            "no integer INT8 kernel was executed. Timings are not INT8 hardware speedups."
        ),
    }


def capture_git_provenance(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Git commit and dirty state via the existing ``environment.collect_git``."""
    from avfuse.member3_runtime.environment import collect_git

    info = collect_git(Path(repo_root) if repo_root is not None else Path.cwd())
    commit = info.get("commit")
    dirty = info.get("dirty")
    if not commit:
        return {"state": "unavailable", "commit": None, "dirty": None,
                "warning": "Git commit unavailable; the code version is not identified."}
    out: dict[str, Any] = {"state": "recorded", "commit": commit, "dirty": dirty,
                           "warning": None}
    if dirty:
        out["warning"] = ("Working tree has uncommitted or untracked changes; the commit does "
                          "not fully identify the code that ran.")
    elif dirty is None:
        out["warning"] = "Dirty state could not be determined."
    return out


def _section(report: dict[str, Any], name: str) -> dict[str, Any]:
    section = report.get(name)
    if not isinstance(section, dict) or "error" in section and len(section) == 1:
        raise ValueError(f"Environment report section {name!r} is missing or failed")
    return section


def environment_block(
    report: dict[str, Any], device: str, *, git: dict[str, Any] | None = None,
    environment_report_ref: str | None = None,
) -> dict[str, Any]:
    """Schema ``environment`` object for ``device``, built from an environment report.

    Device names are never inferred: unavailable names are ``None`` (PyTorch exposes no MPS
    device name; the host architecture is given as a property instead).
    """
    python_version = _section(report, "python").get("version")
    torch_info = _section(report, "torch")
    torch_version, os_platform = torch_info.get("version"), _section(report, "os").get("platform")
    if not (python_version and torch_version and os_platform):
        raise ValueError("Environment report lacks Python, PyTorch or OS version information")
    kind, _, index_text = device.partition(":")
    name: str | None = None
    properties: dict[str, Any] | None = None
    if kind == "cpu":
        cpu = report.get("cpu") if isinstance(report.get("cpu"), dict) else {}
        name = cpu.get("description")
        properties = {k: cpu.get(k) for k in ("logical_cores", "physical_cores")}
    elif kind == "cuda":
        cuda = torch_info.get("cuda") or {}
        devices = [d for d in cuda.get("devices", []) if "name" in d]
        chosen = None
        if index_text.isdigit():
            chosen = next((d for d in devices if d.get("index") == int(index_text)), None)
        elif len(devices) == 1:
            chosen = devices[0]
        if chosen:
            name = chosen["name"]
            properties = {k: v for k, v in chosen.items() if k not in ("index", "name")}
            properties["cuda_runtime_version"] = cuda.get("runtime_version")
    elif kind == "mps":
        mps = torch_info.get("mps") or {}
        properties = {"host_architecture": mps.get("host_architecture")}
    else:
        raise ValueError(f"Unsupported device {device!r}")
    git = git or {}
    return {
        "python_version": python_version, "torch_version": torch_version,
        "os_platform": os_platform,
        "device_metadata": {"type": kind, "name": name, "properties": properties},
        "git_commit": git.get("commit"), "git_dirty": git.get("dirty"),
        "environment_report_ref": environment_report_ref,
    }


_MEASUREMENT_KEYS = (
    "device", "warmup", "repeats", "timing_samples_ms", "mean_ms", "median_ms", "p50_ms",
    "p95_ms", "peak_memory_bytes", "memory_method", "percentile_method", "synchronization",
)


def build_benchmark_record(
    measurement: dict[str, Any], *, run_id: str, experiment_id: str, config: str, scope: str,
    backend: str, batch_size: int, execution_dtype: Any, input_boundary: str,
    input_shapes: dict[str, list[int]] | None, environment: dict[str, Any],
    input_duration_seconds: float | None = None, checkpoint: dict[str, Any] | None = None,
    input_identity: dict[str, Any] | None = None, conversion_overhead_ms: float | None = None,
    conversion_overhead_included_in_latency: bool | None = None, notes: str | None = None,
    timestamp_utc: str | None = None, extensions: dict[str, Any] | None = None,
    schema_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build one schema v0.2 record from a ``benchmark_callable()`` measurement dictionary.

    The measurement dictionary is never written as-is. ``percentile_method`` and
    ``synchronization`` move into ``measurement``; diagnostics with no schema home
    (``peak_memory_reserved_bytes``, ``memory_baseline_bytes``, ``memory_note``) go under
    ``extensions.benchmark_engine``; unavailable memory stays ``None``. The inline-sample
    record is validated (including recomputing statistics from the samples) before returning.

    ``checkpoint`` / ``input_identity`` are hash records from ``provenance`` (or ``None``).
    A ``cache_manifest`` identity yields ``input_identity_sha256 = None`` plus an explanatory
    ``extensions.provenance`` entry, because a manifest digest is not a content digest.
    """
    missing = [k for k in _MEASUREMENT_KEYS if k not in measurement]
    if missing:
        raise ValueError(f"Measurement dictionary lacks required keys: {missing}")
    checkpoint_sha = None
    if checkpoint is not None:
        if checkpoint.get("kind") != prov.KIND_CHECKPOINT:
            raise ValueError("checkpoint must be a 'checkpoint' hash record, "
                             f"got {checkpoint.get('kind')!r}")
        checkpoint_sha = checkpoint["sha256"]
    identity_sha, ext_provenance = None, {}
    if input_identity is not None:
        kind = input_identity.get("kind")
        if kind == prov.KIND_CACHE_CONTENT:
            identity_sha = input_identity["sha256"]
        elif kind == prov.KIND_CACHE_MANIFEST:
            ext_provenance = {
                "cache_manifest_sha256": input_identity["sha256"],
                "input_identity_note": (
                    "Only the cache manifest file was hashed; feature-cache contents were not "
                    "verified, so input_identity_sha256 is null."),
            }
        else:
            raise ValueError("input_identity must be a cache content or manifest record, "
                             f"got {kind!r}")

    engine = {k: measurement.get(k) for k in
              ("peak_memory_reserved_bytes", "memory_baseline_bytes", "memory_note")}
    merged: dict[str, Any] = {}
    if any(v is not None for v in engine.values()):
        merged["benchmark_engine"] = engine
    if ext_provenance:
        merged["provenance"] = ext_provenance
    for key, value in (extensions or {}).items():
        if key in merged:
            raise ValueError(f"extensions key {key!r} is reserved by the record builder")
        merged[key] = copy.deepcopy(value)

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "run_id": run_id,
        "timestamp_utc": timestamp_utc or utc_timestamp(), "experiment_id": experiment_id,
        "config": config, "scope": scope, "device": str(measurement["device"]),
        "backend": backend, "precision": precision_block(config, execution_dtype),
        "batch_size": batch_size, "warmup": measurement["warmup"],
        "repeats": measurement["repeats"],
        "workload": {
            "input_boundary": input_boundary, "input_shapes": copy.deepcopy(input_shapes),
            "input_duration_seconds": input_duration_seconds,
            "checkpoint_sha256": checkpoint_sha, "input_identity_sha256": identity_sha,
        },
        "mean_ms": float(measurement["mean_ms"]), "median_ms": float(measurement["median_ms"]),
        "p50_ms": float(measurement["p50_ms"]), "p95_ms": float(measurement["p95_ms"]),
        "measurement": {"percentile_method": measurement["percentile_method"],
                        "synchronization": measurement["synchronization"]},
        "timing_samples_ms": [float(s) for s in measurement["timing_samples_ms"]],
        "peak_memory_bytes": measurement["peak_memory_bytes"],
        "memory_method": measurement["memory_method"],
        "conversion_overhead_ms": conversion_overhead_ms,
        "conversion_overhead_included_in_latency": conversion_overhead_included_in_latency,
        "environment": copy.deepcopy(environment),
    }
    if merged:
        record["extensions"] = merged
    if notes is not None:
        record["notes"] = notes
    validate_benchmark_records([record], schema_path=schema_path)
    return record


# ------------------------------------------------------------------ benchmark JSONL writing
def _timing_artifact(run_id: str, record_id: str, samples: list[float]) -> dict[str, Any]:
    return {"artifact_type": ARTIFACT_TYPES["samples"], "artifact_version": ARTIFACT_VERSION,
            "run_id": run_id, "record_id": record_id, "unit": "ms", "repeats": len(samples),
            "timing_samples_ms": samples}


def write_benchmark_records(
    records: list[dict[str, Any]], run_dir: str | Path, *, run_id: str | None = None,
    timing_samples_storage: str = "inline", overwrite: bool = False,
    filename: str = RECORDS_NAME, schema_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate every record, then write ``filename`` (JSONL) and optional sample files.

    Order: validate the whole batch; pre-check that no destination exists (unless
    ``overwrite``); write sample files; read each back and compare; re-validate the
    reference-form records; write the JSONL atomically last; read it back. On failure,
    files created by this call are removed and (with ``overwrite``) replaced files are
    restored from the bytes held in memory. Records are never silently dropped or
    deduplicated: duplicates raise :class:`DuplicateRecordError`.

    ``timing_samples_ref`` is relative to ``run_dir`` (and to the JSONL file's directory).
    Returns ``{"records_path", "record_count", "sample_paths", "records"}`` with run-relative paths.
    """
    if timing_samples_storage not in STORAGE_MODES:
        raise ValueError(f"timing_samples_storage must be one of {STORAGE_MODES}")
    if not records:
        raise ValueError("No benchmark records to write")
    run_dir = Path(run_dir)
    validate_relative_path(filename)
    records = copy.deepcopy(records)
    validate_benchmark_records(records, run_id=run_id, schema_path=schema_path)

    sample_files: dict[str, dict[str, Any]] = {}
    if timing_samples_storage == "separate_file":
        for record in records:
            if "timing_samples_ms" not in record:
                raise BenchmarkValidationError(
                    ["separate_file storage needs inline samples to move"])
            rid = record_id_for(record)
            ref = f"{SAMPLES_DIR}/{rid}.json"
            samples = record.pop("timing_samples_ms")
            record.pop("timing_samples_ref", None)
            record["timing_samples_ref"] = ref
            sample_files[ref] = _timing_artifact(record["run_id"], rid, samples)

    targets = {filename: None, **{ref: None for ref in sample_files}}
    paths = {rel: resolve_within(run_dir, rel) for rel in targets}
    if not overwrite:
        for rel, path in paths.items():
            if path.exists() or path.is_symlink():
                raise ArtifactExistsError(f"Refusing to overwrite existing artifact: {rel!r}")
    jsonl = ("\n".join(serialize_jsonl_line(r) for r in records) + "\n").encode("utf-8")

    created: list[Path] = []
    backups: dict[Path, bytes] = {}
    try:
        for rel, content in sample_files.items():
            path = paths[rel]
            if path.exists():
                backups[path] = path.read_bytes()
            else:
                created.append(path)
            write_json_artifact(path, content, overwrite=overwrite)
            stored = load_timing_samples(run_dir, rel)
            if stored != content["timing_samples_ms"]:
                raise ArtifactError(f"Timing-sample read-back mismatch: {rel!r}")
        if sample_files:
            validate_benchmark_records(records, run_id=run_id, run_dir=run_dir,
                                       schema_path=schema_path)
        path = paths[filename]
        if path.exists():
            backups[path] = path.read_bytes()
        else:
            created.append(path)
        atomic_write_bytes(path, jsonl, overwrite=overwrite)
        if path.read_bytes() != jsonl:
            raise ArtifactError(f"JSONL read-back mismatch: {filename!r}")
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        for path, data in backups.items():
            try:
                atomic_write_bytes(path, data, overwrite=True)
            except OSError:
                pass
        raise
    return {"records_path": filename, "record_count": len(records),
            "sample_paths": sorted(sample_files), "records": records}


# ------------------------------------------------------------------ stage-profile artifact
def build_stage_profile_artifact(
    profile: dict[str, Any], *, run_id: str, created_at_utc: str | None = None
) -> dict[str, Any]:
    """Wrap a ``profile_pipeline_stages()`` result as an ``avfuse_stage_profile`` artifact.

    The profile is embedded unchanged under ``profile``. It is **not** a benchmark record: no
    precision configuration, batch size or ``schema_version`` is invented, no stage is
    relabelled, and no end-to-end figure is added.
    """
    if not isinstance(profile, dict) or profile.get("profile_type") != "pipeline_stage_profile":
        raise ValueError("Expected a profile_pipeline_stages() result "
                         "(profile_type 'pipeline_stage_profile')")
    if profile.get("end_to_end_measured") is not False:
        raise ValueError("Stage profiles must not claim an end-to-end measurement")
    for stage in profile.get("stages", []):
        if stage.get("status") == "not_provided" and any(k in stage for k in LATENCY_KEYS):
            raise ValueError(
                f"Stage {stage.get('stage')!r} is not_provided but carries timing data")
    snapshot = copy.deepcopy(profile)
    prov.canonical_json_bytes(snapshot)
    return {
        "artifact_type": ARTIFACT_TYPES["stage_profile"], "artifact_version": ARTIFACT_VERSION,
        "run_id": run_id, "created_at_utc": created_at_utc or utc_timestamp(),
        "profile": snapshot,
    }


# --------------------------------------------------------------------------- run context
@dataclass
class RunContext:
    """Identifiers and paths for one run. Create with :func:`create_run_context`."""

    run_id: str
    run_dir: Path
    experiment_id: str
    created_at_utc: str
    overwrite: bool
    timing_samples_storage: str
    config_sha256: str
    provenance: dict[str, Any]
    environment_report: dict[str, Any]
    git: dict[str, Any]
    environment_report_ref: str | None
    checkpoint: dict[str, Any] | None = None
    input_identity: dict[str, Any] | None = None
    artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)
    finalized: bool = False

    def path(self, rel: str) -> Path:
        return resolve_within(self.run_dir, rel)

    def register(self, rel: str, artifact_type: str, *, required: bool = True) -> None:
        if self.finalized:
            raise RunFinalizedError("Run manifest already written; create a new run")
        validate_relative_path(rel)
        if rel in RESERVED_NAMES:
            raise UnsafePathError(f"{rel!r} is reserved for the run manifest/integrity anchor")
        if not self.path(rel).is_file():
            raise ArtifactError(f"Cannot register missing artifact: {rel!r}")
        self.artifacts[rel] = {"artifact_type": artifact_type, "required": required,
                               "created_at_utc": utc_timestamp()}

    def write_json_artifact(self, rel: str, value: Any, *, artifact_type: str,
                            required: bool = True) -> dict[str, Any]:
        if self.finalized:
            raise RunFinalizedError("Run manifest already written; create a new run")
        if rel in RESERVED_NAMES:
            raise UnsafePathError(f"{rel!r} is reserved for the run manifest/integrity anchor")
        info = write_json_artifact(self.path(rel), value, overwrite=self.overwrite)
        self.register(rel, artifact_type, required=required)
        return info

    def environment_for(self, device: str) -> dict[str, Any]:
        """Schema ``environment`` object for a record measured on ``device``."""
        return environment_block(self.environment_report, device, git=self.git,
                                 environment_report_ref=self.environment_report_ref)

    def build_record(self, measurement: dict[str, Any], **fields: Any) -> dict[str, Any]:
        """:func:`build_benchmark_record` with this run's id, experiment, environment and hashes."""
        fields.setdefault("run_id", self.run_id)
        fields.setdefault("experiment_id", self.experiment_id)
        fields.setdefault("environment", self.environment_for(str(measurement["device"])))
        fields.setdefault("checkpoint", self.checkpoint)
        fields.setdefault("input_identity", self.input_identity)
        return build_benchmark_record(measurement, **fields)

    def write_benchmark_records(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        if self.finalized:
            raise RunFinalizedError("Run manifest already written; create a new run")
        result = write_benchmark_records(
            records, self.run_dir, run_id=self.run_id,
            timing_samples_storage=self.timing_samples_storage, overwrite=self.overwrite)
        self.register(result["records_path"], ARTIFACT_TYPES["records"])
        for rel in result["sample_paths"]:
            self.register(rel, ARTIFACT_TYPES["samples"])
        return result

    def write_stage_profile(self, profile: dict[str, Any],
                            filename: str = STAGE_PROFILE_NAME) -> dict[str, Any]:
        artifact = build_stage_profile_artifact(profile, run_id=self.run_id)
        return self.write_json_artifact(filename, artifact,
                                        artifact_type=ARTIFACT_TYPES["stage_profile"],
                                        required=False)

    def finalize(self) -> dict[str, Any]:
        """Write the manifest and anchor; roll back both if either write fails.

        The two files cannot be committed as one filesystem transaction. This method
        nevertheless restores their pre-call bytes/state on an ordinary write failure.
        A single writer per run directory is required (see the module concurrency note).
        """
        if self.finalized:
            raise RunFinalizedError("Run manifest already written")
        entries = []
        for rel in sorted(self.artifacts):
            meta = self.artifacts[rel]
            path = self.path(rel)
            if not path.is_file():
                raise ArtifactError(
                    f"Registered artifact is missing, refusing to finalize: {rel!r}")
            entries.append({
                "path": rel, "artifact_type": meta["artifact_type"],
                "size_bytes": path.stat().st_size, "sha256": prov.sha256_file(path),
                "created_at_utc": meta["created_at_utc"], "required": meta["required"],
            })
        manifest = {
            "artifact_type": ARTIFACT_TYPES["manifest"], "artifact_version": ARTIFACT_VERSION,
            "run_id": self.run_id, "created_at_utc": utc_timestamp(), "hash_algorithm": "sha256",
            "self_hash_policy": (f"Excludes {MANIFEST_NAME!r} and {ANCHOR_NAME!r}; the manifest's "
                                 f"SHA-256 is stored in {ANCHOR_NAME!r}."),
            "artifacts": entries,
        }
        manifest_path = self.path(MANIFEST_NAME)
        anchor_path = self.path(ANCHOR_NAME)
        targets = (manifest_path, anchor_path)
        previous: dict[Path, bytes | None] = {}
        for target in targets:
            if target.is_symlink():
                raise UnsafePathError(
                    f"Refusing to finalize through a symbolic link: {target.name!r}"
                )
            if target.exists() and not target.is_file():
                raise ArtifactError(f"Finalization target is not a regular file: {target.name!r}")
            previous[target] = target.read_bytes() if target.is_file() else None

        integrity_dir = anchor_path.parent
        integrity_dir_existed = integrity_dir.exists()
        try:
            write_json_artifact(manifest_path, manifest, overwrite=self.overwrite)
            anchor = {
                "artifact_type": ARTIFACT_TYPES["anchor"], "artifact_version": ARTIFACT_VERSION,
                "run_id": self.run_id, "hash_algorithm": "sha256",
                "run_manifest": {
                    "path": MANIFEST_NAME, "sha256": prov.sha256_file(manifest_path),
                    "size_bytes": manifest_path.stat().st_size,
                },
                "artifacts": {e["path"]: e["sha256"] for e in entries},
            }
            write_json_artifact(anchor_path, anchor, overwrite=self.overwrite)
        except BaseException as original_error:
            rollback_errors: list[str] = []
            for target in reversed(targets):
                old_bytes = previous[target]
                try:
                    if old_bytes is None:
                        target.unlink(missing_ok=True)
                    elif self.overwrite:
                        atomic_write_bytes(target, old_bytes, overwrite=True)
                    # With overwrite disabled, an existing destination is never replaced by
                    # atomic_write_bytes, so its original bytes are already intact.
                except (OSError, ArtifactError) as rollback_error:
                    # Preserve the original failure and report rollback trouble.
                    rollback_errors.append(
                        f"{target.name}: {type(rollback_error).__name__}: {rollback_error}"
                    )
            if not integrity_dir_existed:
                try:
                    integrity_dir.rmdir()
                except OSError:
                    pass
            self.finalized = False
            if rollback_errors:
                raise ArtifactError(
                    "Run finalization failed and rollback was incomplete: "
                    + "; ".join(rollback_errors)
                ) from original_error
            raise
        self.finalized = True
        return manifest

    def verify(self) -> dict[str, Any]:
        return verify_run_integrity(self.run_dir)


def _read_config_value(config: dict[str, Any], dotted: str, default: Any) -> Any:
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _resolve_input(path: str | Path | None, repo_root: Path) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else repo_root / p


def create_run_context(
    config: dict[str, Any], *, runs_root: str | Path, repo_root: str | Path | None = None,
    run_id: str | None = None, checkpoint_path: str | Path | None = None,
    cache_dir: str | Path | None = None, cache_manifest_path: str | Path | None = None,
    environment_report: dict[str, Any] | None = None, allow_unset_inputs: bool = False,
    validate_config: bool = True, now: datetime | None = None,
) -> RunContext:
    """Create the run directory and write config snapshot, environment report and provenance.

    ``config`` is a resolved benchmark configuration (``configs/benchmark.example.json``
    layout), validated against the existing ``benchmark_config.schema.json`` unless
    ``validate_config=False``. Paths default to ``config['checkpoint_path']`` and
    ``config['cache_manifest_path']`` (relative paths resolve against ``repo_root``);
    ``cache_dir`` has no config key and must be given explicitly to hash feature-cache
    *contents*. Hashing happens here, before any timing.

    If a hash is requested (``reproducibility.hash_checkpoint`` / ``hash_input_cache``) but its
    input is unset, :class:`MissingInputError` is raised unless ``allow_unset_inputs=True``,
    in which case the provenance records ``status: "not_provided"`` (never a hash). Hashing
    runs before the directory is created, so a failure leaves no half-made run.

    Segment D records ``reproducibility.seed`` and ``torch_num_threads`` but does not apply them.
    ``output.directory`` and ``reproducibility.environment_report_path`` are not used as write
    targets: each run keeps its own copy inside its run directory so the manifest covers it.
    """
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    if validate_config:
        schema = json.loads(CONFIG_SCHEMA_PATH.read_text(encoding="utf-8"))
        problems = sorted(Draft202012Validator(schema).iter_errors(config), key=str)
        if problems:
            raise ArtifactError("Invalid benchmark config: " + "; ".join(
                f"{'/'.join(map(str, e.absolute_path)) or '(root)'}: {e.message[:200]}"
                for e in problems[:5]))
    moment = now or datetime.now(timezone.utc)
    created = utc_timestamp(moment)
    run_id = validate_run_id(run_id) if run_id is not None else generate_run_id(moment)
    overwrite = bool(_read_config_value(config, "output.overwrite", False))
    storage = _read_config_value(config, "output.timing_samples_storage", "inline")
    if storage not in STORAGE_MODES:
        raise ValueError(f"timing_samples_storage must be one of {STORAGE_MODES}")
    experiment_id = config.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise ValueError("config['experiment_id'] must be a non-empty string")

    def flag(key: str, default: bool) -> bool:
        return bool(_read_config_value(config, f"reproducibility.{key}", default))

    config_record = prov.hash_config(config)  # also proves the config is canonicalizable
    hashes: dict[str, Any] = {"config": config_record}

    def input_hash(label: str, enabled: bool, build: Any, given: Path | None) -> dict[str, Any]:
        if not enabled:
            return {"status": "hashing_disabled"}
        if given is None:
            if allow_unset_inputs:
                return {"status": "not_provided"}
            raise prov.MissingInputError(
                f"{label} hashing is enabled but no path was provided "
                "(pass allow_unset_inputs=True for synthetic runs)")
        return build(given)

    ckpt_path = _resolve_input(checkpoint_path or config.get("checkpoint_path"), root)
    manifest_path = _resolve_input(cache_manifest_path or config.get("cache_manifest_path"), root)
    cache_path = _resolve_input(cache_dir, root)
    hashes["checkpoint"] = input_hash("Checkpoint", flag("hash_checkpoint", True),
                                      prov.hash_checkpoint, ckpt_path)
    want_cache = flag("hash_input_cache", True)
    if not want_cache:
        hashes["cache_manifest"] = {"status": "hashing_disabled"}
        hashes["feature_cache_content"] = {"status": "hashing_disabled"}
    else:
        if cache_path is None and manifest_path is None and not allow_unset_inputs:
            raise prov.MissingInputError(
                "Input-cache hashing is enabled but neither cache_dir nor a cache manifest "
                "was provided (pass allow_unset_inputs=True for synthetic runs)")
        hashes["cache_manifest"] = (prov.hash_cache_manifest(manifest_path)
                                    if manifest_path is not None else {"status": "not_provided"})
        if cache_path is not None:
            hashes["feature_cache_content"] = prov.hash_feature_cache(cache_path)
        else:
            hashes["feature_cache_content"] = {"status": "not_provided"}
            if manifest_path is not None:
                hashes["feature_cache_content"]["note"] = (
                    "Only the manifest file was hashed; feature-cache contents were not verified.")

    git = (capture_git_provenance(root) if flag("record_git_commit", True)
           else {"state": "not_recorded", "commit": None, "dirty": None, "warning": None})
    report = (environment_report if environment_report is not None
              else collect_environment_report(root))

    run_dir = Path(runs_root) / run_id
    if run_dir.is_symlink():
        raise UnsafePathError("Run directory must not be a symbolic link")
    try:
        run_dir.mkdir(parents=True, exist_ok=overwrite)
    except FileExistsError:
        raise ArtifactExistsError(f"Run directory already exists: {run_id!r}") from None

    torch_info = report.get("torch") if isinstance(report.get("torch"), dict) else {}
    cuda = torch_info.get("cuda") if isinstance(torch_info.get("cuda"), dict) else {}
    mps = torch_info.get("mps") if isinstance(torch_info.get("mps"), dict) else {}
    ctx = RunContext(
        run_id=run_id, run_dir=run_dir, experiment_id=experiment_id, created_at_utc=created,
        overwrite=overwrite, timing_samples_storage=storage, config_sha256=config_record["sha256"],
        provenance={}, environment_report=report, git=git, environment_report_ref=None,
    )
    if flag("record_environment_report", True):
        dest = ctx.path(ENVIRONMENT_NAME)
        tmp = _prepare_destination(dest, overwrite)
        try:
            write_report(report, tmp)  # the existing environment-report writer
            _commit_temp(tmp, dest, overwrite)
        finally:
            tmp.unlink(missing_ok=True)
        ctx.environment_report_ref = ENVIRONMENT_NAME
        ctx.register(ENVIRONMENT_NAME, ARTIFACT_TYPES["environment"])
    if flag("snapshot_config", True):
        ctx.write_json_artifact(CONFIG_NAME, config, artifact_type=ARTIFACT_TYPES["config"])

    ctx.checkpoint = hashes["checkpoint"] if "sha256" in hashes["checkpoint"] else None
    for key in ("feature_cache_content", "cache_manifest"):
        if "sha256" in hashes[key]:
            ctx.input_identity = hashes[key]
            break
    ctx.provenance = {
        "artifact_type": ARTIFACT_TYPES["provenance"], "artifact_version": ARTIFACT_VERSION,
        "run_id": run_id, "created_at_utc": created, "experiment_id": experiment_id,
        "run_id_note": "Random unique identifier; not a content hash.",
        "hashes": hashes, "git": git,
        "environment": {
            "python_version": (report.get("python") or {}).get("version"),
            "torch_version": torch_info.get("version"),
            "os_platform": (report.get("os") or {}).get("platform"),
            "cuda_available": cuda.get("available"),
            "cuda_runtime_version": cuda.get("runtime_version"),
            "cuda_device_names": [d.get("name") for d in cuda.get("devices", []) if "name" in d],
            "mps_available": mps.get("available"),
            "environment_report_ref": ctx.environment_report_ref,
            "note": "The device actually used is recorded per benchmark record, not here.",
        },
        "reproducibility": {
            "seed": _read_config_value(config, "reproducibility.seed", None),
            "torch_num_threads_requested": _read_config_value(
                config, "reproducibility.torch_num_threads", None),
            "applied_by_segment_d": False,
        },
        "disclaimers": [
            "Environment capability checks are not performance measurements.",
            "A tiny dtype/SDPA check does not establish research-model compatibility.",
            "Fake quantization is simulated INT8, not hardware INT8.",
        ],
    }
    ctx.write_json_artifact(PROVENANCE_NAME, ctx.provenance,
                            artifact_type=ARTIFACT_TYPES["provenance"])
    return ctx


# ------------------------------------------------------------------- integrity verification
def _load_json_object(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, "missing"
    except (OSError, ValueError):
        return None, "unreadable"
    return (value, None) if isinstance(value, dict) else (None, "not_an_object")


def verify_run_integrity(run_dir: str | Path, *, check_references: bool = True) -> dict[str, Any]:
    """Recompute sizes and SHA-256 of every manifest entry; never modifies anything.

    Reports ``valid``, ``missing`` (``required`` flag per entry), ``modified`` (with reasons
    ``size_mismatch`` / ``hash_mismatch``), ``unreadable`` and ``invalid_entries`` (unsafe
    paths) separately, plus the anchor check and ``unlisted_files`` (informational; they do
    not make ``ok`` false). ``ok`` is true only if the manifest and anchor are consistent,
    every required artifact is present, and nothing is modified or invalid.
    """
    run_dir = Path(run_dir)
    report: dict[str, Any] = {
        "report_type": "avfuse_integrity_report", "report_version": "1.0",
        "run_id": None, "ok": False, "manifest_status": "valid", "anchor_status": "valid",
        "valid": [], "missing": [], "modified": [], "unreadable": [], "invalid_entries": [],
        "reference_problems": [], "unlisted_files": [],
    }
    manifest, problem = _load_json_object(run_dir / MANIFEST_NAME)
    if manifest is None or manifest.get("artifact_type") != ARTIFACT_TYPES["manifest"] \
            or not isinstance(manifest.get("artifacts"), list):
        report["manifest_status"] = problem or "invalid_structure"
        report["anchor_status"] = "not_checked"
        return report
    report["run_id"] = manifest.get("run_id")

    seen: set[str] = set()
    for entry in manifest["artifacts"]:
        rel = entry.get("path") if isinstance(entry, dict) else None
        try:
            validate_relative_path(rel)
            if rel in RESERVED_NAMES or rel in seen:
                raise UnsafePathError(f"reserved or duplicate path: {rel!r}")
            seen.add(rel)
            path = resolve_within(run_dir, rel)
        except (UnsafePathError, TypeError) as exc:
            report["invalid_entries"].append({"path": rel if isinstance(rel, str) else None,
                                              "reason": str(exc)})
            continue
        required = bool(entry.get("required", True))
        if not path.is_file():
            report["missing"].append({"path": rel, "required": required})
            continue
        try:
            size = path.stat().st_size
            digest = prov.sha256_file(path)
        except (OSError, prov.ProvenanceError) as exc:
            report["unreadable"].append({"path": rel, "error": type(exc).__name__})
            continue
        reasons = []
        if size != entry.get("size_bytes"):
            reasons.append("size_mismatch")
        if digest != entry.get("sha256"):
            reasons.append("hash_mismatch")
        if reasons:
            report["modified"].append({
                "path": rel, "reasons": reasons, "expected_size_bytes": entry.get("size_bytes"),
                "actual_size_bytes": size, "expected_sha256": entry.get("sha256"),
                "actual_sha256": digest})
        else:
            report["valid"].append(rel)

    anchor, problem = _load_json_object(run_dir / ANCHOR_NAME)
    if anchor is None:
        report["anchor_status"] = (
            "missing" if problem == "missing" else
            "unreadable" if problem == "unreadable" else "invalid_structure"
        )
    else:
        run_manifest_anchor = anchor.get("run_manifest")
        anchor_artifacts = anchor.get("artifacts")
        valid_artifact_map = (
            isinstance(anchor_artifacts, dict)
            and all(
                isinstance(rel, str)
                and isinstance(digest, str)
                and re.fullmatch(r"[0-9a-f]{64}", digest) is not None
                for rel, digest in anchor_artifacts.items()
            )
        )
        valid_run_manifest = (
            isinstance(run_manifest_anchor, dict)
            and run_manifest_anchor.get("path") == MANIFEST_NAME
            and isinstance(run_manifest_anchor.get("sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", run_manifest_anchor.get("sha256", "")) is not None
            and isinstance(run_manifest_anchor.get("size_bytes"), int)
            and not isinstance(run_manifest_anchor.get("size_bytes"), bool)
            and run_manifest_anchor.get("size_bytes", -1) >= 0
        )
        valid_anchor = (
            anchor.get("artifact_type") == ARTIFACT_TYPES["anchor"]
            and anchor.get("artifact_version") == ARTIFACT_VERSION
            and anchor.get("hash_algorithm") == "sha256"
            and isinstance(anchor.get("run_id"), str)
            and valid_run_manifest
            and valid_artifact_map
        )
        if not valid_anchor:
            report["anchor_status"] = "invalid_structure"
        else:
            try:
                manifest_digest = prov.sha256_file(run_dir / MANIFEST_NAME)
                manifest_size = (run_dir / MANIFEST_NAME).stat().st_size
            except (OSError, prov.ProvenanceError):
                manifest_digest = None
                manifest_size = None
            listed = {e["path"]: e.get("sha256") for e in manifest["artifacts"]
                      if isinstance(e, dict) and isinstance(e.get("path"), str)}
            if (
                run_manifest_anchor.get("sha256") != manifest_digest
                or run_manifest_anchor.get("size_bytes") != manifest_size
                or anchor_artifacts != listed
                or anchor.get("run_id") != manifest.get("run_id")
            ):
                report["anchor_status"] = "mismatch"

    if check_references and RECORDS_NAME in report["valid"]:
        try:
            lines = (run_dir / RECORDS_NAME).read_text(encoding="utf-8").splitlines()
            for number, line in enumerate(lines, 1):
                record = json.loads(line)
                ref = record.get("timing_samples_ref")
                if ref is None:
                    continue
                try:
                    samples = load_timing_samples(run_dir, ref)
                    if len(samples) != record.get("repeats"):
                        report["reference_problems"].append(
                            {"line": number, "ref": ref, "reason": "sample count != repeats"})
                except ArtifactError as exc:
                    report["reference_problems"].append(
                        {"line": number, "ref": ref, "reason": str(exc)[:200]})
                except UnsafePathError as exc:
                    report["reference_problems"].append(
                        {"line": number, "ref": ref, "reason": str(exc)[:200]})
        except (OSError, ValueError) as exc:
            report["reference_problems"].append(
                {"line": None, "ref": None, "reason": str(exc)[:200]})

    listed_paths = seen | RESERVED_NAMES
    for current, dirs, files in os.walk(run_dir):
        dirs.sort()
        for name in sorted(files):
            rel = Path(current, name).relative_to(run_dir).as_posix()
            if rel not in listed_paths and not name.endswith(".tmp"):
                report["unlisted_files"].append(rel)

    report["ok"] = (
        report["anchor_status"] == "valid"
        and not any(m["required"] for m in report["missing"])
        and not report["modified"] and not report["unreadable"]
        and not report["invalid_entries"] and not report["reference_problems"]
    )
    report["counts"] = {k: len(report[k]) for k in
                        ("valid", "missing", "modified", "unreadable", "invalid_entries")}
    return report
