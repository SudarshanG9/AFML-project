"""Glue between prediction generation, model benchmarking and a Segment D run.

This is library code (the user-facing CLI is Segment F). A run is finalized and verified
only when every requested mode completed. A blocked run (unsupported mode) keeps a
``execution_summary.json`` describing why, writes no predictions or benchmark records, and
is NOT finalized. An exception also leaves a failure summary (best effort) and is re-raised.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from avfuse.shared.contracts import PRECISION_MODES

from . import provenance as prov
from .artifacts import ArtifactError, RunContext
from .feature_loader import (
    FORMAT_DIMODIF_NPZ,
    ClipRef,
    make_feature_provider,
)
from .model_benchmark import (
    INPUT_BOUNDARY,
    SCOPE,
    ConversionBenchmark,
    benchmark_precision_modes,
)
from .predictions import generate_predictions, load_clip_refs

PREDICTIONS_NAME = "predictions.jsonl"
SUMMARY_NAME = "execution_summary.json"
PREDICTIONS_ARTIFACT_TYPE = "avfuse_predictions"
SUMMARY_ARTIFACT_TYPE = "avfuse_execution_summary"
SUMMARY_VERSION = "1.0"


class MissingInputError(FileNotFoundError):
    """A required real-data input path is missing. No substitute data is ever generated."""


def build_dimodif_adapter(upstream_root: str | Path, checkpoint_name: str | None, device: str):
    """Instantiate Member 1's real adapter. ``checkpoint_name`` has no silent default."""
    if checkpoint_name is None:
        raise ValueError("checkpoint_name must be given explicitly ('lavdf' or 'avdeepfake1m')")
    root = Path(upstream_root)
    if not root.is_dir():
        raise MissingInputError(f"DiMoDif upstream root not found: {root}")
    from avfuse.member1_fusion.dimodif_adapter import DiMoDifAdapter  # heavy/optional import
    return DiMoDifAdapter(upstream_root=root, checkpoint_name=checkpoint_name, device=device)


def _write_summary(ctx: RunContext, summary: dict[str, Any]) -> None:
    ctx.write_json_artifact(SUMMARY_NAME, summary, artifact_type=SUMMARY_ARTIFACT_TYPE)


def execute_experiment(
    model: Any,
    ctx: RunContext,
    clips: Sequence[ClipRef],
    feature_provider: Callable[[ClipRef], Mapping[str, torch.Tensor]],
    *,
    checkpoint_sha256: str,
    split: str = "val",
    modes: Sequence[str] = PRECISION_MODES,
    allow_partial: bool = False,
    max_clips: int | None = None,
    device: str = "cpu",
    warmup: int = 10,
    repeats: int = 50,
    percentile_method: str = "linear",
    measure_memory: bool = True,
    benchmark: bool = True,
    benchmark_clip_id: str | None = None,
    conversion_benchmarks: Mapping[str, ConversionBenchmark] | None = None,
    expected_checkpoint_sha256: str | None = None,
    require_lavdf_hash: bool = False,
    protocol_path: str | Path | None = None,
    unseal: bool = False,
    purpose: str | None = None,
    feature_description: Mapping[str, Any] | None = None,
    data_origin: str = "unspecified",
) -> dict[str, Any]:
    """Predictions -> benchmark -> summary -> (finalize, verify) for one run.

    Returns the execution summary plus ``finalized`` (True only for a completed run).
    ``data_origin`` is free text (e.g. ``"synthetic_test"``) recorded verbatim.
    """
    modes = tuple(modes)
    summary: dict[str, Any] = {
        "artifact_type": SUMMARY_ARTIFACT_TYPE, "artifact_version": SUMMARY_VERSION,
        "run_id": ctx.run_id, "experiment_id": ctx.experiment_id, "split": split,
        "requested_modes": list(modes), "checkpoint_sha256": checkpoint_sha256,
        "features": dict(feature_description or {}), "data_origin": data_origin,
        "labels_accessed": False,
        "disclaimers": [
            "Cached-feature detector inference only; not raw-video end-to-end latency.",
            "No accuracy is computed or claimed here; evaluation belongs to Member 2.",
            "Simulated INT8 modes (when implemented) are not hardware INT8.",
        ],
    }
    try:
        if split == "test" and benchmark:
            raise ValueError(
                "Model benchmarking must use a validation sample, never the sealed test split; "
                "run the test prediction pass with benchmark=False after validation benchmarking."
            )
        pred_path = ctx.path(PREDICTIONS_NAME)
        report = generate_predictions(
            model, clips, feature_provider, run_id=ctx.run_id,
            checkpoint_sha256=checkpoint_sha256, output_path=pred_path, split=split,
            modes=modes, allow_partial=allow_partial, overwrite=ctx.overwrite,
            max_clips=max_clips, expected_checkpoint_sha256=expected_checkpoint_sha256,
            require_lavdf_hash=require_lavdf_hash, protocol_path=protocol_path,
            unseal=unseal, purpose=purpose)
        summary["predictions"] = report.to_dict()
        if not report.published:
            summary.update(status="blocked", four_mode_complete=False,
                           message="Unsupported mode(s) in a strict run; nothing was published "
                                   "as a completed experiment.")
            _write_summary(ctx, summary)
            return dict(summary, finalized=False)
        ctx.register(PREDICTIONS_NAME, PREDICTIONS_ARTIFACT_TYPE)
        summary["predictions_sha256"] = prov.sha256_file(pred_path)

        status = report.status
        if status == "complete" and not report.full_split:
            status = "partial"
            summary["dataset_complete"] = False
            summary["incomplete_reason"] = (
                "Only a subset of the requested split was processed; this development run "
                "must not be finalized as a complete experiment."
            )
        else:
            summary["dataset_complete"] = report.full_split
        four = report.four_mode_complete
        if benchmark:
            sample = next((c for c in clips if c.split == split
                           and (benchmark_clip_id is None or c.clip_id == benchmark_clip_id)),
                          None)
            if sample is None:
                raise ValueError("benchmark_clip_id is not a clip of the requested split")
            bench = benchmark_precision_modes(
                model, feature_provider(sample), ctx=ctx, device=device, modes=modes,
                warmup=warmup, repeats=repeats, percentile_method=percentile_method,
                measure_memory=measure_memory, allow_partial=allow_partial,
                checkpoint_sha256=checkpoint_sha256,
                conversion_benchmarks=conversion_benchmarks)
            summary["benchmark"] = {
                "scope": SCOPE, "input_boundary": INPUT_BOUNDARY, "status": bench.status,
                "benchmarked_modes": list(bench.benchmarked_modes),
                "backend": "none",
                "sample_clip_id": sample.clip_id,
                "input_identity_sha256": bench.input_identity_sha256,
                "record_count": len(bench.records)}
            four = four and bench.four_mode_complete
            if bench.status == "blocked":
                status = "blocked"
            elif bench.status == "partial" and status == "complete":
                status = "partial"
        summary.update(status=status, four_mode_complete=four,
                       finalize_policy="run is finalized only when status == 'complete'")
        _write_summary(ctx, summary)
    except BaseException as exc:
        if not ctx.finalized:
            try:
                fail = dict(summary, status="failed", four_mode_complete=False,
                            error={"type": type(exc).__name__, "message": str(exc)})
                ctx.write_json_artifact("failure_summary.json", fail,
                                        artifact_type=SUMMARY_ARTIFACT_TYPE, required=False)
            except (OSError, ValueError, TypeError, ArtifactError):
                pass  # never mask the original error
        raise

    # Finalize only a run whose requested modes all completed (partial dev runs stay open).
    if status == "complete":
        ctx.finalize()
        ctx.verify()
        return dict(summary, finalized=True)
    return dict(summary, finalized=False)


def run_dimodif_reference(
    ctx: RunContext,
    *,
    upstream_root: str | Path,
    checkpoint_name: str | None,
    manifest_path: str | Path,
    feature_root: str | Path,
    device: str = "cpu",
    feature_format: str = FORMAT_DIMODIF_NPZ,
    split: str = "val",
    max_clips: int | None = None,
    expected_checkpoint_sha256: str | None = None,
    require_lavdf_hash: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Real reference-mode path: whole-video manifest + NPZ features + DiMoDifAdapter.

    Paths are caller-supplied (nothing hardcoded). Missing inputs raise
    :class:`MissingInputError`; no data is fabricated. Only ``reference`` is requested.
    """
    if not Path(manifest_path).is_file():
        raise MissingInputError(f"Whole-video manifest not found: {manifest_path}")
    if not Path(feature_root).is_dir():
        raise MissingInputError(f"Feature root not found: {feature_root}")
    clips = load_clip_refs(manifest_path, split)
    provider = make_feature_provider(feature_format, feature_root)
    adapter = build_dimodif_adapter(upstream_root, checkpoint_name, device)
    if checkpoint_name == "lavdf":
        from avfuse.member2_data.readiness import DIMODIF_LAVDF_SHA256

        if (expected_checkpoint_sha256 is not None
                and expected_checkpoint_sha256 != DIMODIF_LAVDF_SHA256):
            raise ValueError(
                "The 'lavdf' checkpoint must be checked against Member 2's agreed "
                "DIMODIF_LAVDF_SHA256; a different expected hash is not accepted."
            )
        expected_checkpoint_sha256 = DIMODIF_LAVDF_SHA256
        require_lavdf_hash = False
    return execute_experiment(
        adapter, ctx, clips, provider, checkpoint_sha256=adapter.checkpoint_sha256,
        split=split, modes=("reference",), max_clips=max_clips, device=device,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        require_lavdf_hash=require_lavdf_hash,
        feature_description={"format": feature_format, "feature_root": str(feature_root),
                             "manifest": str(manifest_path),
                             "manifest_sha256": prov.sha256_file(Path(manifest_path)),
                             "checkpoint_name": checkpoint_name},
        data_origin="real_cached_features", **kwargs)
