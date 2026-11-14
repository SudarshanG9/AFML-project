"""Prediction JSONL generation (one row per ``clip_id x config``), label-free.

Rows carry exactly the required fields of ``contracts/schemas/prediction.schema.json``:
``run_id, clip_id, split, config, score_fake, checkpoint_sha256``. Ground truth reaches the
evaluator only through Member 2's manifest.

Safety properties
* ``val`` is the default split. ``test`` needs a frozen protocol file, ``unseal=True`` and a
  purpose; Member 2's ``open_test`` is called exactly once, *before* any test feature is read.
* All rows are buffered, validated as a whole and published atomically (Segment D's
  ``atomic_write_bytes``). A failure part-way leaves no prediction file.
* A strict run (default) publishes nothing unless every requested mode completed for every
  clip. ``allow_partial=True`` is a development aid and is reported as ``partial``.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from torch import Tensor

from avfuse.member2_data.predictions import validate_predictions
from avfuse.member2_data.seal import open_test
from avfuse.shared.contracts import PRECISION_MODES
from avfuse.shared.jsonl import read_jsonl

from .artifacts import atomic_write_bytes, serialize_jsonl_line
from .feature_loader import SPLITS, ClipRef
from .runner import (
    SWEEP_BLOCKED,
    SWEEP_COMPLETE,
    SWEEP_PARTIAL,
    run_precision_sweep,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
PREDICTION_FIELDS = ("run_id", "clip_id", "split", "config", "score_fake", "checkpoint_sha256")


class CheckpointMismatchError(RuntimeError):
    """Actual checkpoint hash differs from the expected one. No predictions were generated."""


class TestAccessError(PermissionError):
    """Test-split access requirements were not met (raised before any test feature is read)."""

    __test__ = False  # not a pytest class


@dataclass
class PredictionReport:
    status: str                      # complete | partial | blocked
    split: str
    run_id: str
    requested_modes: tuple[str, ...]
    completed_modes: tuple[str, ...]
    unsupported_modes: dict[str, str]
    checkpoint_sha256: str
    n_clips: int                     # clips actually processed
    n_clips_in_split: int            # clips of this split in the manifest
    n_rows: int
    output_path: Path | None         # None unless published
    published: bool
    four_mode_complete: bool
    full_split: bool                 # True only if every clip of the split was processed
    test_access_logged: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["output_path"] = None if self.output_path is None else str(self.output_path)
        d["requested_modes"] = list(self.requested_modes)
        d["completed_modes"] = list(self.completed_modes)
        return d


def load_clip_refs(manifest_path: str | Path, split: str) -> list[ClipRef]:
    """Read a (whole-video) manifest JSONL into label-free :class:`ClipRef` objects.

    Only ``clip_id``, ``split`` and ``feature_path`` are kept; labels, manipulation types
    and source ids never leave this function. Rows of other splits are discarded unread.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    p = Path(manifest_path)
    if not p.is_file():
        raise FileNotFoundError(f"Missing manifest: {p}")
    refs = [ClipRef(clip_id=str(r["clip_id"]), split=str(r["split"]),
                    feature_path=r.get("feature_path"))
            for r in read_jsonl(p) if r.get("split") == split]
    ids = [r.clip_id for r in refs]
    if len(set(ids)) != len(ids):
        raise ValueError("Manifest contains duplicate clip_id values in this split")
    if not refs:
        raise ValueError(f"Manifest has no rows for split {split!r}")
    return refs


def authorize_test_access(
    *,
    split: str,
    protocol_path: str | Path | None,
    unseal: bool,
    purpose: str | None,
    configs: Sequence[str],
) -> bool:
    """Gate for the sealed test split. Returns True if (and only if) access was logged."""
    if split != "test":
        return False
    if protocol_path is None or not Path(protocol_path).is_file():
        raise TestAccessError("Test split needs an existing frozen protocol file "
                              "(calibrate and freeze on val first)")
    if unseal is not True:
        raise TestAccessError("Test split is sealed: pass unseal=True explicitly")
    if not isinstance(purpose, str) or not purpose.strip():
        raise TestAccessError("Test split needs a non-empty purpose string")
    open_test(protocol_path, purpose.strip(), list(configs), True)  # logs once
    return True


def check_checkpoint_hash(actual: str, expected: str | None = None,
                          require_lavdf: bool = False) -> None:
    """Fail loudly on malformed or mismatching hashes; never substitutes either value."""
    if not isinstance(actual, str) or not _SHA256.match(actual):
        raise CheckpointMismatchError("checkpoint_sha256 must be 64 lowercase hex characters, "
                                      f"got {actual!r}")
    if expected is not None and require_lavdf:
        raise ValueError("Pass either expected_checkpoint_sha256 or require_lavdf_hash, not both")
    if require_lavdf:
        from avfuse.member2_data.readiness import DIMODIF_LAVDF_SHA256
        expected = DIMODIF_LAVDF_SHA256
    if expected is not None and actual != expected:
        raise CheckpointMismatchError(
            f"Checkpoint SHA-256 mismatch: actual {actual} != expected {expected}. "
            "No predictions were generated; the checkpoint and expected hash were left as is.")


def _rows_for(run_id: str, clip: ClipRef, sha: str, scores: Mapping[str, Tensor]) -> list[dict]:
    rows = []
    for mode, tensor in scores.items():
        if tensor.numel() != 1:
            raise ValueError("Prediction generation supports one clip (batch size 1) per call; "
                             f"{mode} returned {tuple(tensor.shape)}")
        rows.append({"run_id": run_id, "clip_id": clip.clip_id, "split": clip.split,
                     "config": mode, "score_fake": float(tensor.reshape(-1)[0]),
                     "checkpoint_sha256": sha})
    return rows


def generate_predictions(
    model,
    clips: Sequence[ClipRef],
    feature_provider: Callable[[ClipRef], Mapping[str, Tensor]],
    *,
    run_id: str,
    checkpoint_sha256: str,
    output_path: str | Path,
    split: str = "val",
    modes: Sequence[str] = PRECISION_MODES,
    allow_partial: bool = False,
    overwrite: bool = False,
    max_clips: int | None = None,
    expected_checkpoint_sha256: str | None = None,
    require_lavdf_hash: bool = False,
    protocol_path: str | Path | None = None,
    unseal: bool = False,
    purpose: str | None = None,
) -> PredictionReport:
    """Run ``modes`` on every clip of ``split`` and publish a validated prediction JSONL.

    ``clips`` must be :class:`ClipRef` objects (no labels). ``max_clips`` limits a
    development run to the first N clips (reported as ``full_split=False``).
    The file is published only when the run is ``complete`` (strict) or ``partial``
    (``allow_partial=True``); a blocked run returns its report and writes nothing.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    modes = tuple(modes)
    check_checkpoint_hash(checkpoint_sha256, expected_checkpoint_sha256, require_lavdf_hash)
    selected = [c for c in clips if c.split == split]
    if not selected:
        raise ValueError(f"No clips with split {split!r} were supplied")
    if len({c.clip_id for c in selected}) != len(selected):
        raise ValueError("Duplicate clip_id in the supplied clips")
    n_in_split = len(selected)
    if max_clips is not None:
        if max_clips < 1:
            raise ValueError("max_clips must be >= 1")
        selected = selected[:max_clips]

    # Sealed-test gate: must pass before any feature file is touched.
    logged = authorize_test_access(split=split, protocol_path=protocol_path, unseal=unseal,
                                   purpose=purpose, configs=modes)

    dest = Path(output_path)
    if dest.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing predictions: {dest.name!r}")

    rows: list[dict] = []
    known_unsupported: dict[str, str] = {}
    completed: tuple[str, ...] = ()
    status = SWEEP_COMPLETE
    processed = 0
    for clip in selected:
        features = feature_provider(clip)  # missing/invalid features raise here
        # ModeFailedError propagates; nothing has been published at this point.
        sweep = run_precision_sweep(model, features, modes, allow_partial=allow_partial,
                                    known_unsupported=known_unsupported)
        processed += 1
        for m in sweep.unsupported_modes:
            known_unsupported[m] = sweep.results[m].error_message or "not implemented"
        if sweep.status == SWEEP_BLOCKED:
            status, completed = SWEEP_BLOCKED, sweep.completed_modes
            break  # strict: stop at the first clip; no further features are read
        status = sweep.status
        completed = sweep.completed_modes
        rows.extend(_rows_for(run_id, clip, checkpoint_sha256, sweep.scores))

    four = set(completed) == set(PRECISION_MODES) and status == SWEEP_COMPLETE
    report = PredictionReport(
        status=status, split=split, run_id=run_id, requested_modes=modes,
        completed_modes=completed, unsupported_modes=dict(known_unsupported),
        checkpoint_sha256=checkpoint_sha256, n_clips=processed, n_clips_in_split=n_in_split,
        n_rows=0, output_path=None, published=False, four_mode_complete=four,
        full_split=(processed == n_in_split and status != SWEEP_BLOCKED),
        test_access_logged=logged)
    if status == SWEEP_BLOCKED:
        report.notes.append("Blocked: unsupported mode(s) in a strict run. No predictions "
                            "were written; no mode was scored as 0.")
        return report

    validate_predictions(rows)  # schema + duplicates + single checkpoint hash (Member 2)
    if {r["run_id"] for r in rows} != {run_id}:
        raise ValueError("Internal error: inconsistent run_id")
    data = ("\n".join(serialize_jsonl_line(r) for r in rows) + "\n").encode("utf-8")
    atomic_write_bytes(dest, data, overwrite=overwrite)
    report.n_rows = len(rows)
    report.output_path = dest
    report.published = True
    if status == SWEEP_PARTIAL:
        report.notes.append("PARTIAL development output: unsupported mode(s) "
                            f"{sorted(known_unsupported)} have no rows. Not a four-mode run.")
    if not four:
        report.notes.append("Not a four-mode result.")
    return report
