"""Stable interface wiring Member 2 features to Member 1 model."""
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from avfuse.member1_fusion.model_adapter import ModelAdapter
from avfuse.shared.contracts import PRECISION_MODES


@torch.inference_mode()
def run_four_modes(model: ModelAdapter, features: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Return scores. Actual predictions JSONL and label custody handled elsewhere."""
    return {mode: model.predict_from_cached_features(features, mode).detach().clone()
            for mode in PRECISION_MODES}


# ---------------------------------------------------------------------------------------
# Segment E: per-mode orchestration with explicit status.
# ---------------------------------------------------------------------------------------
STATUS_COMPLETED = "completed"
STATUS_UNSUPPORTED = "unsupported"
STATUS_FAILED = "failed"

SWEEP_COMPLETE = "complete"      # every requested mode completed
SWEEP_BLOCKED = "blocked"        # strict run with at least one unsupported mode
SWEEP_PARTIAL = "partial"        # allow_partial run with at least one unsupported mode

NO_PREDICTION_NOTE = "No prediction was generated for this mode."


class ScoreValidationError(ValueError):
    """A model returned a score tensor that violates the score_fake contract."""


class ModeFailedError(RuntimeError):
    """A mode raised an ordinary runtime error. The original exception is ``__cause__``."""

    def __init__(self, mode: str, cause: BaseException, partial: "SweepResult | None" = None):
        super().__init__(f"Mode {mode!r} failed: {type(cause).__name__}: {cause}")
        self.mode = mode
        self.partial = partial


@dataclass(frozen=True)
class ModeResult:
    mode: str
    status: str
    scores: Tensor | None = None
    error_type: str | None = None
    error_message: str | None = None
    note: str | None = None

    def to_dict(self) -> dict:
        """JSON-safe description (never contains score tensors)."""
        return {"mode": self.mode, "status": self.status, "error_type": self.error_type,
                "error_message": self.error_message, "note": self.note}


@dataclass(frozen=True)
class SweepResult:
    requested_modes: tuple[str, ...]
    results: dict[str, ModeResult]
    allow_partial: bool

    def _modes(self, status: str) -> tuple[str, ...]:
        return tuple(m for m in self.requested_modes if self.results[m].status == status)

    @property
    def completed_modes(self) -> tuple[str, ...]:
        return self._modes(STATUS_COMPLETED)

    @property
    def unsupported_modes(self) -> tuple[str, ...]:
        return self._modes(STATUS_UNSUPPORTED)

    @property
    def failed_modes(self) -> tuple[str, ...]:
        return self._modes(STATUS_FAILED)

    @property
    def status(self) -> str:
        if self.failed_modes:
            return STATUS_FAILED
        if not self.unsupported_modes:
            return SWEEP_COMPLETE
        return SWEEP_PARTIAL if self.allow_partial and self.completed_modes else SWEEP_BLOCKED

    @property
    def four_mode_complete(self) -> bool:
        """True only if all four agreed configurations completed."""
        return set(self.completed_modes) == set(PRECISION_MODES)

    @property
    def scores(self) -> dict[str, Tensor]:
        return {m: r.scores for m, r in self.results.items()
                if r.status == STATUS_COMPLETED and r.scores is not None}

    def summary(self) -> dict:
        return {
            "status": self.status,
            "requested_modes": list(self.requested_modes),
            "completed_modes": list(self.completed_modes),
            "unsupported_modes": {m: self.results[m].error_message
                                  for m in self.unsupported_modes},
            "failed_modes": {m: self.results[m].error_message for m in self.failed_modes},
            "four_mode_complete": self.four_mode_complete,
            "allow_partial": self.allow_partial,
            "modes": {m: self.results[m].to_dict() for m in self.requested_modes},
        }


def infer_batch_size(features: Mapping[str, Tensor]) -> int:
    video = features["video_tokens"]
    return int(video.shape[0]) if video.ndim == 3 else 1


def validate_scores(mode: str, scores: object, batch_size: int) -> Tensor:
    """Check the ``score_fake`` contract. Never clips, rescales or inverts."""
    if not isinstance(scores, Tensor):
        raise ScoreValidationError(f"{mode}: model returned {type(scores).__name__}, "
                                   "expected a Tensor")
    if not scores.is_floating_point():
        raise ScoreValidationError(f"{mode}: scores must be floating point, got {scores.dtype}")
    if scores.shape != (batch_size,):
        raise ScoreValidationError(f"{mode}: expected exactly one score per batch item, shape "
                                   f"({batch_size},), got {tuple(scores.shape)}")
    if not torch.isfinite(scores).all():
        raise ScoreValidationError(f"{mode}: scores contain non-finite values")
    if bool((scores < 0).any()) or bool((scores > 1).any()):
        raise ScoreValidationError(f"{mode}: scores must lie in [0, 1] (not clipped); got "
                                   f"min={float(scores.min())}, max={float(scores.max())}")
    return scores.detach().to("cpu").clone()


@torch.inference_mode()
def run_precision_sweep(
    model: ModelAdapter,
    features: Mapping[str, Tensor],
    modes: Sequence[str] = PRECISION_MODES,
    allow_partial: bool = False,
    known_unsupported: Mapping[str, str] | None = None,
) -> SweepResult:
    """Run ``modes`` on one feature set and report an explicit status for each.

    * ``NotImplementedError`` -> ``unsupported`` (reason kept, no scores).
    * Any other exception -> :class:`ModeFailedError` (never relabelled as unsupported).
    * Output must satisfy :func:`validate_scores`; invalid output is a failure.

    ``known_unsupported`` (mode -> reason) lets multi-clip callers avoid calling a mode
    already known to be unsupported. Every requested mode is attempted otherwise, so a
    strict (default) run reports *all* unsupported modes at once; ``status`` is then
    ``blocked`` and nothing should be published as a completed experiment.
    ``allow_partial=True`` is a development aid; its status is ``partial``.
    """
    requested = tuple(modes)
    if not requested:
        raise ValueError("modes must not be empty")
    bad = [m for m in requested if m not in PRECISION_MODES]
    if bad:
        raise ValueError(f"Unknown precision mode(s) {bad}; use {PRECISION_MODES}")
    if len(set(requested)) != len(requested):
        raise ValueError("modes must not contain duplicates")
    batch = infer_batch_size(features)
    known = dict(known_unsupported or {})
    results: dict[str, ModeResult] = {}
    for mode in requested:
        if mode in known:
            results[mode] = ModeResult(mode, STATUS_UNSUPPORTED, error_type="NotImplementedError",
                                       error_message=known[mode], note=NO_PREDICTION_NOTE)
            continue
        try:
            raw = model.predict_from_cached_features(features, mode)
        except NotImplementedError as exc:
            results[mode] = ModeResult(mode, STATUS_UNSUPPORTED, error_type=type(exc).__name__,
                                       error_message=str(exc) or "not implemented",
                                       note=NO_PREDICTION_NOTE)
            continue
        except Exception as exc:
            results[mode] = ModeResult(mode, STATUS_FAILED, error_type=type(exc).__name__,
                                       error_message=str(exc))
            raise ModeFailedError(
                mode, exc, SweepResult(tuple(results), dict(results), allow_partial)) from exc
        try:
            scores = validate_scores(mode, raw, batch)
        except ScoreValidationError as exc:
            results[mode] = ModeResult(mode, STATUS_FAILED, error_type=type(exc).__name__,
                                       error_message=str(exc))
            raise ModeFailedError(
                mode, exc, SweepResult(tuple(results), dict(results), allow_partial)) from exc
        results[mode] = ModeResult(mode, STATUS_COMPLETED, scores=scores)
    return SweepResult(requested, results, allow_partial)
