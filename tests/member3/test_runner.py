"""Segment E runner orchestration with mock adapters. Synthetic, not detector performance."""
import re

import pytest
import torch

from avfuse.member1_fusion.diagnostic_model import DiagnosticModel
from avfuse.member1_fusion.model_adapter import ModelAdapter
from avfuse.member3_runtime.runner import (
    ModeFailedError,
    ScoreValidationError,
    run_four_modes,
    run_precision_sweep,
    validate_scores,
)
from avfuse.shared.contracts import PRECISION_MODES

FEATURES = {"video_tokens": torch.randn(1, 5, 8), "audio_tokens": torch.randn(1, 3, 8)}


class Mock(ModelAdapter):
    def __init__(self, supported=PRECISION_MODES, outputs=None, boom=None):
        self.supported, self.outputs, self.boom = set(supported), outputs or {}, boom
        self.calls, self.inputs, self.inference = [], [], []

    def predict_from_cached_features(self, features, mode):
        self.calls.append(mode)
        self.inputs.append(features)
        self.inference.append(torch.is_inference_mode_enabled())
        if self.boom == mode:
            raise RuntimeError("kernel exploded")
        if mode not in self.supported:
            raise NotImplementedError(f"{mode} not integrated")
        return self.outputs.get(mode, torch.tensor([0.25]))


def test_four_supported_modes_complete():
    m = Mock()
    r = run_precision_sweep(m, FEATURES)
    assert r.status == "complete" and r.four_mode_complete
    assert r.completed_modes == PRECISION_MODES and not r.unsupported_modes
    assert set(r.scores) == set(PRECISION_MODES)
    assert all(v.shape == (1,) and 0 <= float(v) <= 1 for v in r.scores.values())


def test_same_input_and_inference_mode_for_every_mode():
    m = Mock()
    run_precision_sweep(m, FEATURES)
    assert all(x is FEATURES for x in m.inputs) and len(m.inputs) == 4
    assert all(m.inference)


def test_reference_only_model_strict_four_mode_is_blocked():
    m = Mock(supported=("reference",))
    r = run_precision_sweep(m, FEATURES)
    assert r.status == "blocked" and not r.four_mode_complete
    assert r.unsupported_modes == ("low_precision", "restore_value_output", "restore_query")
    assert r.completed_modes == ("reference",)
    for mode in r.unsupported_modes:
        res = r.results[mode]
        assert res.status == "unsupported" and res.scores is None
        assert res.error_type == "NotImplementedError" and "not integrated" in res.error_message
        assert "No prediction was generated" in res.note
    assert "low_precision" not in r.scores  # never a fake 0


def test_explicit_reference_only_request_completes():
    r = run_precision_sweep(Mock(supported=("reference",)), FEATURES, modes=("reference",))
    assert r.status == "complete" and r.completed_modes == ("reference",)
    assert not r.four_mode_complete


def test_partial_mode_lists_supported_and_unsupported():
    r = run_precision_sweep(Mock(supported=("reference",)), FEATURES, allow_partial=True)
    assert r.status == "partial" and not r.four_mode_complete
    s = r.summary()
    assert s["completed_modes"] == ["reference"] and len(s["unsupported_modes"]) == 3
    assert s["four_mode_complete"] is False


def test_partial_with_nothing_supported_is_blocked():
    r = run_precision_sweep(Mock(supported=()), FEATURES, allow_partial=True)
    assert r.status == "blocked"


def test_known_unsupported_not_called_again():
    m = Mock(supported=("reference",))
    run_precision_sweep(m, FEATURES, known_unsupported={"low_precision": "pending"},
                        allow_partial=True)
    assert "low_precision" not in m.calls


def test_runtime_failure_propagates_and_is_not_unsupported():
    m = Mock(boom="low_precision")
    with pytest.raises(ModeFailedError) as ei:
        run_precision_sweep(m, FEATURES)
    assert ei.value.mode == "low_precision"
    assert isinstance(ei.value.__cause__, RuntimeError)
    assert ei.value.partial.results["low_precision"].status == "failed"
    assert ei.value.partial.unsupported_modes == ()


@pytest.mark.parametrize("bad,match", [
    (torch.tensor([1.5]), r"\[0, 1\]"),
    (torch.tensor([-0.1]), r"\[0, 1\]"),
    (torch.tensor([float("nan")]), "non-finite"),
    (torch.tensor([float("inf")]), "non-finite"),
    (torch.tensor([0.2, 0.3]), "one score per batch item"),
    (torch.tensor(0.2), "one score per batch item"),
    (torch.tensor([1]), "floating"),
])
def test_invalid_outputs_are_failures_not_clipped(bad, match):
    m = Mock(outputs={"reference": bad})
    with pytest.raises(ModeFailedError) as ei:
        run_precision_sweep(m, FEATURES, modes=("reference",))
    assert isinstance(ei.value.__cause__, ScoreValidationError)
    assert re.search(match, str(ei.value.__cause__))


def test_batch_of_two_needs_two_scores():
    feats = {"video_tokens": torch.randn(2, 5, 8), "audio_tokens": torch.randn(2, 3, 8)}
    ok = Mock(outputs={"reference": torch.tensor([0.1, 0.9])})
    assert run_precision_sweep(ok, feats, modes=("reference",)).scores["reference"].shape == (2,)
    with pytest.raises(ModeFailedError):
        run_precision_sweep(Mock(), feats, modes=("reference",))


def test_validate_scores_keeps_orientation_and_values():
    out = validate_scores("reference", torch.tensor([0.9]), 1)
    assert float(out) == pytest.approx(0.9)


def test_bad_mode_requests_rejected():
    with pytest.raises(ValueError, match="Unknown precision mode"):
        run_precision_sweep(Mock(), FEATURES, modes=("int4",))
    with pytest.raises(ValueError):
        run_precision_sweep(Mock(), FEATURES, modes=())
    with pytest.raises(ValueError, match="duplicates"):
        run_precision_sweep(Mock(), FEATURES, modes=("reference", "reference"))


def test_backward_compatible_run_four_modes_with_diagnostic_model():
    torch.manual_seed(0)
    model = DiagnosticModel(16).eval()
    feats = {"audio_tokens": torch.randn(1, 12, 16), "video_tokens": torch.randn(1, 8, 16)}
    out = run_four_modes(model, feats)
    assert tuple(out) == PRECISION_MODES
    r = run_precision_sweep(model, feats)
    assert r.four_mode_complete
