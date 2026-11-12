"""Independent TEST_FIXTURE numerical checks; no real data or checkpoint inference."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib
import math

import pytest
import torch
from torch import nn

from avfuse.member1_fusion.attention import fusion_attention
from avfuse.member1_fusion.precision import fake_quant_int8
from avfuse.member1_fusion import dimodif_intervention as intervention
from avfuse.member1_fusion import dayc_precision_audit as audit

MODES = ("reference", "low_precision", "restore_value_output", "restore_query")


def independent_quantizer(x):
    maximum = x.abs().max()
    if maximum == 0:
        return x.clone()
    scale = maximum / 127
    return torch.clamp(torch.round(x / scale), min=-127, max=127) * scale


def restored_names(mode):
    return {"reference": {"Q", "K", "V", "attention_output"},
            "low_precision": set(), "restore_value_output": {"V", "attention_output"},
            "restore_query": {"Q"}}[mode]


def test_hand_checked_quantizer_and_zero():
    x = torch.tensor([[-127., -0.51, -0.5, 0.], [0.49, 0.5, 0.51, 63.6]])
    expected = torch.tensor([[-127., -1., 0., 0.], [0., 0., 1., 64.]])
    assert torch.equal(fake_quant_int8(x), expected)
    assert torch.equal(independent_quantizer(x), expected)
    assert torch.equal(fake_quant_int8(torch.zeros(2, 7)), torch.zeros(2, 7))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("backend", ["sdpa", "dense"])
@pytest.mark.parametrize("mask_kind", ["none", "boolean", "additive", "fully_blocked_row"])
def test_independent_four_boundary_oracles(mode, backend, mask_kind):
    generator = torch.Generator().manual_seed(1826)
    operands = [torch.randn(1, 2, 7, 5, generator=generator) for _ in range(3)]
    snapshots = [x.clone() for x in operands]
    permitted = torch.ones(7, 7, dtype=torch.bool)
    permitted[:, -2:] = False
    if mask_kind == "fully_blocked_row":
        permitted[0] = False
    mask = None if mask_kind == "none" else permitted
    if mask_kind == "additive":
        mask = torch.zeros(7, 7).masked_fill(~permitted, float("-inf"))
    preserved = restored_names(mode)
    q, k, v = [x if name in preserved else independent_quantizer(x)
               for name, x in zip(("Q", "K", "V"), operands)]
    if backend == "sdpa":
        raw = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.)
    else:
        logits = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(q.shape[-1]))
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf")) if mask.dtype == torch.bool else logits + mask
        raw = torch.nan_to_num(torch.softmax(logits, dim=-1), nan=0.) @ v
    expected = raw if "attention_output" in preserved else independent_quantizer(raw)
    seen = []
    def check_operands(name, before, after):
        golden = before if name in preserved else independent_quantizer(before)
        torch.testing.assert_close(after, golden, atol=0, rtol=0)
        assert after.dtype == torch.float32
        seen.append(name)
    actual = fusion_attention(*operands, mode=mode, backend=backend, attn_mask=mask,
                              diagnostic_hook=check_operands)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert seen == ["Q", "K", "V", "attention_output"]
    for before, after in zip(snapshots, operands):
        assert torch.equal(before, after)


class FixtureDetector(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for _ in range(5):
            layer = nn.Module()
            layer.self_attn = nn.MultiheadAttention(8, 2, batch_first=True, dropout=0.)
            layers.append(layer)
        self.transformer_encoder = nn.Module()
        self.transformer_encoder.layers = nn.ModuleList(layers)

    def forward(self, x):
        mask = torch.zeros(x.shape[1], x.shape[1], dtype=torch.bool)
        mask[:, -2:] = True  # Keep padded tokens and their original blocking semantics.
        for layer in self.transformer_encoder.layers:
            x, _ = layer.self_attn(x, x, x, attn_mask=mask, need_weights=False)
        return x


@pytest.fixture
def fixture_adapter(tmp_path):
    fastpath = torch.backends.mha.get_fastpath_enabled()
    with torch.random.fork_rng():
        torch.manual_seed(87)
        torch.backends.mha.set_fastpath_enabled(False)
        model = FixtureDetector().eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        checkpoint = tmp_path / "TEST_FIXTURE_not_pretrained.pth"
        checkpoint.write_bytes(b"TEST_FIXTURE; no pretrained checkpoint")
        def forward_reference(features):
            logits = model(features["tokens"])
            return {"logits": logits, "score_fake": logits.mean((1, 2)).sigmoid()}
        def forward_with_mode(features, mode, trace):
            with intervention.intervene_dimodif_attention(model, mode, trace=trace):
                return forward_reference(features)
        adapter = SimpleNamespace(model=model, checkpoint_path=checkpoint,
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            config={"nhead": 2, "d_model": 8}, forward_reference=forward_reference,
            forward_with_mode=forward_with_mode)
        features = {"tokens": torch.randn(1, 9, 8)}
        try:
            yield adapter, features
        finally:
            torch.backends.mha.set_fastpath_enabled(fastpath)


@torch.inference_mode()
@pytest.mark.parametrize("mode", MODES)
def test_five_layer_trace_with_independent_operand_equality(fixture_adapter, monkeypatch, mode):
    adapter, features = fixture_adapter
    original = intervention.fusion_attention
    assertions = []
    def oracle_checked_attention(*args, **kwargs):
        original_hook = kwargs["diagnostic_hook"]
        def hook(name, before, after):
            golden = before if name in restored_names(mode) else independent_quantizer(before)
            torch.testing.assert_close(after, golden, atol=0, rtol=0)
            assertions.append(name)
            original_hook(name, before, after)
        kwargs["diagnostic_hook"] = hook
        return original(*args, **kwargs)
    monkeypatch.setattr(intervention, "fusion_attention", oracle_checked_attention)
    before = audit.check_model_invariants(adapter, checkpoint_sha256=adapter.checkpoint_sha256)
    inputs_before = features["tokens"].clone()
    methods = [layer.self_attn.forward for layer in adapter.model.transformer_encoder.layers]
    trace = []
    adapter.forward_with_mode(features, mode, trace)
    result = audit.validate_precision_trace(trace, mode=mode)
    assert result["rows"] == 20 and len(assertions) == 20
    after = audit.check_model_invariants(adapter, checkpoint_sha256=adapter.checkpoint_sha256)
    assert before["state_dict_sha256"] == after["state_dict_sha256"]
    assert methods == [layer.self_attn.forward for layer in adapter.model.transformer_encoder.layers]
    assert torch.equal(inputs_before, features["tokens"])


@torch.inference_mode()
def test_synthetic_parity_and_injected_cleanup(fixture_adapter):
    adapter, features = fixture_adapter
    report = audit.reference_parity_on_validation_feature(adapter, features)
    assert report["state_unchanged"] and report["method_cleanup_success_and_exception"]
    # This is fixture parity, explicitly not real DiMoDif validation parity.
    assert report["scope"].endswith("not population validation")


@torch.inference_mode()
def test_error_during_forward_restores_methods_and_releases_audit(fixture_adapter, monkeypatch):
    adapter, features = fixture_adapter
    original = adapter.forward_with_mode
    def failing(features, mode, trace):
        with intervention.intervene_dimodif_attention(adapter.model, mode, trace):
            raise RuntimeError("TEST_FIXTURE injected forward error")
    monkeypatch.setattr(adapter, "forward_with_mode", failing)
    with pytest.raises(RuntimeError, match="injected forward error"):
        audit.reference_parity_on_validation_feature(adapter, features)
    audit.check_model_invariants(adapter, checkpoint_sha256=adapter.checkpoint_sha256)
    monkeypatch.setattr(adapter, "forward_with_mode", original)
    assert audit.reference_parity_on_validation_feature(adapter, features)["status"] == "PASS"


def test_concurrent_same_model_audit_rejected(fixture_adapter):
    adapter, _ = fixture_adapter
    with audit._exclusive_audit(adapter.model):
        with pytest.raises(RuntimeError, match="Concurrent or nested"):
            with audit._exclusive_audit(adapter.model):
                pytest.fail("nested audit allowed")


@torch.inference_mode()
@pytest.mark.parametrize("defect", ["duplicate", "mode", "shape", "dtype", "nan", "preserved_Q", "rounding"])
def test_trace_summary_rejects_defects(fixture_adapter, defect):
    adapter, features = fixture_adapter
    trace = []
    adapter.forward_with_mode(features, "restore_query", trace)
    if defect == "duplicate":trace[-1] = deepcopy(trace[0])
    elif defect == "mode":trace[0]["mode"] = "reference"
    elif defect == "shape":trace[0]["shape"] = [1, 2, 0, 4]
    elif defect == "dtype":trace[0]["dtype"] = "torch.int8"
    elif defect == "nan":trace[1]["max_absolute_error"] = float("nan")
    elif defect == "preserved_Q":trace[0]["fraction_changed"] = .1
    else:trace[1]["max_absolute_error"] = 100.
    with pytest.raises(ValueError):
        audit.validate_precision_trace(trace, mode="restore_query")


def test_zero_error_quantized_trace_allowed():
    trace = [{"layer": layer, "tensor": name, "mode": "restore_query", "shape": [1, 2, 3, 4],
              "dtype": "torch.float32", "max_abs": 0., "scale": 1e-40,
              "mean_absolute_error": 0., "max_absolute_error": 0.,
              "relative_l2_error": 0., "fraction_changed": 0.}
             for layer in range(5) for name in ("Q", "K", "V", "attention_output")]
    assert audit.validate_precision_trace(trace, mode="restore_query")["status"] == "PASS"
    with pytest.raises(ValueError):audit.expected_mode_boundaries("fifth_mode")


@pytest.mark.parametrize("defect", ["training", "trainable", "checkpoint", "heads", "patch", "layer_count"])
def test_model_invariants_fail_closed(fixture_adapter, defect):
    adapter, _ = fixture_adapter
    if defect == "training":adapter.model.train()
    elif defect == "trainable":next(adapter.model.parameters()).requires_grad_(True)
    elif defect == "checkpoint":adapter.checkpoint_sha256 = "0" * 64
    elif defect == "heads":adapter.config["nhead"] = 4
    elif defect == "layer_count":adapter.model.transformer_encoder.layers = adapter.model.transformer_encoder.layers[:4]
    else:
        with intervention.intervene_dimodif_attention(adapter.model, "restore_query"):
            with pytest.raises(ValueError):
                audit.check_model_invariants(adapter, checkpoint_sha256=adapter.checkpoint_sha256)
        return
    with pytest.raises(ValueError):
        audit.check_model_invariants(adapter, checkpoint_sha256=adapter.checkpoint_sha256)



from scripts.dayC_member1 import query_audit


def test_pilot_missing_c0_stops_before_features(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("feature/model access before C0")
    monkeypatch.setattr(query_audit.preflight, "_read_val_manifest", forbidden)
    report = query_audit.pilot_entry(tmp_path, Path("absent.json"), Path("model"), Path("test/features.npz"))
    assert report["status"] == "BLOCKED"
    assert "C0 acknowledgement" in report["blockers"][0]
    assert not report["feature_contents_read"]


@pytest.mark.parametrize("release", [{"status": "BLOCKED"}, {"status": "READY", "split": "test"}])
def test_pilot_executor_denies_unreleased_or_test_input(tmp_path, release, monkeypatch):
    from scripts import dayB_inference
    def forbidden(*args, **kwargs):
        pytest.fail("model constructed before release")
    monkeypatch.setattr(dayB_inference, "create_adapter", forbidden)
    with pytest.raises(PermissionError):
        query_audit._run_pilot(tmp_path, tmp_path, release)


def synthetic_pilot_release(tmp_path):
    path = tmp_path / "TEST_FIXTURE_features.npz"
    path.write_bytes(b"synthetic fixture bytes, not an encoder cache")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"status": "READY", "split": "val", "clip_id": "TEST_FIXTURE",
            "input_paths": {"fixture": path.name}, "input_hashes": {"fixture": digest},
            "feature_path": path.name, "feature_sha256": digest}


@torch.inference_mode()
def test_pilot_synthetic_execution_reuses_adapter_and_tensors(fixture_adapter, tmp_path, monkeypatch):
    from scripts import dayB_inference
    adapter, features = fixture_adapter
    calls = []
    def create(args):
        calls.append("create")
        return adapter
    def load(path):
        calls.append("load")
        return features
    monkeypatch.setattr(dayB_inference, "create_adapter", create)
    monkeypatch.setattr(dayB_inference, "load_features", load)
    monkeypatch.setattr(query_audit.preflight, "CHECKPOINT", adapter.checkpoint_sha256)
    report = query_audit._run_pilot(tmp_path, tmp_path, synthetic_pilot_release(tmp_path))
    assert calls == ["create", "load"]
    assert report["status"] == "PILOT_COMPLETE_PENDING_C8A"
    assert set(report["modes"]) == {"low_precision", "restore_value_output", "restore_query"}
    assert all(len(value["trace"]) == 20 for value in report["modes"].values())
    assert report["state_and_inputs_unchanged"]


@torch.inference_mode()
@pytest.mark.parametrize("defect", ["nonfinite_score", "input_mutation", "model_mutation"])
def test_synthetic_pilot_failures_never_claim_completion(fixture_adapter, tmp_path, monkeypatch, defect):
    from scripts import dayB_inference
    adapter, features = fixture_adapter
    original = adapter.forward_with_mode
    def altered(features, mode, trace):
        result = original(features, mode, trace)
        if mode == "restore_query":
            if defect == "nonfinite_score":result["score_fake"] = torch.tensor([float("nan")])
            elif defect == "input_mutation":features["tokens"].add_(1)
            else:next(adapter.model.parameters()).add_(1)
        return result
    monkeypatch.setattr(adapter, "forward_with_mode", altered)
    monkeypatch.setattr(dayB_inference, "create_adapter", lambda args: adapter)
    monkeypatch.setattr(dayB_inference, "load_features", lambda path: features)
    monkeypatch.setattr(query_audit.preflight, "CHECKPOINT", adapter.checkpoint_sha256)
    with pytest.raises((ValueError, RuntimeError)):
        query_audit._run_pilot(tmp_path, tmp_path, synthetic_pilot_release(tmp_path))


def test_pilot_cache_or_runtime_block_precedes_requested_feature(tmp_path, monkeypatch):
    # Isolated metadata-gate ordering test. No real assets, C0 ACK or bypass CLI.
    fake = tmp_path / "fixture"
    fake.write_text("TEST_FIXTURE")
    monkeypatch.setattr(query_audit, "_c0_release", lambda root: {"TEST_FIXTURE": True})
    monkeypatch.setattr(query_audit, "require_frozen_inputs", lambda *args: None)
    monkeypatch.setattr(query_audit, "require_sha256", lambda *args: None)
    spec = {"local_inputs": {name: {"path": "fixture", "sha256": "0" * 64} for name in
           ["checkpoint", "upstream_source", "validation_manifest", "ready_val_manifest",
            "environment_reference", "predictions_reference", "predictions_low_precision", "predictions_restore_value_output"]},
            "expected_baseline": {"branch": "member1"}}
    keys = ("python", "platform", "versions", "torch_threads", "torch_interop_threads", "cuda_available")
    accepted = dict.fromkeys(keys, "accepted")
    monkeypatch.setattr(query_audit, "read_strict_json", lambda path:
                        {"reference_environment": accepted} if path == fake else spec)
    source = tmp_path / "model/src/models.py"
    source.parent.mkdir(parents=True)
    source.write_text("# TEST_FIXTURE")
    spec["local_inputs"]["upstream_source"]["path"] = "model/src/models.py"
    spec_path = tmp_path / "spec.json"
    spec_path.write_text("TEST_FIXTURE")
    monkeypatch.setattr(query_audit.preflight, "_git", lambda root, *args: "member1")
    monkeypatch.setattr(query_audit.preflight, "_read_val_manifest", lambda path: [])
    monkeypatch.setattr(query_audit.preflight, "_reconcile_val_manifests", lambda *args: set())
    monkeypatch.setattr(query_audit.preflight, "_check_predictions", lambda *args: {})
    monkeypatch.setattr(query_audit.preflight, "_check_feature_paths", lambda *args:
                        {"existing_local_paths": 0, "unresolved_paths": ["TEST_FIXTURE"]})
    monkeypatch.setattr(query_audit.preflight, "_environment", lambda: dict.fromkeys(keys, "local"))
    requested = tmp_path / "DO_NOT_OPEN_test_feature.npz"
    real_inside = query_audit.ensure_inside_repo
    def guarded_inside(root, path, **kwargs):
        if path == requested:
            pytest.fail("requested feature resolved before cohort/runtime readiness")
        return real_inside(root, path, **kwargs)
    monkeypatch.setattr(query_audit, "ensure_inside_repo", guarded_inside)
    report = query_audit.pilot_entry(tmp_path, spec_path, tmp_path / "model", requested)
    assert len(report["blockers"]) == 2
    assert report["runtime_differences"]
    assert report["status"] == "BLOCKED"
    assert not report["feature_contents_read"]
