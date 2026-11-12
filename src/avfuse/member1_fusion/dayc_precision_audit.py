"""Day C diagnostic checks only; preserve the existing production precision math.

Trace summaries validate structure and preserved boundaries. They cannot prove
quantizer equality without operands; independent operand-oracle tests are a
separate requirement. No research conclusion follows from synthetic checks.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import math
import threading

import torch

from .dayc_guards import require_sha256
from .dimodif_intervention import intervene_dimodif_attention

_BOUNDARIES = ("Q", "K", "V", "attention_output")
_TABLE = {
    "reference": ("FP32", "FP32", "FP32", "FP32"),
    "low_precision": ("fake_INT8", "fake_INT8", "fake_INT8", "fake_INT8"),
    "restore_value_output": ("fake_INT8", "fake_INT8", "FP32", "FP32"),
    "restore_query": ("FP32", "fake_INT8", "fake_INT8", "fake_INT8"),
}
_ACTIVE_MODELS: set[int] = set()
_LOCK = threading.Lock()


def expected_mode_boundaries(mode: str) -> dict[str, str]:
    if mode not in _TABLE:
        raise ValueError(f"Unknown precision mode: {mode}")
    return dict(zip(_BOUNDARIES, _TABLE[mode]))


def validate_precision_trace(trace: list[dict], *, mode: str, n_layers: int = 5) -> dict:
    boundaries = expected_mode_boundaries(mode)
    if type(n_layers) is not int or n_layers < 1 or len(trace) != n_layers * 4:
        raise ValueError("Trace must contain exactly four boundaries per layer")
    seen = set()
    shapes = {}
    for row in trace:
        layer, tensor = row.get("layer"), row.get("tensor")
        key = (layer, tensor)
        if (type(layer) is not int or not 0 <= layer < n_layers or tensor not in boundaries
                or key in seen or row.get("mode") != mode):
            raise ValueError("Duplicate, missing, out-of-range or wrong-mode trace boundary")
        seen.add(key)
        shape = row.get("shape")
        if (not isinstance(shape, list) or len(shape) != 4
                or any(type(dim) is not int or dim < 1 for dim in shape)
                or row.get("dtype") != "torch.float32"):
            raise ValueError("Trace requires positive [batch,heads,tokens,head_dim] FP32 shape")
        shapes.setdefault(layer, shape)
        if shapes[layer] != shape:
            raise ValueError("Self-attention boundary shapes differ within a layer")
        stats = {}
        for name in ("max_abs", "scale", "mean_absolute_error", "max_absolute_error",
                     "relative_l2_error", "fraction_changed"):
            value = row.get(name)
            if (isinstance(value, bool) or not isinstance(value, (float, int))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"Invalid diagnostic statistic: {name}")
            stats[name] = value
        if (stats["fraction_changed"] > 1 or stats["scale"] <= 0
                or stats["mean_absolute_error"] > stats["max_absolute_error"] + 1e-12):
            raise ValueError("Inconsistent diagnostic statistics")
        errors = (stats["mean_absolute_error"], stats["max_absolute_error"],
                  stats["relative_l2_error"], stats["fraction_changed"])
        if boundaries[tensor] == "FP32" and any(value != 0 for value in errors):
            raise ValueError(f"Preserved {tensor} changed at layer {layer}")
        if stats["fraction_changed"] == 0 and any(value != 0 for value in errors):
            raise ValueError("Zero changed fraction contradicts nonzero error")
        if boundaries[tensor] == "fake_INT8":
            tolerance = max(1e-7, stats["max_abs"] * 2e-6)
            if stats["max_absolute_error"] > stats["scale"] / 2 + tolerance:
                raise ValueError("Fake-INT8 error exceeds rounding bound")
    return {"status": "PASS", "mode": mode, "n_layers": n_layers, "rows": len(trace),
            "boundaries": boundaries, "scope": "summary structure/preservation/rounding bounds",
            "operand_oracle_equality": "not inferable from summary; requires independent tests",
            "nonzero_quantization_error_required": False}


def _state_digest(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def check_model_invariants(adapter, *, checkpoint_sha256: str) -> dict:
    model = adapter.model
    if adapter.checkpoint_sha256 != checkpoint_sha256:
        raise ValueError("Adapter checkpoint identity differs")
    require_sha256(adapter.checkpoint_path, checkpoint_sha256)
    if any(module.training for module in model.modules()):
        raise ValueError("All detector modules must be in eval mode")
    if any(p.requires_grad or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("Detector parameters must remain frozen FP32")
    layers = list(model.transformer_encoder.layers)
    if len(layers) != 5:
        raise ValueError("Expected all five joint AV attention layers")
    for layer in layers:
        module = layer.self_attn
        if (not isinstance(module, torch.nn.MultiheadAttention) or not module.batch_first
                or module.num_heads != adapter.config["nhead"]
                or module.embed_dim != adapter.config["d_model"]
                or module.embed_dim % module.num_heads
                or module.in_proj_weight is None or module.bias_k is not None
                or module.bias_v is not None or module.add_zero_attn
                or tuple(module.in_proj_weight.shape) != (3 * module.embed_dim, module.embed_dim)
                or module.forward.__func__ is not torch.nn.MultiheadAttention.forward):
            raise ValueError("Attention architecture differs or a live monkey-patch remains")
    return {"status": "PASS", "checkpoint_sha256": checkpoint_sha256,
            "n_layers": 5, "state_dict_sha256": _state_digest(model),
            "live_attention_patch": False, "state_comparison_scope":
            "snapshot; caller must compare before/after the audited operation"}


@contextmanager
def _exclusive_audit(model):
    key = id(model)
    with _LOCK:
        if key in _ACTIVE_MODELS:
            raise RuntimeError("Concurrent or nested audit on the same detector is forbidden")
        _ACTIVE_MODELS.add(key)
    try:
        yield
    finally:
        with _LOCK:
            _ACTIVE_MODELS.remove(key)


@torch.inference_mode()
def reference_parity_on_validation_feature(adapter, features) -> dict:
    """Caller must first bind features to an approved validation manifest row.

    This function does not establish split membership from tensor contents.
    Never call it on a held-out feature or inside the final score/benchmark run.
    """
    with _exclusive_audit(adapter.model):
        before = check_model_invariants(adapter, checkpoint_sha256=adapter.checkpoint_sha256)
        methods = [layer.self_attn.forward for layer in adapter.model.transformer_encoder.layers]
        snapshots = {key: value.detach().clone() for key, value in features.items()}
        trace = []
        try:
            original = adapter.forward_reference(features)
            instrumented = adapter.forward_with_mode(features, mode="reference", trace=trace)
            for result in (original, instrumented):
                if not torch.isfinite(result["logits"]).all() or not torch.isfinite(result["score_fake"]).all():
                    raise ValueError("Nonfinite parity output")
                if not ((result["score_fake"] >= 0) & (result["score_fake"] <= 1)).all():
                    raise ValueError("Parity fake probability outside [0,1]")
            torch.testing.assert_close(original["logits"], instrumented["logits"], atol=1e-5, rtol=1e-4)
            torch.testing.assert_close(original["score_fake"], instrumented["score_fake"], atol=1e-6, rtol=1e-5)
            trace_audit = validate_precision_trace(trace, mode="reference")
            # Exercise the unchanged context manager's finally path without an
            # additional model call or any intentionally bad feature input.
            class CleanupProbe(Exception):
                pass
            try:
                with intervene_dimodif_attention(adapter.model, mode="restore_query"):
                    raise CleanupProbe()
            except CleanupProbe:
                pass
        finally:
            after = check_model_invariants(adapter, checkpoint_sha256=adapter.checkpoint_sha256)
            if before["state_dict_sha256"] != after["state_dict_sha256"]:
                raise RuntimeError("State dictionary changed during parity audit")
            if methods != [layer.self_attn.forward for layer in adapter.model.transformer_encoder.layers]:
                raise RuntimeError("Original attention methods were not restored")
            if set(features) != set(snapshots) or any(not torch.equal(features[k], v) for k, v in snapshots.items()):
                raise RuntimeError("Input feature tensors changed during parity audit")
        return {"status": "PASS", "logits_atol": 1e-5, "logits_rtol": 1e-4,
                "score_atol": 1e-6, "score_rtol": 1e-5, "trace": trace_audit,
                "state_unchanged": True, "inputs_unchanged": True,
                "method_cleanup_success_and_exception": True,
                "scope": "parity on caller-authorized input; not population validation"}
