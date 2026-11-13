"""Simulated numerical precision only; NOT integer storage or an INT8 kernel."""
import torch
from torch import Tensor
from avfuse.shared.contracts import PRECISION_MODES, PrecisionMode


def fake_quant_int8(x: Tensor) -> Tensor:
    """Symmetric per-tensor quant-dequant; x stays floating point.

    Uses x's dynamic max-abs per call. Zero tensors remain zero. This is NOT
    representative of a statically calibrated integer runtime.
    """
    if not x.is_floating_point():
        raise TypeError("fake quant expects floating point input")
    max_abs = x.detach().abs().amax()
    eps = torch.finfo(x.dtype).tiny
    scale = max_abs.clamp_min(eps) / 127.0
    return (x / scale).round().clamp(-127.0, 127.0) * scale


def prepare_qkv(q: Tensor, k: Tensor, v: Tensor, mode: PrecisionMode):
    if mode not in PRECISION_MODES:
        raise ValueError(f"Unknown precision mode: {mode}")
    if mode == "reference":
        return q, k, v
    if mode == "low_precision":
        return fake_quant_int8(q), fake_quant_int8(k), fake_quant_int8(v)
    if mode == "restore_value_output":
        return fake_quant_int8(q), fake_quant_int8(k), v
    # restore_query
    return q, fake_quant_int8(k), fake_quant_int8(v)


def quantize_output(out: Tensor, mode: PrecisionMode) -> Tensor:
    if mode in ("low_precision", "restore_query"):
        return fake_quant_int8(out)
    return out




def quantization_diagnostics(
    original: Tensor,
    modified: Tensor,
) -> dict:
    """Measure the numerical effect of a precision intervention.

    Diagnostic only. Do not run this during latency benchmarks
    because reading scalar values can synchronize the GPU.
    """
    if original.shape != modified.shape:
        raise ValueError("Diagnostic tensor shapes do not match")

    if original.dtype != modified.dtype:
        raise ValueError("Diagnostic tensor dtypes do not match")

    if not torch.isfinite(original).all():
        raise ValueError("Original tensor contains NaN/Inf")

    if not torch.isfinite(modified).all():
        raise ValueError("Modified tensor contains NaN/Inf")

    x = original.detach().float()
    y = modified.detach().float()

    error = (x - y).abs()
    max_abs = x.abs().amax()

    scale = max_abs.clamp_min(
        torch.finfo(x.dtype).tiny
    ) / 127.0

    original_norm = torch.linalg.vector_norm(x)
    error_norm = torch.linalg.vector_norm(x - y)

    relative_l2 = error_norm / original_norm.clamp_min(
        torch.finfo(x.dtype).eps
    )

    return {
        "shape": list(x.shape),
        "dtype": str(original.dtype),
        "max_abs": float(max_abs.item()),
        "scale": float(scale.item()),
        "mean_absolute_error": float(error.mean().item()),
        "max_absolute_error": float(error.max().item()),
        "relative_l2_error": float(relative_l2.item()),
        "fraction_changed": float(
            (x != y).float().mean().item()
        ),
    }
