
"""Precision intervention in pretrained DiMoDif attention.

This module does NOT replace pretrained weights.
It temporarily intercepts MultiheadAttention.forward.

Designed for the verified DiMoDif architecture:
- batch_first=True
- joint audio-video self-attention
- frozen inference
- standard packed QKV projections
- no causal attention
"""

from contextlib import contextmanager
from types import MethodType

import torch
import torch.nn.functional as F

from avfuse.shared.contracts import (
    PRECISION_MODES,
    PrecisionMode,
)

from .attention import fusion_attention
from .precision import quantization_diagnostics


def _additive_mask(mask, dtype, device):
    """Convert MHA masks to additive SDPA masks.

    MHA: boolean True means BLOCKED.
    Additive mask: -inf means BLOCKED.
    """
    if mask.dtype == torch.bool:
        return torch.zeros(
            mask.shape,
            dtype=dtype,
            device=device,
        ).masked_fill_(
            mask.to(device),
            float("-inf"),
        )

    if mask.is_floating_point():
        return mask.to(
            device=device,
            dtype=dtype,
        )

    raise TypeError("Unsupported attention mask dtype")


def _prepare_mask(
    attn_mask,
    key_padding_mask,
    batch,
    heads,
    query_length,
    key_length,
    query,
):
    """Prepare masks without changing their semantics."""

    bias = None

    if attn_mask is not None:

        if attn_mask.ndim == 2:
            expected = (query_length, key_length)

            if tuple(attn_mask.shape) != expected:
                raise ValueError(
                    "Incorrect 2D attention mask shape"
                )

        elif attn_mask.ndim == 3:
            expected = (
                batch * heads,
                query_length,
                key_length,
            )

            if tuple(attn_mask.shape) != expected:
                raise ValueError(
                    "Incorrect 3D attention mask shape"
                )

            attn_mask = attn_mask.reshape(
                batch,
                heads,
                query_length,
                key_length,
            )

        else:
            raise ValueError(
                "Expected a 2D or 3D MHA attention mask"
            )

        bias = _additive_mask(
            attn_mask,
            query.dtype,
            query.device,
        )

    if key_padding_mask is not None:

        if tuple(key_padding_mask.shape) != (
            batch,
            key_length,
        ):
            raise ValueError(
                "Incorrect key padding mask shape"
            )

        padding_bias = _additive_mask(
            key_padding_mask[:, None, None, :],
            query.dtype,
            query.device,
        )

        bias = (
            padding_bias
            if bias is None
            else bias + padding_bias
        )

    return bias


def _make_forward(layer_index, mode, trace):
    """Construct a patched MHA forward for one layer."""

    def patched_forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        average_attn_weights=True,
        is_causal=False,
    ):
        if self.training:
            raise RuntimeError(
                "Precision experiment requires eval mode"
            )

        if not self.batch_first:
            raise RuntimeError(
                "Expected batch_first=True"
            )

        if need_weights:
            raise RuntimeError(
                "This detector uses need_weights=False"
            )

        if is_causal:
            raise RuntimeError(
                "Unexpected causal attention"
            )

        if query is not key or query is not value:
            raise RuntimeError(
                "Expected DiMoDif joint self-attention"
            )

        if (
            self.in_proj_weight is None
            or self.bias_k is not None
            or self.bias_v is not None
            or self.add_zero_attn
            or self.kdim != self.embed_dim
            or self.vdim != self.embed_dim
        ):
            raise RuntimeError(
                "Unsupported MHA configuration"
            )

        if query.ndim != 3:
            raise ValueError(
                "Expected input [B, T, D]"
            )

        batch, tokens, embed_dim = query.shape

        if embed_dim != self.embed_dim:
            raise ValueError(
                "Input embedding dimension mismatch"
            )

        heads = self.num_heads
        head_dim = self.head_dim

        # Single packed projection, matching the original
        # MultiheadAttention self-attention computation.
        projected = F.linear(
            query,
            self.in_proj_weight,
            self.in_proj_bias,
        )

        # [B, T, 3D] -> [3, B, H, T, d_head]
        projected = projected.reshape(
            batch,
            tokens,
            3,
            heads,
            head_dim,
        ).permute(2, 0, 3, 1, 4)

        q, k, v = projected.unbind(0)

        mask = _prepare_mask(
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            batch=batch,
            heads=heads,
            query_length=tokens,
            key_length=tokens,
            query=query,
        )

        diagnostic_hook = None

        if trace is not None:

            def record(name, before, after):
                statistics = quantization_diagnostics(
                    before,
                    after,
                )

                trace.append({
                    "layer": layer_index,
                    "tensor": name,
                    "mode": mode,
                    **statistics,
                })

            diagnostic_hook = record

        # Our previously tested attention implementation.
        attended = fusion_attention(
            q=q,
            k=k,
            v=v,
            mode=mode,
            backend="sdpa",
            attn_mask=mask,
            diagnostic_hook=diagnostic_hook,
        )

        # [B, H, T, d_head] -> [B, T, D]
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .reshape(batch, tokens, embed_dim)
        )

        # Use the ORIGINAL pretrained output projection.
        output = self.out_proj(attended)

        # TransformerEncoderLayer expects (output, weights).
        return output, None

    return patched_forward


@contextmanager
def intervene_dimodif_attention(
    model,
    mode: PrecisionMode,
    trace: list[dict] | None = None,
):
    """Temporarily patch every DiMoDif attention layer.

    The original forward methods are always restored,
    including when model inference raises an exception.

    Do not run different modes concurrently on the same
    model instance.
    """

    if mode not in PRECISION_MODES:
        raise ValueError(
            f"Unknown precision mode: {mode}"
        )

    layers = list(
        model.transformer_encoder.layers
    )

    if not layers:
        raise RuntimeError(
            "No Transformer layers found"
        )

    originals = []

    try:
        for index, layer in enumerate(layers):
            module = layer.self_attn

            originals.append(
                (module, module.forward)
            )

            replacement = _make_forward(
                layer_index=index,
                mode=mode,
                trace=trace,
            )

            module.forward = MethodType(
                replacement,
                module,
            )

        yield

    finally:
        # Restore original forward methods even on failure.
        for module, original_forward in originals:
            module.forward = original_forward
