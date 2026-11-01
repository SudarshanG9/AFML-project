"""Attention correctness reference and PyTorch SDPA wrapper.

Inputs shaped [batch, heads, query_or_key_tokens, head_dim].
SDPA boolean attention masks: True = permitted attention.
"""
import math
import torch
from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention
from avfuse.shared.contracts import PrecisionMode
from .precision import prepare_qkv, quantize_output


def dense_attention(q: Tensor, k: Tensor, v: Tensor, attn_mask: Tensor | None = None) -> Tensor:
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = (q @ k.transpose(-2, -1)) * scale
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask
    weights = torch.softmax(scores, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)
    return weights @ v


def sdpa_attention(q: Tensor, k: Tensor, v: Tensor, attn_mask: Tensor | None = None) -> Tensor:
    # Passing dropout_p=0.0 matters: SDPA applies dropout in eval mode otherwise.
    return scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)


def fusion_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mode: PrecisionMode = "reference",
    backend: str = "sdpa",
    attn_mask: Tensor | None = None,
) -> Tensor:
    """Experiment only at Q/K/V and fused output; weights/projections unchanged.

    In the targeted policy, unquantized V and output are restored.
    In the control policy, only Q is restored. These costs are not assumed equal.
    """
    qq, kk, vv = prepare_qkv(q, k, v, mode)
    if backend == "sdpa":
        out = sdpa_attention(qq, kk, vv, attn_mask)
    elif backend == "dense":
        out = dense_attention(qq, kk, vv, attn_mask)
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    return quantize_output(out, mode)


def score_matrix_bytes(q: Tensor, k: Tensor) -> int:
    """Naive full score tensor size (not the actual allocation in fused SDPA)."""
    batch, heads, n_q, _ = q.shape
    n_k = k.shape[-2]
    return batch * heads * n_q * n_k * q.element_size()
