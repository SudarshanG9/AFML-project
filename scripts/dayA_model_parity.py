
"""Day A / Day 4: real pretrained DiMoDif attention parity."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from avfuse.member1_fusion.attention import (
    dense_attention,
    sdpa_attention,
)
from avfuse.member1_fusion.day2_probe import load_features
from avfuse.member1_fusion.dimodif_adapter import DiMoDifAdapter


ATOL = 1e-4
RTOL = 1e-4


def convert_mask(mask, batch, heads, nq, nk):
    """Convert native MHA mask to our SDPA mask convention."""

    if mask is None:
        return None

    if mask.ndim == 2:
        assert tuple(mask.shape) == (nq, nk)

    elif mask.ndim == 3:
        assert tuple(mask.shape) == (
            batch * heads, nq, nk
        )
        mask = mask.reshape(batch, heads, nq, nk)

    else:
        raise ValueError(
            f"Unexpected native attention mask: {mask.shape}"
        )

    if mask.dtype == torch.bool:
        # Native MHA: True means blocked.
        # SDPA: True means allowed.
        return ~mask

    # Native additive masks already use 0 / -inf.
    assert torch.is_floating_point(mask)
    return mask


def make_hook(layer_index, records, output_dir):
    """Reconstruct and verify pretrained attention."""

    def hook(module, args, kwargs, original_output):

        assert not module.training
        assert module.batch_first
        assert module.in_proj_weight is not None
        assert not module.add_zero_attn
        assert module.bias_k is None
        assert module.bias_v is None

        if kwargs.get("key_padding_mask") is not None:
            raise RuntimeError(
                "Unexpected key-padding mask in DiMoDif"
            )

        if kwargs.get("is_causal", False):
            raise RuntimeError(
                "Unexpected causal attention"
            )

        xq, xk, xv = args[:3]

        batch = xq.shape[0]
        heads = module.num_heads
        head_dim = module.head_dim
        embed_dim = module.embed_dim

        assert embed_dim == heads * head_dim

        # The native module stores Q/K/V projection
        # matrices in a single packed weight tensor.
        weights = module.in_proj_weight.chunk(3, dim=0)

        if module.in_proj_bias is None:
            biases = (None, None, None)
        else:
            biases = module.in_proj_bias.chunk(3, dim=0)

        projected = []

        # Reuse exactly the pretrained projection weights.
        for x, weight, bias in zip(
            (xq, xk, xv), weights, biases
        ):
            y = F.linear(x, weight, bias)

            # [B, T, E] -> [B, H, T, D]
            y = y.reshape(
                batch, x.shape[1], heads, head_dim
            ).transpose(1, 2)

            projected.append(y)

        q, k, v = projected

        mask = convert_mask(
            kwargs.get("attn_mask"),
            batch,
            heads,
            q.shape[-2],
            k.shape[-2],
        )

        # Our two standalone FP32 implementations.
        dense_result = dense_attention(q, k, v, mask)
        sdpa_result = sdpa_attention(q, k, v, mask)

        dense_sdpa_error = float(
            (dense_result - sdpa_result)
            .abs().max().item()
        )

        torch.testing.assert_close(
            dense_result,
            sdpa_result,
            atol=ATOL,
            rtol=RTOL,
        )

        # Merge attention heads:
        # [B, H, T, D] -> [B, T, E]
        merged = (
            sdpa_result.transpose(1, 2)
            .contiguous()
            .reshape(batch, q.shape[-2], embed_dim)
        )

        # The original pretrained output projection.
        reconstructed = module.out_proj(merged)

        if not isinstance(original_output, tuple):
            raise RuntimeError(
                "Unexpected native MHA return type"
            )

        native = original_output[0]

        native_error = float(
            (reconstructed - native)
            .abs().max().item()
        )

        print(
            f"Layer {layer_index}: "
            f"dense/SDPA error={dense_sdpa_error:.3e}, "
            f"SDPA/native error={native_error:.3e}"
        )

        torch.testing.assert_close(
            reconstructed,
            native,
            atol=ATOL,
            rtol=RTOL,
        )

        records.append({
            "layer": layer_index,
            "q_shape": list(q.shape),
            "k_shape": list(k.shape),
            "v_shape": list(v.shape),
            "dense_sdpa_max_abs_error": dense_sdpa_error,
            "sdpa_native_max_abs_error": native_error,
        })

        # Save only actual layer-0 attention inputs,
        # for Member 3's operator benchmark.
        if layer_index == 0:
            torch.save({
                "q": q.detach().cpu(),
                "k": k.detach().cpu(),
                "v": v.detach().cpu(),
                "attn_mask": (
                    mask.detach().cpu()
                    if mask is not None else None
                ),
            }, output_dir / "layer0_qkv.pt")

        # Temporarily substitute mathematically
        # equivalent FP32 attention output.
        # Hooks will be removed after inference.
        return (reconstructed, *original_output[1:])

    return hook


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--features",
        type=Path,
        default=Path(
            "data/pilot/day2_features/features1.npz"
        ),
    )

    parser.add_argument(
        "--upstream",
        type=Path,
        default=Path("third_party/dimodif"),
    )

    parser.add_argument(
        "--device",
        choices=("cpu", "mps", "cuda"),
        default="cpu",
    )

    args = parser.parse_args()

    output_dir = Path("artifacts/dayA")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the original frozen checkpoint.
    adapter = DiMoDifAdapter(
        upstream_root=args.upstream,
        checkpoint_name="lavdf",
        device=args.device,
    )

    features = load_features(args.features)

    # Run 1: Completely original reference inference.
    with torch.inference_mode():
        reference = adapter.forward_reference(features)

    print(
        "Original fake score:",
        reference["score_fake"].detach().cpu().tolist()
    )

    records = []
    handles = []

    layers = adapter.model.transformer_encoder.layers

    # Attach temporary hooks to all five attention layers.
    for index, layer in enumerate(layers):
        handle = layer.self_attn.register_forward_hook(
            make_hook(index, records, output_dir),
            with_kwargs=True,
        )
        handles.append(handle)

    try:
        # Run 2: Same model, reconstructed FP32 SDPA.
        with torch.inference_mode():
            substituted = adapter.forward_reference(features)

    finally:
        # Important: remove hooks even if a test fails.
        for handle in handles:
            handle.remove()

    if len(records) != len(layers):
        raise RuntimeError(
            f"Expected {len(layers)} records, "
            f"got {len(records)}"
        )

    # Detector-level parity.
    logits_error = float(
        (
            reference["logits"] -
            substituted["logits"]
        ).abs().max().item()
    )

    score_error = float(
        (
            reference["score_fake"] -
            substituted["score_fake"]
        ).abs().max().item()
    )

    torch.testing.assert_close(
        reference["logits"],
        substituted["logits"],
        atol=ATOL,
        rtol=RTOL,
    )

    torch.testing.assert_close(
        reference["score_fake"],
        substituted["score_fake"],
        atol=ATOL,
        rtol=RTOL,
    )

    report = {
        "model": "DiMoDif",
        "checkpoint_sha256": adapter.checkpoint_sha256,
        "feature_file": str(args.features),
        "device": str(adapter.device),
        "torch_version": torch.__version__,
        "layers_checked": len(records),
        "layers": records,
        "logit_max_abs_error": logits_error,
        "fake_score_max_abs_error": score_error,
        "passed": True,
    }

    destination = output_dir / "model_parity.json"

    destination.write_text(
        json.dumps(report, indent=2)
    )

    print("\nDetector logit error:", logits_error)
    print("Detector fake-score error:", score_error)
    print("All five layers passed:", len(records) == 5)
    print("Saved:", destination)


if __name__ == "__main__":
    main()
