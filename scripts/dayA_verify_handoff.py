
"""Day A: verify the real DiMoDif attention benchmark handoff."""

import json
from pathlib import Path

import torch

from avfuse.member1_fusion.attention import (
    dense_attention,
    sdpa_attention,
    fusion_attention,
    score_matrix_bytes,
)


ARTIFACT_DIR = Path("artifacts/dayA")


def main():
    parity_path = ARTIFACT_DIR / "model_parity.json"
    qkv_path = ARTIFACT_DIR / "layer0_qkv.pt"
    day2_path = Path("artifacts/day2/attention_audit.json")

    parity = json.loads(parity_path.read_text())
    day2 = json.loads(day2_path.read_text())

    # Confirm the same original pretrained checkpoint.
    assert parity["checkpoint_sha256"] == (
        day2["checkpoint_sha256"]
    )

    assert parity["passed"] is True
    assert parity["layers_checked"] == 5

    # Load captured real projected attention tensors.
    saved = torch.load(
        qkv_path,
        map_location="cpu",
        weights_only=True,
    )

    q = saved["q"]
    k = saved["k"]
    v = saved["v"]
    mask = saved["attn_mask"]

    expected_shape = (1, 4, 1202, 16)

    assert tuple(q.shape) == expected_shape
    assert tuple(k.shape) == expected_shape
    assert tuple(v.shape) == expected_shape

    for tensor in (q, k, v):
        assert tensor.dtype == torch.float32
        assert torch.isfinite(tensor).all()

    if mask is not None:
        assert mask.shape[-2:] == (1202, 1202)

        assert mask.dtype == torch.bool or (
            torch.is_floating_point(mask)
        )

    # Verify analytical memory is unchanged.
    score_bytes = score_matrix_bytes(q, k)
    assert score_bytes == 23_116_864

    # Compare direct operators with shared interface.
    with torch.inference_mode():
        direct_dense = dense_attention(q, k, v, mask)

        direct_sdpa = sdpa_attention(q, k, v, mask)

        shared_dense = fusion_attention(
            q, k, v,
            mode="reference",
            backend="dense",
            attn_mask=mask,
        )

        shared_sdpa = fusion_attention(
            q, k, v,
            mode="reference",
            backend="sdpa",
            attn_mask=mask,
        )

    # Wrapper should preserve the direct implementation.
    torch.testing.assert_close(
        shared_dense, direct_dense,
        atol=1e-6, rtol=1e-6,
    )

    torch.testing.assert_close(
        shared_sdpa, direct_sdpa,
        atol=1e-6, rtol=1e-6,
    )

    # Both reference backends must agree.
    torch.testing.assert_close(
        shared_dense, shared_sdpa,
        atol=1e-4, rtol=1e-4,
    )

    max_error = float(
        (shared_dense - shared_sdpa).abs().max()
    )

    report = {
        "checkpoint_sha256": parity["checkpoint_sha256"],
        "source_parity_report": str(parity_path),
        "source_qkv": str(qkv_path),
        "input_kind": "real",
        "q_shape": list(q.shape),
        "k_shape": list(k.shape),
        "v_shape": list(v.shape),
        "dtype": str(q.dtype),
        "mask_shape": (
            list(mask.shape)
            if mask is not None else None
        ),
        "score_matrix_bytes": score_bytes,
        "reference_interface_max_abs_error": max_error,
        "passed": True,
    }

    destination = (
        ARTIFACT_DIR / "operator_handoff.json"
    )

    destination.write_text(
        json.dumps(report, indent=2)
    )

    print(json.dumps(report, indent=2))
    print("\nREAL QKV HANDOFF VERIFIED")


if __name__ == "__main__":
    main()
