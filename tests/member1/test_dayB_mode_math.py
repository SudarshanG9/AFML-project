"""Independent Day B precision checks beyond the existing manual lowp test."""

import pytest
import torch

from avfuse.member1_fusion.attention import fusion_attention, sdpa_attention
from avfuse.member1_fusion.precision import fake_quant_int8, prepare_qkv
from avfuse.shared.contracts import PRECISION_MODES


def _inputs():
    generator = torch.Generator().manual_seed(20261111)
    return tuple(torch.randn(1, 2, 7, 8, generator=generator) for _ in range(3))


def test_restored_value_output_equals_manual_targeted_math():
    q, k, v = _inputs()
    expected = sdpa_attention(fake_quant_int8(q), fake_quant_int8(k), v)
    actual = fusion_attention(q, k, v, mode="restore_value_output", backend="sdpa")
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    qq, kk, vv = prepare_qkv(q, k, v, "restore_value_output")
    torch.testing.assert_close(qq, fake_quant_int8(q), atol=0, rtol=0)
    torch.testing.assert_close(kk, fake_quant_int8(k), atol=0, rtol=0)
    assert vv is v


def test_query_control_remains_independent():
    q, k, v = _inputs()
    expected = fake_quant_int8(sdpa_attention(q, fake_quant_int8(k), fake_quant_int8(v)))
    actual = fusion_attention(q, k, v, mode="restore_query", backend="sdpa")
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_reference_retains_original_operands_and_output():
    q, k, v = _inputs()
    qq, kk, vv = prepare_qkv(q, k, v, "reference")
    assert qq is q and kk is k and vv is v
    actual = fusion_attention(q, k, v, mode="reference", backend="sdpa")
    torch.testing.assert_close(actual, sdpa_attention(q, k, v), atol=0, rtol=0)


@pytest.mark.parametrize("mode", PRECISION_MODES)
def test_modes_preserve_input_tensors_and_float32(mode):
    operands = _inputs()
    snapshots = tuple(tensor.clone() for tensor in operands)
    output = fusion_attention(*operands, mode=mode, backend="sdpa")

    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()
    for tensor, snapshot in zip(operands, snapshots):
        assert tensor.dtype == torch.float32
        assert torch.equal(tensor, snapshot)


def test_quantizer_matches_hand_checked_per_tensor_int8():
    # One tensor-wide scale: max(abs(x)) / 127 = 1. These expected values
    # independently check signed rounding and that the second row does not
    # receive a separate scale. The tensor still executes in floating point.
    x = torch.tensor([
        [-127.0, -0.51, 0.0, 0.49],
        [31.7, 0.51, 63.6, -63.6],
    ], dtype=torch.float32)
    expected = torch.tensor([
        [-127.0, -1.0, 0.0, 0.0],
        [32.0, 1.0, 64.0, -64.0],
    ], dtype=torch.float32)

    actual = fake_quant_int8(x)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.dtype == x.dtype
