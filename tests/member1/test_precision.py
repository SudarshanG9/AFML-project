import torch
from avfuse.member1_fusion.precision import fake_quant_int8, prepare_qkv
from avfuse.member1_fusion.attention import fusion_attention
from avfuse.shared.contracts import PRECISION_MODES


def test_quant_preserves_shape_dtype_and_zero():
    x = torch.tensor([-3.12, 0.0, 1.19],dtype=torch.float32)
    y = fake_quant_int8(x)
    assert y.dtype == x.dtype and y.shape == x.shape
    assert torch.equal(fake_quant_int8(torch.zeros(4)), torch.zeros(4))
    assert float(y.abs().max()) <= float(x.abs().max()) + 1e-6


def test_restore_paths_preserve_original_unquantized_operand():
    q,k,v = [torch.tensor([[[[0.13,0.79],[-0.41,1.24]]]]) for _ in range(3)]
    qq,kk,vv = prepare_qkv(q,k,v,'restore_value_output')
    assert torch.equal(vv,v) and not torch.equal(qq,q)
    qq2,kk2,vv2 = prepare_qkv(q,k,v,'restore_query')
    assert torch.equal(qq2,q) and not torch.equal(vv2,v)


def test_four_paths_return_valid_output():
    torch.manual_seed(4)
    q,k,v = torch.randn(1,2,3,8), torch.randn(1,2,4,8), torch.randn(1,2,4,8)
    for mode in PRECISION_MODES:
        output = fusion_attention(q,k,v,mode=mode)
        assert output.shape == (1,2,3,8) and torch.isfinite(output).all()


def test_manual_low_precision_attention():
    import torch

    from avfuse.member1_fusion.precision import (
        fake_quant_int8,
    )
    from avfuse.member1_fusion.attention import (
        fusion_attention,
        sdpa_attention,
    )

    torch.manual_seed(123)

    q = torch.randn(1, 2, 7, 8)
    k = torch.randn(1, 2, 7, 8)
    v = torch.randn(1, 2, 7, 8)

    # Independent construction of the mathematical
    # definition of the low-precision experiment.
    q_hat = fake_quant_int8(q)
    k_hat = fake_quant_int8(k)
    v_hat = fake_quant_int8(v)

    raw = sdpa_attention(q_hat, k_hat, v_hat)
    expected = fake_quant_int8(raw)

    actual = fusion_attention(
        q, k, v,
        mode="low_precision",
        backend="sdpa",
    )

    torch.testing.assert_close(
        actual,
        expected,
        atol=0,
        rtol=0,
    )