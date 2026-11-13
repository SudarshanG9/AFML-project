import torch
from avfuse.member1_fusion.attention import dense_attention, sdpa_attention, score_matrix_bytes
import math
import pytest

def test_dense_matches_sdpa_without_mask():
    torch.manual_seed(1)
    q, k, v = torch.randn(2, 3, 7, 4), torch.randn(2, 3, 11, 4), torch.randn(2, 3, 11, 5)
    torch.testing.assert_close(dense_attention(q,k,v), sdpa_attention(q,k,v), rtol=1e-5, atol=1e-6)
    assert score_matrix_bytes(q,k) == 2*3*7*11*4


def test_dense_matches_sdpa_with_boolean_mask():
    torch.manual_seed(2)
    q, k, v = torch.randn(1,2,5,8), torch.randn(1,2,7,8), torch.randn(1,2,7,8)
    mask = torch.ones(1,1,5,7,dtype=torch.bool)
    mask[...,2:] = False
    torch.testing.assert_close(dense_attention(q,k,v,mask), sdpa_attention(q,k,v,mask), rtol=1e-5, atol=1e-6)



# Day 4.1 — Independently verified dense-attention tests


def test_day4_oracle_uniform_attention():
    """
    If Q is zero, both keys receive equal attention.
    Therefore output equals the average of V.
    """

    q = torch.zeros(1, 1, 1, 2)

    k = torch.tensor([[[[1.0, 0.0],
                        [0.0, 1.0]]]])

    v = torch.tensor([[[[2.0, 4.0],
                        [6.0, 8.0]]]])

    expected = torch.tensor([[[[4.0, 6.0]]]])

    actual = dense_attention(q, k, v)

    torch.testing.assert_close(
        actual,
        expected,
        atol=1e-6,
        rtol=1e-6,
    )


def test_day4_oracle_attention_scaling():
    """
    Independently verify QK^T / sqrt(d)
    and the softmax weighted sum.
    """

    q = torch.tensor([[[[1.0, 0.0]]]])

    k = torch.tensor([[[[1.0, 0.0],
                        [0.0, 1.0]]]])

    v = torch.tensor([[[[2.0, 4.0],
                        [6.0, 8.0]]]])

    # Raw dot products are [1, 0].
    # Head dimension is 2.
    s1 = 1.0 / math.sqrt(2.0)
    s2 = 0.0

    # Independently calculate softmax.
    exp1 = math.exp(s1)
    exp2 = math.exp(s2)

    w1 = exp1 / (exp1 + exp2)
    w2 = exp2 / (exp1 + exp2)

    expected = torch.tensor([[[[
        w1 * 2.0 + w2 * 6.0,
        w1 * 4.0 + w2 * 8.0,
    ]]]])

    actual = dense_attention(q, k, v)

    torch.testing.assert_close(
        actual,
        expected,
        atol=1e-6,
        rtol=1e-6,
    )


def test_day4_oracle_boolean_mask():
    """
    Only key 0 is allowed to contribute.
    Therefore the output must equal V[0].
    """

    q = torch.zeros(1, 1, 1, 2)
    k = torch.zeros(1, 1, 2, 2)

    v = torch.tensor([[[[2.0, 4.0],
                        [6.0, 8.0]]]])

    # Our dense API uses:
    # True = allowed, False = blocked.
    allowed_mask = torch.tensor([
        [True, False]
    ])

    expected = torch.tensor([[[[2.0, 4.0]]]])

    actual = dense_attention(
        q, k, v,
        attn_mask=allowed_mask,
    )

    torch.testing.assert_close(
        actual,
        expected,
        atol=1e-6,
        rtol=1e-6,
    )




@pytest.mark.parametrize(
    "batch,heads,nq,nk,dim,value_dim",
    [
        (1, 1, 1, 1, 2, 2),
        (1, 2, 5, 7, 3, 5),
        (2, 3, 11, 13, 7, 7),
        (1, 4, 31, 37, 16, 16),
        (1, 4, 1202, 1202, 16, 16),
    ],
)
def test_day4_sdpa_shape_parity(
    batch, heads, nq, nk, dim, value_dim
):
    generator = torch.Generator().manual_seed(101)

    q = torch.randn(
        batch, heads, nq, dim, generator=generator
    )
    k = torch.randn(
        batch, heads, nk, dim, generator=generator
    )
    v = torch.randn(
        batch, heads, nk, value_dim, generator=generator
    )

    dense_output = dense_attention(q, k, v)
    sdpa_output = sdpa_attention(q, k, v)

    assert dense_output.shape == (
        batch, heads, nq, value_dim
    )

    assert sdpa_output.shape == dense_output.shape

    torch.testing.assert_close(
        dense_output,
        sdpa_output,
        atol=1e-5,
        rtol=1e-4,
    )



def test_day4_sdpa_boolean_mask_parity():
    generator = torch.Generator().manual_seed(102)

    q = torch.randn(2, 3, 9, 8, generator=generator)
    k = torch.randn(2, 3, 13, 8, generator=generator)
    v = torch.randn(2, 3, 13, 5, generator=generator)

    # Broadcast across batches and heads.
    allowed = torch.ones(
        1, 1, 9, 13, dtype=torch.bool
    )

    # Block the last four keys for every query.
    allowed[..., -4:] = False

    # Block additional pairs for alternating queries.
    allowed[..., ::2, 2:4] = False

    # Every query must retain a permitted key.
    assert allowed.any(dim=-1).all()

    dense_output = dense_attention(
        q, k, v, attn_mask=allowed
    )
    sdpa_output = sdpa_attention(
        q, k, v, attn_mask=allowed
    )

    torch.testing.assert_close(
        dense_output,
        sdpa_output,
        atol=1e-5,
        rtol=1e-4,
    )


def test_day4_sdpa_additive_mask_parity():
    generator = torch.Generator().manual_seed(103)

    q = torch.randn(1, 4, 9, 8, generator=generator)
    k = torch.randn(1, 4, 13, 8, generator=generator)
    v = torch.randn(1, 4, 13, 8, generator=generator)

    mask = torch.zeros(9, 13, dtype=torch.float32)

    # Completely block the last four key positions.
    mask[:, -4:] = float("-inf")

    # Reduce preference for another permitted key.
    mask[:, 3] = -0.7

    dense_output = dense_attention(
        q, k, v, attn_mask=mask
    )
    sdpa_output = sdpa_attention(
        q, k, v, attn_mask=mask
    )

    torch.testing.assert_close(
        dense_output,
        sdpa_output,
        atol=1e-5,
        rtol=1e-4,
    )



def test_day4_padding_mask_excludes_padded_keys():
    """
    Masked padded keys must not affect attention output.
    """

    torch.manual_seed(201)

    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 7, 8)
    v = torch.randn(1, 2, 7, 8)

    valid_length = 5

    # Only the first five key positions are permitted.
    allowed_mask = torch.zeros(
        1, 1, 4, 7,
        dtype=torch.bool
    )
    allowed_mask[..., :valid_length] = True

    # Reference: physically remove the padded keys.
    expected = dense_attention(
        q,
        k[..., :valid_length, :],
        v[..., :valid_length, :],
    )

    # Compare masked attention implementations.
    dense_output = dense_attention(
        q, k, v, allowed_mask
    )
    sdpa_output = sdpa_attention(
        q, k, v, allowed_mask
    )

    torch.testing.assert_close(
        dense_output, expected,
        atol=1e-5, rtol=1e-4
    )

    torch.testing.assert_close(
        sdpa_output, expected,
        atol=1e-5, rtol=1e-4
    )

    # Deliberately corrupt the padded positions.
    modified_k = k.clone()
    modified_v = v.clone()

    modified_k[..., valid_length:, :] = 10000.0
    modified_v[..., valid_length:, :] = -10000.0

    modified_output = sdpa_attention(
        q, modified_k, modified_v, allowed_mask
    )

    # Correct masking makes this modification irrelevant.
    torch.testing.assert_close(
        modified_output, sdpa_output,
        atol=1e-5, rtol=1e-4
    )

def test_day4_fully_masked_query_row():
    """
    A query with no permitted keys must produce
    a finite zero output.
    """

    torch.manual_seed(202)

    q = torch.randn(1, 2, 3, 8)
    k = torch.randn(1, 2, 5, 8)
    v = torch.randn(1, 2, 5, 8)

    allowed_mask = torch.ones(
        3, 5, dtype=torch.bool
    )

    # Completely block query row 1.
    allowed_mask[1, :] = False

    dense_output = dense_attention(
        q, k, v, allowed_mask
    )

    sdpa_output = sdpa_attention(
        q, k, v, allowed_mask
    )

    assert torch.isfinite(dense_output).all()
    assert torch.isfinite(sdpa_output).all()

    # Completely masked row should be zero.
    torch.testing.assert_close(
        dense_output[:, :, 1, :],
        torch.zeros_like(dense_output[:, :, 1, :]),
    )

    torch.testing.assert_close(
        sdpa_output[:, :, 1, :],
        torch.zeros_like(sdpa_output[:, :, 1, :]),
    )

    torch.testing.assert_close(
        dense_output, sdpa_output,
        atol=1e-5, rtol=1e-4
    )



def test_day4_native_mask_conversion():
    """
    Verify MHA blocked-mask to SDPA allowed-mask
    conversion using an independently known answer.
    """

    q = torch.zeros(1, 1, 2, 2)
    k = torch.zeros(1, 1, 2, 2)

    v = torch.tensor([[[[10.0, 20.0],
                        [30.0, 40.0]]]])

    # Native MHA semantics:
    # True = blocked.
    native_blocked_mask = torch.tensor([
        [False, True],
        [True, False],
    ])

    # Convert to standalone SDPA semantics.
    allowed_mask = ~native_blocked_mask

    expected = torch.tensor([[[[10.0, 20.0],
                               [30.0, 40.0]]]])

    dense_output = dense_attention(
        q, k, v, allowed_mask
    )

    sdpa_output = sdpa_attention(
        q, k, v, allowed_mask
    )

    torch.testing.assert_close(
        dense_output, expected
    )

    torch.testing.assert_close(
        sdpa_output, expected
    )



def test_day4_unmasked_padding_can_contribute():
    """
    Show that padding is not automatically ignored
    when no attention mask is supplied.
    """

    q = torch.zeros(1, 1, 1, 2)
    k = torch.zeros(1, 1, 4, 2)

    v = torch.tensor([[[[2.0, 4.0],
                        [6.0, 8.0],
                        [100.0, 100.0],
                        [100.0, 100.0]]]])

    # No mask: all four keys contribute equally.
    unmasked = dense_attention(q, k, v)

    # Explicitly permit only the first two keys.
    allowed_mask = torch.tensor([
        [True, True, False, False]
    ])

    masked = dense_attention(
        q, k, v, allowed_mask
    )

    torch.testing.assert_close(
        masked,
        torch.tensor([[[[4.0, 6.0]]]])
    )

    torch.testing.assert_close(
        unmasked,
        torch.tensor([[[[52.0, 53.0]]]])
    )

    assert not torch.allclose(masked, unmasked)
