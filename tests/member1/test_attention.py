import torch
from avfuse.member1_fusion.attention import dense_attention, sdpa_attention, score_matrix_bytes


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
