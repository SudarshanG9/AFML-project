import pytest
import torch
from torch import nn

from scripts.dayB_matched_upstream_diagnostic import audit_matched, capture_run


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        layer = nn.TransformerEncoderLayer(32, 4, 64, dropout=0, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=3)

    def forward(self, x):
        return self.transformer_encoder(x)


@pytest.fixture
def toy():
    torch.manual_seed(27)
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        yield ToyModel().eval(), torch.randn(1, 13, 32)
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)


def test_each_target_has_identical_lowp_upstream_and_only_target_restored(toy):
    model, x = toy
    result = audit_matched(model, lambda: model(x), lambda t: float(torch.sigmoid(t[0, 0, 0])))
    assert result['weights_unchanged'] and result['original_forwards_restored'] and result['hooks_restored']
    assert len(result['layers']) == 3
    for target, row in enumerate(result['layers']):
        assert row['prefix_and_target_upstream_bit_exact']
        assert row['projected_Q_bit_exact'] and row['projected_K_bit_exact']
        assert row['quantized_Q_bit_exact'] and row['quantized_K_bit_exact']
        assert row['target_V_unquantized']
        assert row['modes'].count('restore_value_output') == 1
        for record in row['trace']:
            assert record['mode'] == ('restore_value_output' if record['layer'] == target else 'low_precision')
            if record['layer'] == target:
                assert record['fraction_changed'] == 0 if record['tensor'] in ('V', 'attention_output') else record['fraction_changed'] > 0


def test_exception_restores_methods_and_hooks(toy):
    model, _ = toy
    layers = model.transformer_encoder.layers
    original = [layer.self_attn.forward for layer in layers]
    def fail():
        raise RuntimeError('injected diagnostic failure')
    with pytest.raises(RuntimeError, match='injected diagnostic failure'):
        capture_run(model, fail, target=1)
    assert [layer.self_attn.forward for layer in layers] == original
    assert all(not layer.self_attn._forward_pre_hooks and not layer.self_attn._forward_hooks for layer in layers)


def test_invalid_target_refuses_and_cleans_hooks(toy):
    model, x = toy
    with pytest.raises(ValueError, match='Target layer'):
        capture_run(model, lambda: model(x), target=9)
    assert all(not layer.self_attn._forward_pre_hooks and not layer.self_attn._forward_hooks for layer in model.transformer_encoder.layers)
