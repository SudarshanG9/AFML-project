import torch
from torch import nn

from avfuse.member1_fusion.dimodif_intervention import intervene_dimodif_attention


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128,
            dropout=0.0, batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=3)

    def forward(self, x):
        return self.transformer_encoder(x)


def _capture(model, x, mode):
    seen, hooks = [], []
    try:
        for layer in model.transformer_encoder.layers:
            hooks.append(layer.self_attn.register_forward_pre_hook(
                lambda _module, args: seen.append(args[0].detach().clone())
            ))
        with intervene_dimodif_attention(model, mode=mode):
            with torch.inference_mode():
                model(x)
    finally:
        for h in hooks:
            h.remove()
    return seen


def test_first_layer_has_matched_upstream_even_when_whole_model_modes_differ():
    torch.manual_seed(27)
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        model = ToyModel().eval()
        x = torch.randn(1, 13, 64)
        low = _capture(model, x, 'low_precision')
        restored = _capture(model, x, 'restore_value_output')
        assert len(low) == len(restored) == 3
        assert torch.equal(low[0], restored[0])
        assert all(a.shape == b.shape for a, b in zip(low, restored))
        # Do not demand equality for deeper layers; earlier restoration may alter their inputs.
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)
