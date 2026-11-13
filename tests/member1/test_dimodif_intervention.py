
import pytest
import torch
from torch import nn

from avfuse.member1_fusion.dimodif_intervention import (
    intervene_dimodif_attention,
)


class ToyDetector(nn.Module):
    """Small Transformer with DiMoDif-compatible attention."""

    def __init__(self):
        super().__init__()

        layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.0,
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            layer,
            num_layers=3,
        )

    def forward(self, x, mask=None):
        return self.transformer_encoder(
            x,
            mask=mask,
        )


@pytest.fixture
def setup_model():
    torch.manual_seed(42)

    # Match the reference adapter's non-fastpath setup.
    previous_fastpath = (
        torch.backends.mha.get_fastpath_enabled()
    )

    torch.backends.mha.set_fastpath_enabled(False)

    try:
        model = ToyDetector().eval()

        x = torch.randn(2, 19, 64)

        # MHA boolean True means blocked.
        mask = torch.zeros(
            19, 19, dtype=torch.bool
        )
        mask[:, 14:17] = True

        yield model, x, mask

    finally:
        torch.backends.mha.set_fastpath_enabled(
            previous_fastpath
        )


@torch.inference_mode()
def test_reference_matches_original(setup_model):
    model, x, mask = setup_model

    original = model(x, mask)

    with intervene_dimodif_attention(
        model,
        mode="reference",
    ):
        instrumented = model(x, mask)

    torch.testing.assert_close(
        original,
        instrumented,
        rtol=1e-4,
        atol=1e-5,
    )


@torch.inference_mode()
def test_all_modes_are_finite(setup_model):
    model, x, mask = setup_model

    modes = (
        "reference",
        "low_precision",
        "restore_value_output",
        "restore_query",
    )

    for mode in modes:
        with intervene_dimodif_attention(
            model,
            mode=mode,
        ):
            output = model(x, mask)

        assert output.shape == x.shape
        assert torch.isfinite(output).all()


@torch.inference_mode()
def test_low_precision_changes_output(setup_model):
    model, x, mask = setup_model

    with intervene_dimodif_attention(
        model, "reference"
    ):
        reference = model(x, mask)

    with intervene_dimodif_attention(
        model, "low_precision"
    ):
        low_precision = model(x, mask)

    assert not torch.equal(
        reference,
        low_precision,
    )


@torch.inference_mode()
def test_restoration_boundaries(setup_model):
    model, x, mask = setup_model

    trace = []

    with intervene_dimodif_attention(
        model,
        "restore_value_output",
        trace=trace,
    ):
        model(x, mask)

    # Three Transformer layers, four diagnostics each.
    assert len(trace) == 3 * 4

    by_layer = {}

    for record in trace:
        by_layer.setdefault(
            record["layer"], {}
        )[record["tensor"]] = record

    assert set(by_layer) == {0, 1, 2}

    for layer in by_layer.values():

        # The targeted restoration preserves original V.
        assert layer["V"]["fraction_changed"] == 0

        # It also bypasses output fake quantization.
        assert (
            layer["attention_output"]["fraction_changed"]
            == 0
        )

        # Q/K remain fake-quantized.
        assert layer["Q"]["fraction_changed"] > 0
        assert layer["K"]["fraction_changed"] > 0


@torch.inference_mode()
def test_model_weights_are_unchanged(setup_model):
    model, x, mask = setup_model

    before = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }

    with intervene_dimodif_attention(
        model,
        "low_precision",
    ):
        model(x, mask)

    after = model.state_dict()

    for name in before:
        assert torch.equal(
            before[name],
            after[name],
        )


@torch.inference_mode()
def test_original_forward_is_restored(setup_model):
    model, x, mask = setup_model

    original = model(x, mask)

    with intervene_dimodif_attention(
        model,
        "low_precision",
    ):
        model(x, mask)

    restored = model(x, mask)

    torch.testing.assert_close(
        original,
        restored,
        rtol=0,
        atol=0,
    )


def test_invalid_mode_is_rejected(setup_model):
    model, _, _ = setup_model

    with pytest.raises(ValueError):
        with intervene_dimodif_attention(
            model,
            "unsupported_mode",
        ):
            pass


def test_cleanup_after_exception(setup_model):
    model, x, mask = setup_model

    with torch.inference_mode():
        original = model(x, mask)

    with pytest.raises(RuntimeError):
        with intervene_dimodif_attention(
            model,
            "low_precision",
        ):
            raise RuntimeError(
                "Simulated inference failure"
            )

    with torch.inference_mode():
        after = model(x, mask)

    torch.testing.assert_close(
        original,
        after,
        rtol=0,
        atol=0,
    )
