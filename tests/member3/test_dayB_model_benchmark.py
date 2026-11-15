import torch
from torch import nn

from avfuse.member3_runtime.artifacts import validate_benchmark_records
from scripts.dayB_model_benchmark import build_model_benchmark_records, EXPECTED_SHA


class FakeFrozenAdapter:
    checkpoint_sha256 = EXPECTED_SHA

    def __init__(self, checkpoint):
        self.checkpoint_path = checkpoint
        self.model = nn.Linear(1, 1, bias=False).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.inference_mode()
    def predict_from_cached_features(self, features, mode):
        assert mode in ('reference', 'low_precision', 'restore_value_output')
        s = features['video_tokens'].mean() + features['audio_tokens'].mean()
        return torch.sigmoid(s.reshape(1))


def test_benchmark_uses_existing_schema_and_explicit_scope(tmp_path):
    import numpy as np
    checkpoint = tmp_path / 'unit-checkpoint.pth'
    checkpoint.write_bytes(b'unit-test-checkpoint-not-real-weights')
    feature_dir = tmp_path / 'val' / '0000000000000001'
    feature_dir.mkdir(parents=True)
    features_file = feature_dir / 'features.npz'
    np.savez(features_file,
        video_features=np.zeros((5, 768), dtype=np.float32),
        audio_features=np.zeros((6, 768), dtype=np.float32))
    features = {
        'video_tokens': torch.zeros(5, 768),
        'audio_tokens': torch.zeros(6, 768),
    }
    adapter = FakeFrozenAdapter(checkpoint)
    records = build_model_benchmark_records(
        adapter, features, features_file, device='cpu',
        modes=('reference', 'low_precision', 'restore_value_output'),
        warmup=0, repeats=3, run_id='unit-benchmark', experiment_id='dayB-unit',
    )
    validate_benchmark_records(records, run_id='unit-benchmark')
    assert len(records) == 3
    assert {r['config'] for r in records} == {
        'reference', 'low_precision', 'restore_value_output'
    }
    assert all(r['scope'] == 'cached_feature_model' for r in records)
    assert all(r['peak_memory_bytes'] is None for r in records)
    assert all(r['precision']['real_int8_kernel'] is False for r in records)
    assert all(r['workload']['input_boundary'] == 'cached_encoder_tokens'
               for r in records)


def test_benchmark_rejects_unapproved_checkpoint_metadata(tmp_path):
    import pytest
    adapter = FakeFrozenAdapter(tmp_path / "checkpoint.pth")
    adapter.checkpoint_sha256 = "b" * 64
    source = tmp_path / "features.npz"
    source.write_bytes(b"not-read-by-this-guard-test")
    with pytest.raises(RuntimeError, match="checkpoint SHA mismatch"):
        build_model_benchmark_records(
            adapter, {}, source, device="cpu", modes=("reference",),
            warmup=0, repeats=1, run_id="unit-guard", experiment_id="unit-guard",
        )


def test_benchmark_rejects_training_model(tmp_path):
    import pytest
    adapter = FakeFrozenAdapter(tmp_path / "checkpoint.pth")
    adapter.model.train()
    source = tmp_path / "features.npz"
    source.write_bytes(b"not-read-by-this-guard-test")
    with pytest.raises(RuntimeError, match="eval and fully frozen"):
        build_model_benchmark_records(
            adapter, {}, source, device="cpu", modes=("reference",),
            warmup=0, repeats=1, run_id="unit-guard", experiment_id="unit-guard",
        )


def test_benchmark_rejects_trainable_parameters(tmp_path):
    import pytest
    adapter = FakeFrozenAdapter(tmp_path / "checkpoint.pth")
    for parameter in adapter.model.parameters():
        parameter.requires_grad_(True)
    source = tmp_path / "features.npz"
    source.write_bytes(b"not-read-by-this-guard-test")
    with pytest.raises(RuntimeError, match="eval and fully frozen"):
        build_model_benchmark_records(
            adapter, {}, source, device="cpu", modes=("reference",),
            warmup=0, repeats=1, run_id="unit-guard", experiment_id="unit-guard",
        )
