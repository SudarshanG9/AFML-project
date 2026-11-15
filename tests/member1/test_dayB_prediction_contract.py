from types import SimpleNamespace

import numpy as np
import pytest
import torch

from avfuse.member2_data.manifest import write_manifest
from avfuse.member2_data.predictions import validate_predictions
from avfuse.shared.jsonl import read_jsonl
from scripts.dayB_inference import run_validation


class StubAdapter:
    checkpoint_sha256 = 'a' * 64

    def predict_from_cached_features(self, features, mode):
        assert features['video_tokens'].shape == (5, 768)
        assert features['audio_tokens'].shape == (6, 768)
        assert mode == 'reference'
        return torch.tensor([0.625], dtype=torch.float32)


def test_manifest_inference_writes_exact_schema_and_one_row(tmp_path):
    path = tmp_path / 'features.npz'
    np.savez(path,
        video_features=np.zeros((5, 768), dtype=np.float32),
        audio_features=np.zeros((6, 768), dtype=np.float32))
    manifest = tmp_path / 'manifest.jsonl'
    write_manifest([{
        'clip_id': '0000000000000001', 'dataset': 'lavdf',
        'source_id': 'source1', 'identity_id': 'source1',
        'split': 'val', 'label': 1, 'manipulation_type': 'audio_fake',
        'feature_path': str(path),
    }], manifest)
    output = tmp_path / 'predictions.jsonl'
    args = SimpleNamespace(manifest=str(manifest), mode='reference',
                           repo_root=str(tmp_path), run_id='unit-test',
                           out=str(output))
    run_validation(StubAdapter(), args)
    rows = read_jsonl(output)
    validate_predictions(rows)
    assert len(rows) == 1
    assert rows[0]['config'] == 'reference'
    assert rows[0]['clip_id'] == '0000000000000001'
    assert rows[0]['score_fake'] == pytest.approx(0.625)
    assert 'label' not in rows[0]
    assert rows[0]['checkpoint_sha256'] == 'a' * 64


def test_missing_manifest_is_a_real_error(tmp_path):
    args = SimpleNamespace(manifest=str(tmp_path / 'dayB_val_ready.jsonl'),
                           mode='reference', repo_root=str(tmp_path),
                           run_id='missing', out=str(tmp_path / 'out.jsonl'))
    with pytest.raises(FileNotFoundError):
        run_validation(StubAdapter(), args)
