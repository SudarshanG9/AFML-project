import json
from pathlib import Path

import numpy as np
import pytest

from scripts.dayB_prepare_pilot_manifest import build_pilot_manifest
from avfuse.member2_data.manifest import read_manifest, write_manifest


def _fixture(tmp_path: Path):
    root = tmp_path / 'pilot'
    source = tmp_path / 'source.jsonl'
    rows = []
    categories = ('real', 'audio_fake', 'video_fake', 'both_fake')
    for i, category in enumerate(categories):
        clip_id = f'{i + 1:016x}'
        rows.append({
            'clip_id': clip_id, 'dataset': 'lavdf',
            'source_id': f'src{i}', 'identity_id': f'src{i}',
            'split': 'val', 'label': 0 if i == 0 else 1,
            'manipulation_type': category,
            'feature_path': f'data/cache/{clip_id}.pt',
        })
        # Leave one pilot clip deliberately absent.
        if i != 2:
            directory = root / 'val' / clip_id
            directory.mkdir(parents=True)
            np.savez(directory / 'features.npz',
                video_features=np.zeros((5, 768), dtype=np.float32),
                audio_features=np.zeros((6, 768), dtype=np.float32))
    write_manifest(rows, source)
    return source, root


def test_pilot_lists_existing_only_without_changing_labels(tmp_path):
    source, root = _fixture(tmp_path)
    output = tmp_path / 'pilot_ready.jsonl'
    original = source.read_bytes()
    report = build_pilot_manifest(source, root, output)
    assert report['available_records'] == 3
    assert report['missing_records'] == 1
    assert report['status'] == 'PILOT_ONLY_NOT_FULL_VALIDATION'
    rows = read_manifest(output)
    assert len(rows) == 3
    assert all(Path(r['feature_path']).is_file() for r in rows)
    assert all(r['split'] == 'val' for r in rows)
    assert source.read_bytes() == original
    source_by_id = {r['clip_id']: r for r in read_manifest(source)}
    for row in rows:
        assert {k: v for k, v in row.items() if k != 'feature_path'} == {
            k: v for k, v in source_by_id[row['clip_id']].items() if k != 'feature_path'
        }
    with pytest.raises(FileExistsError):
        build_pilot_manifest(source, root, output)


def test_pilot_rejects_nonvalidation_manifest(tmp_path):
    source, root = _fixture(tmp_path)
    rows = read_manifest(source)
    rows[0]['split'] = 'test'
    write_manifest(rows, source)
    with pytest.raises(ValueError, match='validation-only'):
        build_pilot_manifest(source, root, tmp_path / 'out.jsonl')
