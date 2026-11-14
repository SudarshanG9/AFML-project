"""Synthetic full-cohort identity checks, without model or real data access."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pytest
from avfuse.member1_fusion.dayc_cache_mapping import (
    ValidationCacheResolver, verify_mapped_validation_cache, validate_mapping_release)


@pytest.fixture
def cohort(tmp_path):
    mapped = tmp_path/'new'
    original = tmp_path/'obsolete'
    ready, inventory = [], []
    for n in range(1200):
        clip = f'{n:016x}'
        folder = mapped/'val'/clip
        folder.mkdir(parents=True)
        feature = ('feature'+clip).encode()
        meta = json.dumps({'clip_id':clip,'tokens':192}).encode()
        (folder/'features.npz').write_bytes(feature)
        (folder/'meta.json').write_bytes(meta)
        row = {'clip_id':clip, 'split':'val','source_id':clip,'manipulation_type':'real',
               'feature_path':str(original/'val'/clip/'features.npz')}
        ready.append(row)
        inventory.append({**row,'feature_sha256':hashlib.sha256(feature).hexdigest(),
                          'metadata_sha256':hashlib.sha256(meta).hexdigest()})
    return ready, inventory, original, mapped


def test_all_1200_hashes_without_original_or_test_access(cohort, monkeypatch):
    ready, inventory, original, mapped = cohort
    before = deepcopy(ready)
    # Any filesystem operation directed into sealed test or obsolete paths fails.
    originals = {name:getattr(Path,name) for name in ('open','stat','lstat','iterdir')}
    for name, method in originals.items():
        def guarded(self, *args, _method=method, **kwargs):
            assert not self.is_relative_to(original)
            assert 'test' not in self.parts
            return _method(self,*args,**kwargs)
        monkeypatch.setattr(Path,name,guarded)
    proof = verify_mapped_validation_cache(ready,inventory,original,mapped)
    assert proof['verified_count'] == len(proof['records']) == 1200
    assert ready == before and not proof['test_access'] and not proof['manifest_modified']


@pytest.mark.parametrize('defect', ['duplicate','cohort_id','count','split','feature_hash','meta_hash',
                                  'missing_meta','missing_feature','missing_hash','path','source',
                                  'feature_symlink','folder_symlink','val_symlink','traversal'])
def test_identity_defects_fail_closed(cohort, defect):
    ready, inventory, original, mapped = cohort
    target = mapped/'val'/ready[0]['clip_id']
    if defect == 'duplicate': ready[1] = dict(ready[0])
    elif defect == 'cohort_id': inventory[0]['clip_id'] = 'unexpected'
    elif defect == 'count': ready.pop()
    elif defect == 'split': ready[0]['split'] = 'test'
    elif defect == 'feature_hash': (target/'features.npz').write_bytes(b'drift')
    elif defect == 'meta_hash': (target/'meta.json').write_bytes(b'drift')
    elif defect == 'missing_meta': (target/'meta.json').unlink()
    elif defect == 'missing_feature': (target/'features.npz').unlink()
    elif defect == 'missing_hash': inventory[0]['feature_sha256'] = None
    elif defect == 'path': ready[0]['feature_path'] = str(mapped/'test'/'features.npz')
    elif defect == 'source': ready[0]['source_id'] = 'different'
    elif defect == 'traversal': ready[0]['clip_id'] = inventory[0]['clip_id'] = '../test'
    elif defect == 'feature_symlink':
        (target/'features.npz').unlink()
        (target/'features.npz').symlink_to(target/'meta.json')
    elif defect == 'folder_symlink':
        moved = target.with_name('moved')
        target.rename(moved)
        target.symlink_to(moved, target_is_directory=True)
    elif defect == 'val_symlink':
        (mapped/'val').rename(mapped/'actual')
        (mapped/'val').symlink_to(mapped/'actual', target_is_directory=True)
    with pytest.raises(ValueError):
        verify_mapped_validation_cache(ready,inventory,original,mapped)


def test_mapping_release_cannot_change_protocol():
    root = Path(__file__).resolve().parents[2]
    base = root/'configs/dayC/member1'
    parent = json.loads((base/'colab_pilot_release_drift_v2.json').read_text())
    release = json.loads((base/'colab_pilot_release_mapped_v3.json').read_text())
    validate_mapping_release(release,parent)
    release['environment']['versions']['torch'] = 'changed'
    with pytest.raises(ValueError,match='protocol'):
        validate_mapping_release(release,parent)
