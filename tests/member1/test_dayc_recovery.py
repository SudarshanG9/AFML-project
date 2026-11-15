"""Read-only recovery checks on clearly synthetic TEST_FIXTURE files only."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import zipfile

import pytest
from scripts.dayC_member1 import recovery_inventory as scan


def digest(data):return hashlib.sha256(data).hexdigest()


def basic_spec(root, assets):
    return {'assets':assets,'original_archives':{},'cache_root':str(root/'missing_cache'),
        'search_roots':[str(root)],'max_depth':8,'max_entries':20000,
        'validation_id_set_sha256':'0'*64,'validation_feature_hash_aggregate_sha256':'0'*64,
        'validation_metadata_hash_aggregate_sha256':'0'*64,'expected_count':1200}


def inventory(spec):
    with redirect_stdout(io.StringIO()):return scan.recover_inventory(spec)


def test_exact_copy_is_verified_without_substitution_or_writes(tmp_path):
    original=b'{"TEST_FIXTURE":"not an actual G5 context"}'
    candidate=tmp_path/'backup/execution_context.json';candidate.parent.mkdir();candidate.write_bytes(original)
    spec=basic_spec(tmp_path,{'g5_context':{'names':['execution_context.json'],'sha256':digest(original),
                              'recorded_path':str(tmp_path/'original_missing/execution_context.json')}})
    before={str(p.relative_to(tmp_path)):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    report=inventory(spec)
    assert len(report['exact_original_files'])==1
    assert not report['exact_original_files'][0]['recorded_runtime_location']
    assert not report['asset_writes'] and not report['archives_extracted']
    assert before=={str(p.relative_to(tmp_path)):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert not Path(spec['assets']['g5_context']['recorded_path']).exists()


def test_archive_members_are_hashed_but_not_extracted_or_parsed_as_labels(tmp_path):
    original=b'TEST_FIXTURE: opaque canonical bytes; not JSON'
    archive=tmp_path/'AFML_G5_backup.zip'
    with zipfile.ZipFile(archive,'x') as z:
        z.writestr('AFML/data/manifests/manifest_whole.jsonl',original)
        z.writestr('../execution_context.json',original)
        z.writestr('AFML/test/execution_context.json',original)
    spec=basic_spec(tmp_path,{'canonical_manifest':{'names':['manifest_whole.jsonl'],'sha256':digest(original)},
                            'g5_context':{'names':['execution_context.json'],'sha256':digest(original)}})
    report=inventory(spec)
    assert [x['assets'] for x in report['exact_original_files']]==[['canonical_manifest']]
    assert not (tmp_path/'AFML').exists()
    assert not (tmp_path/'execution_context.json').exists()


def test_unmatched_metadata_does_not_become_original(tmp_path):
    p=tmp_path/'execution_context.json';p.write_bytes(b'TEST_FIXTURE different')
    spec=basic_spec(tmp_path,{'g5_context':{'names':[p.name],'sha256':'0'*64}})
    report=inventory(spec)
    assert not report['exact_original_files'] and report['other_candidates']
    assert 'g5_context' in report['assets_not_found_in_inspected_locations']


def test_test_and_train_directories_are_never_opened(tmp_path,monkeypatch):
    for split in ['test','train']:
        d=tmp_path/split;d.mkdir();(d/'execution_context.json').write_text('TEST_FIXTURE')
    original=scan.file_sha
    def guarded(path):
        if any(part in {'test','train'} for part in Path(path).relative_to(tmp_path).parts):
            pytest.fail('sealed directory opened')
        return original(path)
    monkeypatch.setattr(scan,'file_sha',guarded)
    report=inventory(basic_spec(tmp_path,{'g5_context':{'names':['execution_context.json'],'sha256':'0'*64}}))
    assert not report['exact_original_files']


def test_search_budget_is_reported_as_incomplete(tmp_path):
    (tmp_path/'a').mkdir();(tmp_path/'b').mkdir()
    spec=basic_spec(tmp_path,{});spec['max_entries']=1
    assert inventory(spec)['search_incomplete']


@pytest.mark.parametrize('defect',['short','test','duplicate'])
def test_validation_id_loader_requires_all_1200_val_rows(defect):
    rows=[{'clip_id':f'{i:016x}','split':'val'} for i in range(1200)]
    if defect=='short':rows.pop()
    elif defect=='test':rows[0]['split']='test'
    else:rows[-1]['clip_id']=rows[0]['clip_id']
    with pytest.raises(ValueError):scan.validation_ids('\n'.join(json.dumps(x) for x in rows).encode())


def test_all_1200_cache_files_can_be_verified_without_regenerating_metadata(tmp_path):
    # No ready manifest is available. The independently pinned ID and aggregate
    # hashes can still identify all exact fixture bytes without making a manifest.
    root=tmp_path/'TEST_FIXTURE_cache';val=root/'val';val.mkdir(parents=True)
    features=[];metadata=[];ids=[]
    for i in range(1200):
        clip=f'{i:016x}';ids.append(clip);folder=val/clip;folder.mkdir()
        feature=('TEST_FIXTURE feature '+str(i)).encode();meta=('TEST_FIXTURE meta '+str(i)).encode()
        (folder/'features.npz').write_bytes(feature);(folder/'meta.json').write_bytes(meta)
        features.append((clip,digest(feature)));metadata.append((clip,digest(meta)))
    spec=basic_spec(tmp_path,{})
    spec.update(validation_id_set_sha256=scan.ids_sha(ids),
                validation_feature_hash_aggregate_sha256=scan.aggregate_hash(features),
                validation_metadata_hash_aggregate_sha256=scan.aggregate_hash(metadata))
    before={str(p.relative_to(tmp_path)) for p in tmp_path.rglob('*')}
    report=inventory(spec)
    cache=report['validation_caches'][0]
    assert cache['feature_files_present']==cache['metadata_files_present']==1200
    assert cache['all_feature_bytes_match'] and cache['all_metadata_bytes_match']
    assert cache['status']=='EXACT_1200_CACHE_BYTES'
    assert before=={str(p.relative_to(tmp_path)) for p in tmp_path.rglob('*')}
    assert not list(tmp_path.rglob('*.jsonl'))


def test_partial_cache_does_not_pass_without_original_cohort(tmp_path):
    folder=tmp_path/'cache/val/clip';folder.mkdir(parents=True);(folder/'features.npz').write_bytes(b'TEST_FIXTURE')
    report=inventory(basic_spec(tmp_path,{}))
    assert report['validation_caches'][0]['status']=='UNVERIFIED_NEEDS_ORIGINAL_READY_MANIFEST'



def test_all_1200_archived_cache_bytes_can_be_verified_without_extraction(tmp_path):
    archive=tmp_path/'AFML_cache_backup.zip';features=[];metadata=[];ids=[]
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for i in range(1200):
            clip=f'{i:016x}';ids.append(clip)
            feature=('TEST_FIXTURE feature '+str(i)).encode();meta=('TEST_FIXTURE meta '+str(i)).encode()
            z.writestr('backup/val/'+clip+'/features.npz',feature)
            z.writestr('backup/val/'+clip+'/meta.json',meta)
            features.append((clip,digest(feature)));metadata.append((clip,digest(meta)))
        z.writestr('backup/test/DO_NOT_OPEN/features.npz',b'TEST_FIXTURE sealed marker')
    spec=basic_spec(tmp_path,{})
    spec.update(validation_id_set_sha256=scan.ids_sha(ids),
                validation_feature_hash_aggregate_sha256=scan.aggregate_hash(features),
                validation_metadata_hash_aggregate_sha256=scan.aggregate_hash(metadata))
    report=inventory(spec)
    cache=report['archive_cache_candidates'][0]
    assert cache['feature_member_count']==1200
    assert cache['feature_bytes_verified'] and cache['status']=='EXACT_1200_ARCHIVED_CACHE_BYTES'
    assert not cache['usable_without_user_restoration']
    assert list(tmp_path.iterdir())==[archive]
