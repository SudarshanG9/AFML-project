"""Tiny validation-only copy fixtures; never touch actual Drive assets."""
import hashlib,json,zipfile
from pathlib import Path
import pytest
from scripts.dayC_member1.complete_member3_handoff import assemble


def fixture(tmp_path,defect=None):
    cache=tmp_path/'TEST_FIXTURE';rows=[];records=[]
    for i in range(2):
        clip='TEST_FIXTURE_'+str(i);folder=cache/'val'/clip;folder.mkdir(parents=True)
        record={'clip_id':clip}
        for name,key,data in [('features.npz','feature_sha256',b'TEST_FIXTURE_ORIGINAL_TENSOR_BYTES'),('meta.json','metadata_sha256',b'{"status":"ok"}')]:
            (folder/name).write_bytes(data);record[key]=hashlib.sha256(data).hexdigest()
        records.append(record);rows.append({'clip_id':clip,'split':'val'})
    if defect=='test':rows[0]['split']='test'
    if defect=='duplicate':rows[1]['clip_id']=rows[0]['clip_id']
    if defect=='hash':records[0]['feature_sha256']='a'*64
    if defect=='missing':(cache/'val/TEST_FIXTURE_0/meta.json').unlink()
    if defect=='symlink':
        p=cache/'val/TEST_FIXTURE_0/features.npz';p.unlink();p.symlink_to(cache/'val/TEST_FIXTURE_1/features.npz')
    payload={'data/manifests/manifest_whole_val.jsonl':''.join(json.dumps(r)+'\n' for r in rows).encode(),
        'validation/accepted_B2_cache_verification.json':json.dumps({'status':'PASS','test_access':False,'verified_count':2,'records':records}).encode(),
        'original_evidence.zip':b'TEST_FIXTURE_OPAQUE_ZIP_BYTES'}
    hashes={n:hashlib.sha256(b).hexdigest() for n,b in payload.items()};base=tmp_path/'base.zip'
    with zipfile.ZipFile(base,'x') as z:
        for n,b in payload.items():z.writestr(n,b)
        z.writestr('HANDOFF_FILES_SHA256.json',json.dumps(hashes))
    return base,cache,payload


def test_one_complete_zip_exact_bytes_no_test_directory_read(tmp_path,monkeypatch):
    base,cache,payload=fixture(tmp_path);original=Path.open
    def guard(path,*a,**kw):
        assert 'test' not in path.parts,'Attempt to access sealed test directory'
        return original(path,*a,**kw)
    monkeypatch.setattr(Path,'open',guard)
    out=assemble(base,cache,tmp_path/'final.zip',expected_count=2)
    with zipfile.ZipFile(out) as z:
        assert len([n for n in z.namelist() if n.endswith('features.npz')])==2
        assert json.loads(z.read('VALIDATION_CACHE_COPY_STATUS.json'))['status']=='COMPLETE'
        assert z.read('original_evidence.zip')==payload['original_evidence.zip']
        hashes=json.loads(z.read('HANDOFF_FILES_SHA256.json'))
        for n,digest in hashes.items():assert hashlib.sha256(z.read(n)).hexdigest()==digest
    with pytest.raises(ValueError):assemble(base,cache,out,expected_count=2)


@pytest.mark.parametrize('defect',['test','duplicate','hash','missing','symlink'])
def test_fail_closed_without_publishing_or_regenerating(tmp_path,defect):
    base,cache,_=fixture(tmp_path,defect);out=tmp_path/'final.zip'
    with pytest.raises(ValueError):assemble(base,cache,out,expected_count=2)
    assert not out.exists()


def test_recovery_extracts_originals_and_preserves_existing_files(tmp_path):
    from scripts.dayC_member1.restore_and_finish_member3 import restore
    base,_,payload=fixture(tmp_path);project=tmp_path/'RESTORED_TEST_FIXTURE'
    digest=hashlib.sha256(base.read_bytes()).hexdigest()
    restore(base,project,digest);restore(base,project,digest)
    assert (project/'original_evidence.zip').read_bytes()==payload['original_evidence.zip']
    (project/'original_evidence.zip').write_bytes(b'USER_CHANGED_FILE')
    with pytest.raises(RuntimeError):restore(base,project,digest)
    assert (project/'original_evidence.zip').read_bytes()==b'USER_CHANGED_FILE'


def test_recovery_wrong_zip_rejected_before_project_creation(tmp_path):
    from scripts.dayC_member1.restore_and_finish_member3 import restore
    base,_,_=fixture(tmp_path);project=tmp_path/'NEVER_CREATED'
    with pytest.raises(RuntimeError):restore(base,project,'a'*64)
    assert not project.exists()


def test_authorized_drive_root_alias_allowed_with_original_hashes(tmp_path):
    base,cache,_=fixture(tmp_path)
    alias=tmp_path/'DRIVE_PROJECT_SHORTCUT';alias.symlink_to(cache,target_is_directory=True)
    out=assemble(base,alias,tmp_path/'final.zip',expected_count=2)
    assert out.is_file()
