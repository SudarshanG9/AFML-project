"""TEST_FIXTURE exact-byte restoration only; no real assets or inference."""
import hashlib
from pathlib import Path
import zipfile
import pytest
from scripts.dayC_member1 import restore_verified_inputs as restore


def h(data):return hashlib.sha256(data).hexdigest()


def test_restore_originals_from_base_zip_and_preserve_matching_destinations(tmp_path):
    archive=tmp_path/'base.zip';data=b'TEST_FIXTURE original bytes, not historical provenance'
    with zipfile.ZipFile(archive,'x') as z:z.writestr('base/accepted_g5_context.json',data)
    target=tmp_path/'original_run/execution_context.json'
    spec={'context':{'path':str(target),'sha256':h(data),'names':['accepted_g5_context.json']}}
    report=restore.restore_from_backups(tmp_path,spec,[archive])
    assert report[0]['status']=='RESTORED_EXACT_ORIGINAL_BYTES' and target.read_bytes()==data
    stamp=target.stat().st_mtime_ns
    report=restore.restore_from_backups(tmp_path,spec,[])
    assert report[0]['status']=='PRESERVED_MATCHING_FILE' and target.stat().st_mtime_ns==stamp


def test_missing_original_prevents_all_writes(tmp_path):
    archive=tmp_path/'base.zip';data=b'TEST_FIXTURE'
    with zipfile.ZipFile(archive,'x') as z:z.writestr('one.json',data)
    targets={name:{'path':str(tmp_path/name),'sha256':h(data),'names':[name+'.json']} for name in ('one','two')}
    with pytest.raises(FileNotFoundError):restore.restore_from_backups(tmp_path,targets,[archive])
    assert not (tmp_path/'one').exists() and not (tmp_path/'two').exists()


def test_conflicting_destination_is_never_overwritten(tmp_path):
    target=tmp_path/'one.json';target.write_bytes(b'preserve these bytes')
    spec={'one':{'path':str(target),'sha256':h(b'other'),'names':['one.json']}}
    with pytest.raises(FileExistsError):restore.restore_from_backups(tmp_path,spec,[])
    assert target.read_bytes()==b'preserve these bytes'


def test_wrong_source_hash_never_becomes_restored_file(tmp_path):
    with pytest.raises(ValueError):restore.write_original(tmp_path,Path('one'),b'wrong','0'*64)
    assert not (tmp_path/'one').exists()


@pytest.mark.parametrize('kind',['escape','symlink'])
def test_destination_path_safety(tmp_path,kind):
    if kind=='escape':target=tmp_path.parent/'outside.json'
    else:
        target=tmp_path/'link';target.symlink_to(tmp_path/'not_created')
    with pytest.raises(PermissionError):restore.write_original(tmp_path,target,b'TEST_FIXTURE',h(b'TEST_FIXTURE'))


def test_archive_test_members_and_traversal_are_not_sources(tmp_path):
    data=b'TEST_FIXTURE';archive=tmp_path/'backup.zip'
    with zipfile.ZipFile(archive,'x') as z:
        z.writestr('../context.json',data);z.writestr('data/test/context.json',data)
    targets={'one':{'path':str(tmp_path/'context.json'),'sha256':h(data),'names':['context.json']}}
    with pytest.raises(FileNotFoundError):restore.restore_from_backups(tmp_path,targets,[archive])
    assert not (tmp_path/'context.json').exists()


def test_exclusive_write_race_preserves_other_file(tmp_path,monkeypatch):
    target=tmp_path/'race';original=restore.os.link
    def competing(source,destination):
        Path(destination).write_bytes(b'other writer')
        return original(source,destination)
    monkeypatch.setattr(restore.os,'link',competing)
    with pytest.raises(FileExistsError):restore.write_original(tmp_path,target,b'ours',h(b'ours'))
    assert target.read_bytes()==b'other writer'
    assert not list(tmp_path.glob('.restore_*.tmp'))
