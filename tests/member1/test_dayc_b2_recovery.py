"""Synthetic safeguards for preflight-only Colab remediation; no remote access."""
import hashlib,json,zipfile
from pathlib import Path
import pytest
from scripts.dayC_member1.recover_b2_colab_cell import (
    assert_no_model_attempt,verify_or_extract_package,verify_bundle,bundle_recovery)


@pytest.mark.parametrize('marker',[None,'lock','output','attempt','benchmark_attempt'])
def test_recovery_refuses_any_model_attempt(tmp_path,marker):
    day8=tmp_path/'artifacts/dayC/member1/day8';day8.mkdir(parents=True)
    failed=day8/'B2_FAILED';failed.mkdir();(failed/'failure.json').write_text('{"stage":"full_validation_preflight"}')
    paths={'lock':day8/'full_validation_execution.lock','output':tmp_path/'artifacts/predictions/restore_query_val.jsonl',
           'attempt':failed/'validation/attempt_started.json','benchmark_attempt':failed/'benchmarks/short_attempt_started.json'}
    if marker:
        p=paths[marker];p.parent.mkdir(parents=True,exist_ok=True);p.write_text('preserve')
        with pytest.raises(RuntimeError):assert_no_model_attempt(tmp_path)
        assert p.read_text()=='preserve'
    else:assert_no_model_attempt(tmp_path)


def package(tmp_path):
    archive=tmp_path/'source.zip';addon=tmp_path/'TEST_FIXTURE'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('TEST_FIXTURE/scripts/entry.py','print("fixture")')
        z.writestr('TEST_FIXTURE/config.json','{}')
    return archive,addon,hashlib.sha256(archive.read_bytes()).hexdigest()


@pytest.mark.parametrize('defect',[None,'changed','extra_source','symlink','zip_hash'])
def test_package_reuse_is_byte_exact_and_non_overwriting(tmp_path,defect):
    archive,addon,sha=package(tmp_path);verify_or_extract_package(archive,addon,sha)
    if defect=='changed':(addon/'config.json').write_text('changed')
    elif defect=='extra_source':(addon/'scripts/extra.py').write_text('extra')
    elif defect=='symlink':
        p=addon/'config.json';p.unlink();p.symlink_to(addon/'scripts/entry.py')
    elif defect=='zip_hash':sha='0'*64
    if defect is None:verify_or_extract_package(archive,addon,sha)
    else:
        with pytest.raises(ValueError):verify_or_extract_package(archive,addon,sha)


def test_recovery_bundle_preserves_original_and_all_hashes(tmp_path):
    original=tmp_path/'original.zip';data=b'{"status":"TEST_FIXTURE"}'
    hashes={'failure.json':hashlib.sha256(data).hexdigest()}
    with zipfile.ZipFile(original,'w') as z:
        z.writestr('failure.json',data);z.writestr('output_hashes.json',json.dumps(hashes))
    before=original.read_bytes();out=tmp_path/'new.zip';receipt={'status':'TEST_FIXTURE'}
    bundle_recovery(original,receipt,out)
    assert original.read_bytes()==before
    result=verify_bundle(out)
    assert result['failure.json']==data
    assert json.loads(result['recovery/exact_input_restoration.json'])==receipt
    assert json.loads(result['recovery/original_output_hashes.json'])==hashes
    with pytest.raises(FileExistsError):bundle_recovery(original,receipt,out)
