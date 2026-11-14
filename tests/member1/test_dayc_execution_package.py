"""Executable package tests. All freeze/run objects here are TEST_FIXTURE only."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
import pytest
from scripts.dayC_member1 import build_day10_package as b,colab_day10 as c
from avfuse.member1_fusion.dayc_guards import write_new_json,sha256_file
from avfuse.member1_fusion.dayc_freeze import SCORING_FILES


def test_builder_without_real_freeze_does_not_write_anything(tmp_path):
    destination=tmp_path/'artifacts/dayC/member1/colab/output.zip'
    with pytest.raises((OSError,PermissionError,ValueError)):
        b.build(tmp_path,destination,tmp_path/'policy',tmp_path/'run',tmp_path/'anchor')
    assert not destination.exists() and not (tmp_path/'artifacts').exists()


@pytest.mark.parametrize('name',['../escape','/absolute','a/../b','a//b'])
def test_unsafe_archive_names_are_refused(name):
    with pytest.raises(ValueError):b.safe_name(name)


@pytest.mark.parametrize('action',['prepare','run'])
def test_generated_colab_cells_compile_and_bind_actual_zip(action):
    source=b.colab_cell(Path('TEST_FIXTURE.zip'),'a'*64,action)
    compile(source,'TEST_FIXTURE_CELL','exec')
    assert 'a'*64 in source and 'files.download' in source
    assert 'pip install' not in source and 'drive.mount' not in source


def test_archive_verification_detects_tamper_and_never_overwrites(tmp_path):
    archive=tmp_path/'TEST_FIXTURE.zip';payload={'SEALED.jsonl':b'TEST_FIXTURE_OPAQUE_NOT_JSON','code.py':b'# TEST_FIXTURE'}
    digest=b.write_archive(archive,payload)
    with zipfile.ZipFile(archive) as z:z.extractall(tmp_path)
    root=tmp_path/b.NAME
    assert c.verify_package(root)==digest
    assert (root/'SEALED.jsonl').read_bytes()==payload['SEALED.jsonl']
    with pytest.raises(FileExistsError):b.write_archive(archive,payload)
    (root/'code.py').write_bytes(b'changed')
    with pytest.raises(ValueError):c.verify_package(root)


def test_checkpoint_staging_is_exact_and_never_overwrites(tmp_path):
    source=tmp_path/'TEST_FIXTURE_ORIGINAL.pth';source.write_bytes(b'TEST_FIXTURE_WEIGHTS')
    root=tmp_path/'TEST_FIXTURE_EXECUTION';root.mkdir()
    launch={'external_inputs':{'checkpoint':{'path':'model/weights.pth','sha256':sha256_file(source),'suggested_existing_source':str(source)}}}
    c.stage_checkpoint(root,launch);target=root/'model/weights.pth'
    assert target.read_bytes()==source.read_bytes()
    c.stage_checkpoint(root,launch)
    target.write_bytes(b'changed')
    with pytest.raises(ValueError):c.stage_checkpoint(root,launch)
    assert source.read_bytes()==b'TEST_FIXTURE_WEIGHTS' and target.read_bytes()==b'changed'


def test_package_has_complete_import_closure_without_original_checkout(tmp_path):
    original=Path(__file__).resolve().parents[2];root=tmp_path/'TEST_FIXTURE_SOURCE';root.mkdir()
    base=root/b.BASE;base.parent.mkdir(parents=True);shutil.copyfile(original/b.BASE,base)
    registry={}
    for name in SCORING_FILES:
        target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((original/name).read_bytes())
        registry['source:'+name]={'path':name,'sha256':sha256_file(target)}
    registry['checkpoint']={'path':'model/weights.pth','sha256':'a'*64}
    for name in ['configs/dayC/member1/C8A_context.json','policy.json','run.json','anchor.json','c9.json','proposal.json']:
        target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text('{"scope":"TEST_FIXTURE"}')
    frozen={'inputs':registry,'c9_approval':{'path':'c9.json'},'source_proposal':{'path':'proposal.json'}}
    payload=b.collect_payload(root,frozen,root/'policy.json',root/'run.json',root/'anchor.json')
    assert not any(Path(n).suffix in {'.pth','.pt','.npz'} for n in payload)
    archive=tmp_path/'TEST_FIXTURE.zip';b.write_archive(archive,payload)
    extraction=tmp_path/'EXTRACTED';extraction.mkdir()
    with zipfile.ZipFile(archive) as z:z.extractall(extraction)
    project=extraction/b.NAME
    # Isolated Python starts with only this extracted project on its source path.
    command=[sys.executable,'-I','-B','-c',
        "import sys;sys.path[:0]=["+repr(str(project/'src'))+","+repr(str(project))+"];from scripts.dayC_member1 import colab_day10;from avfuse.member1_fusion.dayc_heldout import execute_heldout;print('IMPORT_CLOSURE_PASS')"]
    result=subprocess.run(command,cwd=project,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'IMPORT_CLOSURE_PASS' in result.stdout


@pytest.fixture
def launch_fixture(tmp_path,monkeypatch):
    root=tmp_path/'TEST_FIXTURE';root.mkdir()
    for name in ('policy','run','anchor'):(root/(name+'.json')).write_text('{"scope":"TEST_FIXTURE"}')
    write_new_json(root/'DAY10_LAUNCH.json',{'policy':'policy.json','run':'run.json','anchor':'anchor.json'})
    approval=root/'approval.json';write_new_json(approval,{'execution_package_manifest_sha256':'b'*64,'execution_package_sha256':'c'*64})
    archive=root/'TEST_FIXTURE.zip';archive.write_bytes(b'TEST_FIXTURE')
    monkeypatch.setattr(c,'verify_package',lambda root:'b'*64)
    monkeypatch.setattr(c,'verify_release',lambda *a,**kw:{'C10_context_sha256':'d'*64})
    monkeypatch.setattr(c,'require_human_approval',lambda *a,**kw:None) # Receipt mechanics tested in guards/freeze suites.
    original=c.sha256_file
    monkeypatch.setattr(c,'sha256_file',lambda p:'c'*64 if Path(p)==archive else original(p))
    return root,approval,archive


def test_package_hash_mismatch_rejected_before_execution_or_staging(launch_fixture,monkeypatch):
    root,approval,archive=launch_fixture
    record=json.loads(approval.read_text());record['execution_package_sha256']='f'*64;approval.write_text(json.dumps(record))
    monkeypatch.setattr(c,'metadata_preflight',lambda *a:pytest.fail('staged before package ACK'))
    monkeypatch.setattr(c,'execute_heldout',lambda *a:pytest.fail('inference before package ACK'))
    with pytest.raises(PermissionError):c.run_authorized(root,approval,archive)
    assert not (root/'artifacts').exists()


def test_inference_failure_returns_evidence_zip_and_never_retries(launch_fixture,monkeypatch,capsys):
    root,approval,archive=launch_fixture;calls=[]
    registry={'canonical_manifest':{'path':'canon'},'ready_test_manifest':{'path':'canon'},'model_source':{'path':'model/src/models.py'},'member2_release':{'path':'m2'},'member3_release':{'path':'m3'}}
    monkeypatch.setattr(c,'metadata_preflight',lambda *a:({}, {'inputs':registry},{}))
    def fail(args):calls.append(args);raise RuntimeError('TEST_FIXTURE injected inference failure')
    monkeypatch.setattr(c,'execute_heldout',fail)
    monkeypatch.setattr(c,'audit_saved_outputs',lambda *a:pytest.fail('audit after failed inference'))
    assert c.run_authorized(root,approval,archive)==1 and len(calls)==1
    archives=list((root/'artifacts/dayC/member1/colab').glob('*/DAYC_C10_review.zip'));assert len(archives)==1
    with zipfile.ZipFile(archives[0]) as z:
        hashes=json.loads(z.read('output_hashes.json'))
        assert set(z.namelist())==set(hashes)|{'output_hashes.json'}
        assert any(name.endswith('failure.json') for name in hashes)
        for name,digest in hashes.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
    assert 'RETURN REVIEW ZIP:' in capsys.readouterr().out
    with pytest.raises(FileExistsError):c.run_authorized(root,approval,archive)
    assert len(calls)==1


def test_prepare_delegates_metadata_only_and_never_claims(launch_fixture,monkeypatch):
    root,_,_=launch_fixture
    frozen={'inputs':{'canonical_manifest':{'path':'NEVER_PARSE_TEST_ROWS'}},'runtime':{'TEST_FIXTURE':True}}
    monkeypatch.setattr(c,'read_strict_json',lambda p:frozen if Path(p).name=='run.json' else {'policy':'policy.json','run':'run.json','anchor':'anchor.json'})
    calls=[];monkeypatch.setattr(c,'stage_checkpoint',lambda *a:calls.append('stage_checkpoint'))
    monkeypatch.setattr(c,'verify_registry',lambda *a:calls.append('opaque_registry_hashes'))
    from scripts.dayC_member1 import run_day8,colab_query_pilot
    monkeypatch.setattr(run_day8,'observed_runtime',lambda:{'TEST_FIXTURE':True})
    monkeypatch.setattr(colab_query_pilot,'check_environment',lambda *a:calls.append('runtime'))
    monkeypatch.setattr(c,'execute_heldout',lambda *a:pytest.fail('inference in prepare'))
    _,_,result=c.metadata_preflight(root)
    assert result['C10_context_sha256']=='d'*64 and calls==['stage_checkpoint','opaque_registry_hashes','runtime']
    assert not (root/'artifacts').exists()
