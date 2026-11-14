"""Synthetic progress/interrupt observability. Never runs a production attempt."""
import ast
import importlib.util
import json
from pathlib import Path
import threading
from types import SimpleNamespace
import hashlib
import pytest
from avfuse.member1_fusion.dayc_progress import ProgressReporter,fresh_review_from_stdout

ROOT=Path(__file__).resolve().parents[2]
CANDIDATE=ROOT/'artifacts/dayC/member1/recovery_engineering/PROGRESS_AND_INTERRUPTION/candidate_sources/src/avfuse/member1_fusion/dayc_recovery.py'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


@pytest.fixture
def candidate_incident(tmp_path,monkeypatch):
    candidate=load('AFML_TEST_FIXTURE_CANDIDATE',CANDIDATE)
    base=load('AFML_TEST_FIXTURE_BASE',ROOT/'tests/member1/test_dayc_incident_recovery.py')
    monkeypatch.setattr(base,'r',candidate)
    incident=base.incident.__wrapped__(tmp_path,monkeypatch)
    return candidate,incident


def test_heartbeat_during_blocking_phase_and_thread_shutdown(tmp_path):
    messages=[];seen=threading.Event()
    def output(text):
        messages.append(json.loads(text))
        if messages[-1]['event']=='HEARTBEAT':seen.set()
    with ProgressReporter(interval=0.01,output=output) as reporter:
        reporter.attach(tmp_path/'progress.jsonl')
        with reporter.phase('TEST_FIXTURE_SLOW_READ',clip_index=3):assert seen.wait(1)
    assert not reporter.thread.is_alive()
    assert any(m['event']=='HEARTBEAT' and m['phase']=='TEST_FIXTURE_SLOW_READ' for m in messages)
    assert messages[-1]['event']=='OPERATIONS_FINISHED'
    assert all(m['elapsed_seconds']>=0 for m in messages)
    assert len((tmp_path/'progress.jsonl').read_text().splitlines())==len(messages)


def test_keyboardinterrupt_logs_and_propagates_without_retry(tmp_path):
    messages=[]
    with pytest.raises(KeyboardInterrupt):
        with ProgressReporter(interval=0.01,output=lambda m:messages.append(json.loads(m))) as reporter:
            reporter.attach(tmp_path/'progress.jsonl')
            with reporter.phase('TEST_FIXTURE_CACHE_CHECK'):raise KeyboardInterrupt()
    assert not reporter.thread.is_alive()
    assert messages[-1]['event']=='OPERATION_INTERRUPTED'
    assert messages[-1]['error_type']=='KeyboardInterrupt'
    assert any(m['event']=='PHASE_FAILED' for m in messages)


def test_progress_log_never_overwrites(tmp_path):
    path=tmp_path/'existing';path.write_text('PRESERVE')
    with pytest.raises(FileExistsError):ProgressReporter().attach(path)
    assert path.read_text()=='PRESERVE'


def test_candidate_success_preserves_scores_and_produces_stable_progress_identity(candidate_incident):
    c,i=candidate_incident
    out=c.execute_recovery(i['root'],i['context'],i['anchor'],i['approval'],i['archive'])
    assert i['adapter'].calls==8
    rows=[json.loads(l) for l in (out/'progress.jsonl').read_text().splitlines()]
    assert any(v['event']=='CACHE_CHECK_COMPLETED' and v['verified_clips']==2 for v in rows)
    assert any(v.get('phase')=='model_initialization_and_frozen_state_verification' for v in rows)
    assert any(v['event']=='SCORING_PROGRESS' for v in rows)
    assert rows[-1]['event']=='OPERATIONS_FINISHED'
    identity=json.loads((out/'progress_identity.json').read_text())
    assert hashlib.sha256((out/'progress.jsonl').read_bytes()).hexdigest()==identity['sha256']
    report=c.audit_recovery(i['root'],i['context'],i['anchor'],i['approval'],i['archive'])
    assert report['count_per_mode']==2
    for mode in c.MODES:
        values=[json.loads(line) for line in (out/'predictions'/(mode+'_test.jsonl')).read_text().splitlines()]
        assert [v['score_fake'] for v in values]==[0.25,0.25]
    (out/'progress.jsonl').write_text('TAMPER')
    with pytest.raises(ValueError):c.audit_recovery(i['root'],i['context'],i['anchor'],i['approval'],i['archive'])


def test_candidate_interruption_preserves_failure_and_consumed_lock(candidate_incident,monkeypatch):
    c,i=candidate_incident
    monkeypatch.setattr(c,'inspect_cache_identity',lambda *a:(_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):c.execute_recovery(i['root'],i['context'],i['anchor'],i['approval'],i['archive'])
    lock,ledger,out=c.recovery_paths(i['root']);original=ledger.read_bytes()
    assert i['adapter'].calls==0 and lock.exists()
    assert json.loads((out/'failure.json').read_text())['error'].startswith('KeyboardInterrupt')
    assert json.loads((out/'progress.jsonl').read_text().splitlines()[-1])['event']=='OPERATION_INTERRUPTED'
    with pytest.raises(ValueError):c.execute_recovery(i['root'],i['context'],i['anchor'],i['approval'],i['archive'])
    assert ledger.read_bytes()==original and i['adapter'].calls==0


def test_candidate_invalid_authorization_claims_nothing(candidate_incident):
    c,i=candidate_incident;i['approval'].unlink()
    with pytest.raises(FileNotFoundError):c.execute_recovery(i['root'],i['context'],i['anchor'],i['approval'],i['archive'])
    assert not c.recovery_paths(i['root'])[0].exists()


@pytest.mark.parametrize('marker',[False,True])
def test_actual_generated_cell_downloads_only_new_emission(candidate_incident,tmp_path,marker):
    c,_=candidate_incident;script=c.colab_cell('TEST_FIXTURE.zip','a'*64,'run');compile(script,'CELL','exec')
    tree=ast.parse(script);start=next(i for i,node in enumerate(tree.body) if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='markers' for t in node.targets))
    nodes=tree.body[start:start+2]
    review=tmp_path/'OLD_REVIEW.zip';review.write_bytes(b'EXISTING_OLD_ARCHIVE');downloads=[]
    stdout=('RETURN REVIEW ZIP: '+str(review)+'\n') if marker else 'Traceback: already claimed\n'
    namespace={'Path':Path,'hashlib':hashlib,'lines':[stdout],'review':review,'files':SimpleNamespace(download=lambda p:downloads.append(p))}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'DOWNLOAD_GUARD','exec'),namespace)
    assert len(downloads)==int(marker)


def test_fresh_review_helper_rejects_stale_unmarked_and_escaped_files(tmp_path):
    root=tmp_path/'root';root.mkdir();old=root/'old.zip';old.write_bytes(b'OLD')
    assert fresh_review_from_stdout('Exit code: 1',root,old) is None
    assert fresh_review_from_stdout('RETURN REVIEW ZIP: '+str(old),root,old)==old
    other=tmp_path/'outside.zip';other.write_bytes(b'OTHER')
    with pytest.raises(ValueError):fresh_review_from_stdout('RETURN REVIEW ZIP: '+str(other),root,old)
