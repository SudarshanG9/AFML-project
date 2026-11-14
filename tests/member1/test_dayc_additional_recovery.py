"""Further-attempt protocol on synthetic incidents only. Production guards never exercised on real data."""
import hashlib,json,zipfile,importlib.util
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pytest
from avfuse.member1_fusion import dayc_additional_recovery as a
from avfuse.member1_fusion.dayc_guards import write_new_json as save,read_strict_json as read,sha256_file as sha

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('FURTHER_TEST_BASE',ROOT/'tests/member1/test_dayc_incident_recovery.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)


@pytest.fixture
def incident(tmp_path,monkeypatch):
    actual_verify=a.verify_previous_recovery
    monkeypatch.setattr(a,'verify_previous_recovery',lambda root:None)
    monkeypatch.setattr(base,'r',a)
    i=base.incident.__wrapped__(tmp_path,monkeypatch);root=i['root'];entry=lambda p:{'path':str(p.relative_to(root)),'sha256':sha(p)}
    prior=root/'artifacts/dayC/member1/recovery';prior.mkdir();run=prior/a.PREVIOUS_RECOVERY_ID;run.mkdir()
    lock=prior/'recovery_attempt.lock';lock.write_text(a.PREVIOUS_RECOVERY_ID+'\n')
    proposal=root/'prev_proposal.json';save(proposal,{'TEST_FIXTURE':'previous recovery'})
    c9=root/'prev_c9.json';base.receipt(c9,'C9',sha(proposal))
    context=root/'prev_context.json';save(context,{'contract':read(proposal),'proposal':entry(proposal),'amendment_c9_approval':entry(c9)})
    anchor=root/'prev_anchor.json';save(anchor,{'context':entry(context)})
    package=root/'prev_package.zip';package.write_bytes(b'TEST_FIXTURE_PREVIOUS_PACKAGE')
    digest=a.digest_object({'recovery_context_sha256':sha(context),'recovery_anchor_sha256':sha(anchor)})
    approval=root/'prev_approval.json';base.receipt(approval,'C10',digest,execution_package_sha256=sha(package),execution_package_manifest_sha256='d'*64)
    first={'event':'RECOVERY_ATTEMPT_OPENED','run_id':a.PREVIOUS_RECOVERY_ID,'parent_run_id':a.PARENT_ID,'approval_sha256':sha(approval),'context_sha256':digest,'package_sha256':sha(package),'package_manifest_sha256':'d'*64,'parent_archive_sha256':a.PARENT['archive']['sha256']}
    failure={'event':'RECOVERY_ATTEMPT_FAILED','run_id':a.PREVIOUS_RECOVERY_ID,'parent_run_id':a.PARENT_ID,'mode_counts':dict.fromkeys(a.MODES,0),'error':'KeyboardInterrupt: '}
    save(run/'failure.json',failure);save(run/'authorization_bindings.json',first)
    (run/'recovery_approval.json').write_bytes(approval.read_bytes())
    ledger=prior/'recovery_attempt_ledger.jsonl';ledger.write_text(json.dumps(first)+'\n'+json.dumps(failure)+'\n')
    archive=root/'prev_archive.zip';payload={}
    for key,item in a.PARENT.items():payload['parent/'+key+'/'+Path(item['path']).name]=(root/item['path']).read_bytes()
    for name,path in [('recovery/recovery_attempt.lock',lock),('recovery/recovery_attempt_ledger.jsonl',ledger),('recovery/frozen_recovery_context.json',context),('recovery/recovery_anchor.json',anchor),('recovery/recovery_approval.json',approval),('run/failure.json',run/'failure.json'),('run/authorization_bindings.json',run/'authorization_bindings.json'),('run/recovery_approval.json',run/'recovery_approval.json')]:payload[name]=path.read_bytes()
    payload['output_hashes.json']=json.dumps({n:hashlib.sha256(b).hexdigest() for n,b in payload.items()}).encode()
    with zipfile.ZipFile(archive,'x') as z:
        for n,b in payload.items():z.writestr(n,b)
    previous={k:entry(p) for k,p in [('archive',archive),('lock',lock),('ledger',ledger),('context',context),('anchor',anchor),('package',package),('approval',approval)]}
    monkeypatch.setattr(a,'PREVIOUS',previous);monkeypatch.setattr(a,'verify_previous_recovery',actual_verify)
    proposal=root/'next_proposal.json';a.propose(root,i['proof'],proposal)
    c9=root/'next_c9.json';base.receipt(c9,'C9',sha(proposal));ctx,anc=a.finalize(root,proposal,c9,root/'NEXT_CONTEXT')
    _,digest=a.verify_amendment(root,ctx,anc);new=root/'next_receipt.json'
    base.receipt(new,'C10',digest,approval_kind='C10_INCIDENT_ADDITIONAL_RECOVERY',parent_run_id=a.PARENT_ID,recovery_run_id=a.RECOVERY_ID,max_additional_attempts=1,parent_evidence_sha256={k:v['sha256'] for k,v in a.PARENT.items()},execution_package_sha256=sha(i['archive']),execution_package_manifest_sha256='b'*64,previous_recovery_evidence_sha256={k:v['sha256'] for k,v in previous.items()},prior_failed_attempts_acknowledged=2,overall_attempt_limit=3,manual_interruption_acknowledged=True)
    i.update(context=ctx,anchor=anc,approval=new,proposal=proposal,previous=previous);return i


def execute(i):return a.execute_recovery(i['root'],i['context'],i['anchor'],i['approval'],i['archive'])


def test_both_failed_histories_verified_then_one_new_claim(incident):
    i=incident;old={k:(i['root']/v['path']).read_bytes() for k,v in i['previous'].items()}
    out=execute(i);assert i['adapter'].calls==8
    audit=a.audit_recovery(i['root'],i['context'],i['anchor'],i['approval'],i['archive']);assert audit['total_attempts']==3
    review=a.return_review(i['root'],i['context'],i['anchor'],i['approval'])
    with zipfile.ZipFile(review) as z:assert any(n.startswith('previous_recovery/ledger/') for n in z.namelist())
    for k,v in i['previous'].items():assert (i['root']/v['path']).read_bytes()==old[k]
    with pytest.raises(ValueError):execute(i)


@pytest.mark.parametrize('key',['archive','lock','ledger','context','anchor','package','approval'])
def test_any_prior_recovery_tamper_rejected_before_new_claim(incident,key):
    i=incident;(i['root']/i['previous'][key]['path']).write_bytes(b'TAMPERED')
    with pytest.raises(ValueError):execute(i)
    assert not a.recovery_paths(i['root'])[0].exists()


@pytest.mark.parametrize('field',['approval_kind','previous_recovery_evidence_sha256','overall_attempt_limit','manual_interruption_acknowledged','prior_failed_attempts_acknowledged'])
def test_old_or_incomplete_receipt_cannot_authorize_further_attempt(incident,field):
    i=incident;record=read(i['approval']);record.pop(field);i['approval'].write_text(json.dumps(record))
    with pytest.raises(Exception):execute(i)
    assert not a.recovery_paths(i['root'])[0].exists()


def test_successful_or_scored_previous_recovery_is_not_eligible(incident):
    i=incident;p=i['root']/i['previous']['ledger']['path'];p.write_text('SUCCESSFUL_OR_SCORED')
    with pytest.raises(ValueError):a.verify_previous_recovery(i['root'])


def test_old_consumed_recovery_receipt_rejected(incident):
    i=incident;i['approval'].write_bytes((i['root']/i['previous']['approval']['path']).read_bytes())
    with pytest.raises(Exception):execute(i)
    assert not a.recovery_paths(i['root'])[0].exists()


def test_failure_is_permanent_without_another_allowance(incident,monkeypatch):
    i=incident;monkeypatch.setattr(a,'inspect_cache_identity',lambda *args:(_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):execute(i)
    with pytest.raises(ValueError):execute(i)
    assert i['adapter'].calls==0


def test_further_claim_exclusive_under_concurrency(incident):
    root=incident['root']
    def claim():
        try:a.claim_recovery(root,{});return True
        except FileExistsError:return False
    with ThreadPoolExecutor(max_workers=2) as pool:assert sum(pool.map(lambda _:claim(),range(2)))==1


def test_further_package_contains_both_prior_histories(incident):
    i=incident;built=a.build_package(i['root'],i['context'],i['anchor'],i['root']/'next.zip')
    assert built['previous_recovery_evidence_sha256']=={k:v['sha256'] for k,v in i['previous'].items()}
    with zipfile.ZipFile(built['archive']) as z:
        assert a.PACKAGE_PREFIX+'/'+i['previous']['ledger']['path'] in z.namelist()
        assert a.PACKAGE_PREFIX+'/'+a.PARENT['ledger']['path'] in z.namelist()
