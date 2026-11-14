"""Synthetic incident protocol fixtures. No production receipts or real model calls."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib,json,shutil,zipfile
from pathlib import Path
import numpy as np
import pytest
import torch
from avfuse.member1_fusion import dayc_recovery as r
from avfuse.member1_fusion.dayc_guards import sha256_file as sha,write_new_json as save,read_strict_json as read,GATE_ACKS


def receipt(path,gate,context,**extra):
    # A synthetic parent whose hashes differ from production PARENT is patched only
    # within this pytest fixture. These records cannot authorize the real incident.
    save(path,dict(gate=gate,ack_text=GATE_ACKS[gate],context_sha256=context,user_supplied=True,
        approved_at=datetime.now(timezone.utc).isoformat(),acknowledgement_source='TEST_FIXTURE_ONLY',**extra))


@pytest.fixture
def incident(tmp_path,monkeypatch):
    root=tmp_path/'TEST_FIXTURE';root.mkdir();origin=Path(__file__).resolve().parents[2]
    for name in r.SOURCES:
        target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((origin/name).read_bytes())
    model=root/'model/src/models.py';model.parent.mkdir(parents=True);model.write_text('# TEST_FIXTURE')
    (root/'model/config.json').write_text('{}');(root/'model/weights.pth').write_bytes(b'TEST_FIXTURE_WEIGHTS')
    rows=[];cache=root/'CACHE'
    for index in range(2):
        clip='TEST_FIXTURE_'+str(index);folder=cache/'test'/clip;folder.mkdir(parents=True)
        np.savez(folder/'features.npz',video_features=np.zeros((3,768),np.float32),audio_features=np.zeros((4,768),np.float32))
        save(folder/'meta.json',{'status':'ok','clip_id':clip})
        rows.append(dict(clip_id=clip,dataset='TEST_FIXTURE',source_id=clip,identity_id=clip,split='test',label=0,manipulation_type='real',feature_path='UNCHANGED.pt'))
    rows.append(dict(rows[0],clip_id='TEST_FIXTURE_VAL',source_id='VAL',identity_id='VAL',split='val'))
    monkeypatch.setattr(r,'path_resolution',lambda root:{'logical_root':str(cache),'physical_root':str(cache),'scope':'TEST_FIXTURE_ONLY'})
    canonical=root/'canonical.jsonl';canonical.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    entry=lambda p:{'path':str(p.relative_to(root)),'sha256':sha(p)}
    registry={'canonical_manifest':entry(canonical),'ready_test_manifest':entry(canonical),'model_source':entry(model),'checkpoint':entry(root/'model/weights.pth'),'model_config':entry(root/'model/config.json')}
    policy=root/'policy.json';save(policy,{'TEST_FIXTURE':'UNCHANGED_SCIENTIFIC_POLICY'})
    frozen={'inputs':registry,'checkpoint_sha256':'e1e59006d1bb6ad36a3859af5486a6649ed41bba23258b7c41e24f873be7948e','mode_order':list(r.MODES),'fixed_threshold':0.6627789735794068,'heldout_cohort':{'kind':'canonical_split','split':'test','manifest_sha256':sha(canonical),'expected_count':2},'feature_path_policy':{'kind':'split_clip_cache','cache_root':str(cache),'split':'test','feature_filename':'features.npz','metadata_filename':'meta.json'},'runtime':{'TEST_FIXTURE':True},'device':'cpu','backend':'sdpa','batch_size':1,'layers':5,'precision':'TEST_FIXTURE_FAKE_INT8','expected_count':2,'known_test_label_summary_access':True}
    run=root/'run.json';save(run,frozen);anchor=root/'anchor.json';save(anchor,{'TEST_FIXTURE':True})
    oldapproval=root/'old_approval.json';receipt(oldapproval,'C10','a'*64)
    package=root/'original.zip'
    with zipfile.ZipFile(package,'x') as z:
        for name in ['canonical.jsonl','model/config.json','model/src/models.py']:z.write(root/name,'AFML_DayC_Day10_Frozen/'+name)
        z.writestr('AFML_DayC_Day10_Frozen/DAY10_LAUNCH.json','{}')
    day10=root/'artifacts/dayC/member1/day10';day10.mkdir(parents=True);parent=day10/r.PARENT_ID;parent.mkdir()
    lock=day10/'test_attempt.lock';lock.write_text(r.PARENT_ID+'\n')
    opened={'event':'TEST_ATTEMPT_OPENED','run_id':r.PARENT_ID,'frozen_policy_sha256':sha(policy),'frozen_run_sha256':sha(run),'freeze_anchor_sha256':sha(anchor),'approved_c10_ack_sha256':sha(oldapproval)}
    failure={'event':'TEST_ATTEMPT_FAILED','status':'FAILED','run_id':r.PARENT_ID,'mode_counts':dict.fromkeys(r.MODES,0),'automatic_retry_permitted':False}
    save(parent/'attempt_opened.json',opened);save(parent/'attempt_failed.json',failure)
    ledger=day10/'test_attempt_ledger.jsonl';ledger.write_text(json.dumps(opened)+'\n'+json.dumps(failure)+'\n')
    archive=root/'failed.zip'
    payload={'run/attempt_opened.json':(parent/'attempt_opened.json').read_bytes(),'run/attempt_failed.json':(parent/'attempt_failed.json').read_bytes(),'ledger/test_attempt.lock':lock.read_bytes(),'ledger/test_attempt_ledger.jsonl':ledger.read_bytes(),'approval/C10_ack.json':oldapproval.read_bytes()}
    hashes={n:hashlib.sha256(b).hexdigest() for n,b in payload.items()}
    with zipfile.ZipFile(archive,'x') as z:
        for n,b in payload.items():z.writestr(n,b)
        z.writestr('output_hashes.json',json.dumps(hashes))
    mapping={k:entry(p) for k,p in [('archive',archive),('lock',lock),('ledger',ledger),('policy',policy),('run',run),('anchor',anchor),('package',package),('approval',oldapproval)]}
    monkeypatch.setattr(r,'PARENT',mapping)
    monkeypatch.setattr(r,'verify_release',lambda *a,**kw:{'status':'AUTHORIZED_METADATA_ONLY'})
    def registry_check(project,values):
        for item in values.values():r.inside(project,item)
    monkeypatch.setattr(r,'verify_registry',registry_check)
    junit=root/'junit.xml';junit.write_text('<TEST_FIXTURE/>');log=root/'log';log.write_text('TEST_FIXTURE')
    proof=root/'proof.json';save(proof,{'status':'PASS','passed':1,'failed':0,'errors':0,'skipped':0,'source_sha256':{n:sha(root/n) for n in r.SOURCES},'junit':entry(junit),'log':entry(log)})
    proposal=root/'proposal.json';r.propose(root,proof,proposal)
    c9=root/'c9.json';receipt(c9,'C9',sha(proposal));context,amend_anchor=r.finalize(root,proposal,c9,root/'versioned_amendment')
    _,digest=r.verify_amendment(root,context,amend_anchor)
    packet=root/'recovery.zip';packet.write_bytes(b'TEST_FIXTURE_EXECUTION_ZIP');manifest='b'*64
    approval=root/'recovery_approval.json'
    receipt(approval,'C10',digest,approval_kind='C10_INCIDENT_RECOVERY',parent_run_id=r.PARENT_ID,recovery_run_id=r.RECOVERY_ID,
        parent_evidence_sha256={k:v['sha256'] for k,v in mapping.items()},max_additional_attempts=1,
        execution_package_sha256=sha(packet),execution_package_manifest_sha256=manifest)
    monkeypatch.setattr(r,'verify_package',lambda root:manifest)
    from scripts.dayC_member1 import run_day8,colab_query_pilot
    monkeypatch.setattr(run_day8,'observed_runtime',lambda:frozen['runtime']);monkeypatch.setattr(colab_query_pilot,'check_environment',lambda *a:None)
    from scripts import dayB_inference
    class Adapter:
        checkpoint_path=root/'model/weights.pth';config_path=root/'model/config.json';checkpoint_sha256=frozen['checkpoint_sha256']
        calls=0
        def predict_from_cached_features(self,features,mode):self.calls+=1;return torch.tensor([0.25])
    adapter=Adapter();monkeypatch.setattr(dayB_inference,'create_adapter',lambda *a:adapter)
    from avfuse.member1_fusion import dayc_precision_audit
    monkeypatch.setattr(dayc_precision_audit,'check_model_invariants',lambda *a,**kw:{'state_dict_sha256':'c'*64})
    return dict(root=root,context=context,anchor=amend_anchor,approval=approval,archive=packet,adapter=adapter,cache=cache,proposal=proposal,proof=proof,mapping=mapping,frozen=frozen)


def execute(i):return r.execute_recovery(i['root'],i['context'],i['anchor'],i['approval'],i['archive'])


def test_valid_parent_and_distinct_single_recovery_audit(incident):
    i=incident;parent_bytes={k:(i['root']/v['path']).read_bytes() for k,v in i['mapping'].items()}
    out=execute(i);assert i['adapter'].calls==8
    report=r.audit_recovery(i['root'],i['context'],i['anchor'],i['approval'],i['archive'])
    assert report['parent_failed_attempts']==report['recovery_attempts']==1 and report['count_per_mode']==2
    review=r.return_review(i['root'],i['context'],i['anchor'],i['approval'])
    with zipfile.ZipFile(review) as z:
        assert 'parent/ledger/test_attempt_ledger.jsonl' in z.namelist() and 'recovery/recovery_attempt_ledger.jsonl' in z.namelist()
        hashes=json.loads(z.read('output_hashes.json'))
        for n,h in hashes.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
    for k,v in i['mapping'].items():assert (i['root']/v['path']).read_bytes()==parent_bytes[k]
    shutil.rmtree(out)
    with pytest.raises(ValueError):execute(i)
    assert i['adapter'].calls==8


@pytest.mark.parametrize('defect',['lock_missing','ledger_missing','ledger_tamper','archive_checksum','policy','parent_success','nonzero','incomplete','unknown_output'])
def test_parent_incident_tampering_or_wrong_state_rejected(incident,defect):
    i=incident;root=i['root']
    if defect in ['lock_missing','ledger_missing']:(root/i['mapping'][defect.split('_')[0]]['path']).unlink()
    elif defect=='unknown_output':
        p=root/'artifacts/predictions/unknown_test.jsonl';p.parent.mkdir();p.write_text('TEST_FIXTURE')
    else:
        key={'ledger_tamper':'ledger','archive_checksum':'archive','policy':'policy'}.get(defect)
        if key:(root/i['mapping'][key]['path']).write_bytes(b'TAMPERED')
        else:
            # Update the fixture hash only to exercise semantic checks, never production constants.
            arc=root/i['mapping']['archive']['path']
            with zipfile.ZipFile(arc) as z:payload={n:z.read(n) for n in z.namelist()}
            failure=json.loads(payload['run/attempt_failed.json']);entries=[json.loads(l) for l in payload['ledger/test_attempt_ledger.jsonl'].splitlines()]
            if defect=='parent_success':entries[1]['event']='TEST_ATTEMPT_COMPLETED'
            elif defect=='nonzero':failure['mode_counts']['reference']=1;entries[1]['mode_counts']['reference']=1
            elif defect=='incomplete':entries=entries[:1]
            payload['run/attempt_failed.json']=json.dumps(failure).encode();payload['ledger/test_attempt_ledger.jsonl']=''.join(json.dumps(e)+'\n' for e in entries).encode()
            (root/i['mapping']['ledger']['path']).write_bytes(payload['ledger/test_attempt_ledger.jsonl'])
            payload['output_hashes.json']=json.dumps({n:hashlib.sha256(b).hexdigest() for n,b in payload.items() if n!='output_hashes.json'}).encode()
            with zipfile.ZipFile(arc,'w') as z:
                for n,b in payload.items():z.writestr(n,b)
            r.PARENT['archive']['sha256']=sha(arc);r.PARENT['ledger']['sha256']=sha(root/i['mapping']['ledger']['path'])
    with pytest.raises((ValueError,OSError)):r.verify_parent(root)
    assert not r.recovery_paths(root)[0].exists()


@pytest.mark.parametrize('defect',['missing','old_receipt','context','package','manifest','parent','additional','runid','synthetic_marker'])
def test_bad_authorization_fails_before_data_or_claim(incident,monkeypatch,defect):
    i=incident;path=i['approval']
    if defect=='missing':path.unlink()
    elif defect=='old_receipt':path.write_bytes((i['root']/i['mapping']['approval']['path']).read_bytes())
    else:
        record=read(path)
        if defect in ['context','package','manifest']:record[{'context':'context_sha256','package':'execution_package_sha256','manifest':'execution_package_manifest_sha256'}[defect]]='f'*64
        elif defect=='parent':record['parent_run_id']='UNKNOWN'
        elif defect=='additional':record['max_additional_attempts']=2
        elif defect=='runid':record['recovery_run_id']='CHANGED'
        else:record['synthetic_fixture']=True
        path.write_text(json.dumps(record))
    monkeypatch.setattr(r,'read_frozen_cohort',lambda *a:pytest.fail('Data access before valid approval'))
    with pytest.raises(Exception):execute(i)
    assert not r.recovery_paths(i['root'])[0].exists() and i['adapter'].calls==0


def test_exclusive_claim_and_concurrent_claims(incident):
    root=incident['root']
    def claim():
        try:r.claim_recovery(root,{});return 'claimed'
        except FileExistsError:return 'blocked'
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(lambda _:claim(),range(2)))
    assert sorted(results)==['blocked','claimed']
    with pytest.raises(FileExistsError):r.claim_recovery(root,{})


def test_failure_after_claim_retains_lock_and_no_third_attempt(incident,monkeypatch):
    i=incident
    monkeypatch.setattr(r,'inspect_cache_identity',lambda *a:(_ for _ in ()).throw(ValueError('TEST_FIXTURE cache failure')))
    with pytest.raises(ValueError):execute(i)
    lock,ledger,out=r.recovery_paths(i['root']);assert lock.exists()
    assert [json.loads(l)['event'] for l in ledger.read_text().splitlines()]==['RECOVERY_ATTEMPT_OPENED','RECOVERY_ATTEMPT_FAILED']
    assert i['adapter'].calls==0
    shutil.rmtree(out)
    with pytest.raises(ValueError):execute(i)


def test_failure_during_scoring_is_permanent(incident):
    i=incident
    def fail(features,mode):raise RuntimeError('TEST_FIXTURE model failure')
    i['adapter'].predict_from_cached_features=fail
    with pytest.raises(RuntimeError):execute(i)
    with pytest.raises(ValueError):execute(i)


@pytest.mark.parametrize('field',['ledger','prediction','cache'])
def test_saved_recovery_audit_detects_corruption(incident,field):
    i=incident;out=execute(i)
    if field=='ledger':r.recovery_paths(i['root'])[1].write_text('{"event":"CORRUPT"}\n')
    elif field=='prediction':(out/'predictions/reference_test.jsonl').write_text('CORRUPT')
    else:(i['cache']/'test/TEST_FIXTURE_0/meta.json').write_text('{}')
    with pytest.raises((ValueError,KeyError)):r.audit_recovery(i['root'],i['context'],i['anchor'],i['approval'],i['archive'])


def test_wrong_canonical_contract_and_new_source_drift_rejected(incident):
    i=incident;p=read(i['proposal']);p['scientific_contract']['expected_count']=1
    with pytest.raises(ValueError):r.validate_proposal(i['root'],p)
    (i['root']/r.SOURCES[0]).write_text('# CHANGED')
    with pytest.raises(ValueError):execute(i)


def test_package_build_requires_amendment_approval_and_detects_tamper(incident):
    i=incident;out=i['root']/'built.zip'
    with pytest.raises(FileNotFoundError):r.build_package(i['root'],i['root']/'NOT_FROZEN',i['anchor'],out)
    assert not out.exists()
    built=r.build_package(i['root'],i['context'],i['anchor'],out)
    with zipfile.ZipFile(out) as z:
        prefix=r.PACKAGE_PREFIX+'/'
        assert prefix+i['mapping']['ledger']['path'] in z.namelist()
        manifest=json.loads(z.read(prefix+'PACKAGE_MANIFEST.json'))
        for n,h in manifest['files'].items():assert hashlib.sha256(z.read(prefix+n)).hexdigest()==h
    extracted=i['root']/'EXTRACTED';extracted.mkdir()
    with zipfile.ZipFile(out) as z:z.extractall(extracted)
    from scripts.dayC_member1.colab_day10 import verify_package
    project=extracted/r.PACKAGE_PREFIX;verify_package(project)
    (project/r.SOURCES[0]).write_text('TAMPER')
    with pytest.raises(ValueError):verify_package(project)
    for action in ['prepare','run']:compile(r.colab_cell(out.name,built['execution_package_sha256'],action),'TEST_FIXTURE_CELL','exec')


def test_missing_amendment_approval_creates_no_freeze(incident):
    i=incident;out=i['root']/'UNAPPROVED'
    with pytest.raises(FileNotFoundError):r.finalize(i['root'],i['proposal'],i['root']/'NO_APPROVAL',out)
    assert not out.exists()


def test_runtime_failure_before_claim_opens_no_test_data(incident,monkeypatch):
    i=incident
    from scripts.dayC_member1 import colab_query_pilot
    monkeypatch.setattr(colab_query_pilot,'check_environment',lambda *a:(_ for _ in ()).throw(ValueError('TEST_FIXTURE runtime mismatch')))
    monkeypatch.setattr(r,'read_frozen_cohort',lambda *a:pytest.fail('Data before runtime/claim'))
    with pytest.raises(ValueError):execute(i)
    assert not r.recovery_paths(i['root'])[0].exists() and i['adapter'].calls==0


def test_recovery_lock_or_ledger_alone_blocks_execution(incident):
    i=incident;lock,ledger,_=r.recovery_paths(i['root']);ledger.parent.mkdir(parents=True)
    ledger.write_text('CORRUPT_OR_PARTIAL')
    with pytest.raises(ValueError):execute(i)
    assert not lock.exists() and i['adapter'].calls==0


def test_wrong_amendment_c9_receipt_never_finalizes(incident):
    i=incident;approval=i['root']/'stale_c9.json';receipt(approval,'C9','f'*64)
    out=i['root']/'NO_FROZEN_CONTEXT'
    with pytest.raises(PermissionError):r.finalize(i['root'],i['proposal'],approval,out)
    assert not out.exists()


def test_max_additional_attempts_is_integer_not_boolean(incident):
    i=incident;value=read(i['proposal']);value['max_additional_attempts']=True
    with pytest.raises(ValueError):r.validate_proposal(i['root'],value)


def test_success_history_cannot_be_hidden_by_removing_only_output(incident):
    i=incident;out=execute(i);lock,ledger,_=r.recovery_paths(i['root']);original=ledger.read_bytes()
    shutil.rmtree(out)
    with pytest.raises(ValueError):execute(i)
    assert ledger.read_bytes()==original and lock.exists()


def test_cache_root_metadata_guard_rejects_alias_before_claim(tmp_path):
    target=tmp_path/'TEST_FIXTURE_CACHE';target.mkdir();alias=tmp_path/'DRIVE_ALIAS';alias.symlink_to(target,target_is_directory=True)
    r.check_cache_root_metadata({'cache_root':str(target)})
    with pytest.raises(ValueError):r.check_cache_root_metadata({'cache_root':str(alias)})


def test_resolution_changes_only_lookup_not_scientific_policy(incident):
    from copy import deepcopy
    original=deepcopy(incident['frozen'])
    effective=r.effective_execution_contract(original,{'logical_root':original['feature_path_policy']['cache_root'],'physical_root':'/TEST_FIXTURE_PHYSICAL'})
    assert original==incident['frozen']
    assert effective['feature_path_policy']['cache_root']=='/TEST_FIXTURE_PHYSICAL'
    effective['feature_path_policy']['cache_root']=original['feature_path_policy']['cache_root']
    assert effective==original


def test_wrong_or_missing_resolution_cannot_be_approved(incident):
    p=read(incident['proposal']);p['cache_path_resolution']['physical_root']='/WRONG'
    with pytest.raises(ValueError):r.validate_proposal(incident['root'],p)


def test_alias_identity_mismatch_stops_before_claim(incident,monkeypatch):
    i=incident;monkeypatch.setattr(r,'check_resolution_metadata',lambda root:(_ for _ in ()).throw(ValueError('Alias retargeted')))
    with pytest.raises(ValueError):execute(i)
    assert not r.recovery_paths(i['root'])[0].exists()


def test_physical_root_allows_same_frozen_resolver_without_symlink_bypass(tmp_path):
    cache=tmp_path/'PHYSICAL';folder=cache/'test/TEST_FIXTURE';folder.mkdir(parents=True)
    (folder/'features.npz').write_bytes(b'TEST_FIXTURE');(folder/'meta.json').write_text('{}')
    alias=tmp_path/'SHORTCUT';alias.symlink_to(cache,target_is_directory=True)
    policy={'kind':'split_clip_cache','cache_root':str(cache),'split':'test','feature_filename':'features.npz','metadata_filename':'meta.json'}
    from avfuse.member1_fusion.dayc_heldout import resolve_test_feature
    assert resolve_test_feature(tmp_path,{'clip_id':'TEST_FIXTURE','split':'test'},policy)==folder/'features.npz'
    policy['cache_root']=str(alias)
    with pytest.raises(ValueError):resolve_test_feature(tmp_path,{'clip_id':'TEST_FIXTURE','split':'test'},policy)
