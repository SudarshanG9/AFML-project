"""Incident-specific operational recovery. Original frozen sources/artifacts stay untouched."""
from datetime import datetime,timezone
from avfuse.member1_fusion.dayc_progress import ProgressReporter,fresh_review_from_stdout
import hashlib
import json
import os
from pathlib import Path
import zipfile
from avfuse.member1_fusion.dayc_guards import (ensure_inside_repo,read_strict_json,sha256_file,require_sha256,require_new_file,require_human_approval,write_new_json)
from avfuse.member1_fusion.dayc_freeze import require,digest_object,verify_registry,verify_release,prior_test_artifacts
from avfuse.member1_fusion.dayc_heldout import (append_event,safe_run_id,read_frozen_cohort,resolve_test_feature,score_complete_cohort,publish_complete_predictions,validate_saved_rows)
from avfuse.member1_fusion.dayc_validation import MODES
from avfuse.member1_fusion.dayc_test_contract import inspect_cache_identity

PARENT_ID='C10_20261113T191638908596Z'
RECOVERY_ID='C10_RECOVERY_R2_C10_20261113T191638908596Z'
PREVIOUS_RECOVERY_ID='C10_RECOVERY_R1_C10_20261113T191638908596Z'
PREVIOUS={'archive': {'path': 'artifacts/dayC/member1/recovery/C10_RECOVERY_R1_C10_20261113T191638908596Z/returned_evidence/DAYC_C10_RECOVERY_review.zip', 'sha256': '9b5cf3c891493904caf54739b733edb9b5255a15b3c02af245822f75bd22a79e'}, 'lock': {'path': 'artifacts/dayC/member1/recovery/recovery_attempt.lock', 'sha256': 'ffcac04b0eb38394108207a3aabbe2e472a39c9cdf63e8c5137a4e7184d44050'}, 'ledger': {'path': 'artifacts/dayC/member1/recovery/recovery_attempt_ledger.jsonl', 'sha256': 'c9f929ef071990a10f1124b81372bb366158528bc5f924fc1a3629a752e2026c'}, 'context': {'path': 'artifacts/dayC/member1/recovery_engineering/PATH_RESOLUTION_R2/APPROVED_R2_OPERATIONAL_FREEZE/frozen_recovery_context.json', 'sha256': '291b78efd093f49b7a57bcf3baa620e8308ba1481a8789d38c73c8817ac8fce1'}, 'anchor': {'path': 'artifacts/dayC/member1/recovery_engineering/PATH_RESOLUTION_R2/APPROVED_R2_OPERATIONAL_FREEZE/recovery_anchor.json', 'sha256': '63599bf5f50196b6c566adc7effac04235e5a808dbdfd9fe3fdd13f6abc57f31'}, 'package': {'path': 'artifacts/dayC/member1/colab/AFML_DayC_C10_Recovery_R2.zip', 'sha256': 'ab9dd0b87d046b5cde9a87d177c77b959069466d214fd95a31e08aa8ef4419cb'}, 'approval': {'path': 'artifacts/dayC/member1/approvals/C10_recovery_approval.json', 'sha256': '29b355b968750b01df6d31a845f9070f23828242776c0c37c467b2e891de36a9'}}
PARENT={
 'archive':{'path':'artifacts/dayC/member1/day10/'+PARENT_ID+'/returned_evidence/DAYC_C10_review.zip','sha256':'6c5e6aa061d772895ad006f160940d28784ff1440474f7cd88e2794715c0a074'},
 'lock':{'path':'artifacts/dayC/member1/day10/test_attempt.lock','sha256':'33b152e449c7a0d797b75c763d657b2d1f180e19fe6527a457fb3690690e3fc7'},
 'ledger':{'path':'artifacts/dayC/member1/day10/test_attempt_ledger.jsonl','sha256':'1d12976032b58787cde9ee6bd33a06ae9e9d7aae5b8c57e9b7a97c60fb59b1c1'},
 'policy':{'path':'configs/dayC/member1/frozen_precision_policy.json','sha256':'4951455481405602e835902da04478afa10eb9707935e6208aed6a813a0401bf'},
 'run':{'path':'configs/dayC/member1/frozen_test_run.json','sha256':'4ddc0696c91c2595b6592c3a4db5f3303f8899fbe6883bfcf07c6e244969d1e1'},
 'anchor':{'path':'artifacts/dayC/member1/day9/C9_FINAL_20261113T185009376370Z/APPROVED_FREEZE/freeze_anchor.json','sha256':'5a6c2b8b9ac141ef958ed2aeea4e16122a521fe05fc2df8165e3f63c80c689a5'},
 'package':{'path':'artifacts/dayC/member1/colab/AFML_DayC_C10_Frozen_Execution.zip','sha256':'1c230a84b1c860dee3b6b305eda49db219475da1317c924c6b612a61a3b84d2d'},
 'approval':{'path':'artifacts/dayC/member1/approvals/C10_approval.json','sha256':'8ba0147243ae82e6a9413a3b318976ce8bcc137aa348f3c9b4b6340c80a0c0f8'}}
SOURCES=('src/avfuse/member1_fusion/dayc_additional_recovery.py','scripts/dayC_member1/additional_recovery_day10.py','configs/dayC/member1/recovery/additional_authorization.schema.json','configs/dayC/member1/recovery/cache_path_resolution.json','src/avfuse/member1_fusion/dayc_progress.py')
RECOVERY_DIR='artifacts/dayC/member1/additional_recovery'
PACKAGE_PREFIX='AFML_DayC_C10_Recovery_R3'


def inside(root,item):
    path=ensure_inside_repo(root,item['path'],must_exist=True);require_sha256(path,item['sha256']);return path


def verify_parent(root):
    """Pinned production incident; no caller-selectable parent or force flag."""
    paths={k:inside(root,v) for k,v in PARENT.items()}
    with zipfile.ZipFile(paths['archive']) as z:
        names=z.namelist();require(len(names)==len(set(names)) and z.testzip() is None,'Invalid parent archive')
        inventory=json.loads(z.read('output_hashes.json'))
        require(set(names)==set(inventory)|{'output_hashes.json'},'Unlisted parent archive entries')
        for n,h in inventory.items():require(hashlib.sha256(z.read(n)).hexdigest()==h,'Parent archive content hash mismatch')
        require(paths['lock'].read_bytes()==z.read('ledger/test_attempt.lock') and paths['ledger'].read_bytes()==z.read('ledger/test_attempt_ledger.jsonl'),'Parent live ledger differs from original return')
        entries=[json.loads(l) for l in paths['ledger'].read_text().splitlines()]
        require(len(entries)==2 and [e['event'] for e in entries]==['TEST_ATTEMPT_OPENED','TEST_ATTEMPT_FAILED'],'Parent is not exactly one failed attempt')
        require({e['run_id'] for e in entries}=={PARENT_ID} and paths['lock'].read_text().strip()==PARENT_ID,'Wrong parent identity')
        failure=json.loads(z.read('run/attempt_failed.json'))
        require(failure['status']=='FAILED' and failure['mode_counts']==dict.fromkeys(MODES,0) and entries[1]['mode_counts']==dict.fromkeys(MODES,0),'Parent scored or did not fail')
        require(not any('/predictions/' in n or n.endswith('attempt_completed.json') for n in names),'Parent has outputs/completion')
        for key,field in [('policy','frozen_policy_sha256'),('run','frozen_run_sha256'),('anchor','freeze_anchor_sha256'),('approval','approved_c10_ack_sha256')]:
            require(entries[0][field]==PARENT[key]['sha256'],'Parent frozen/approval identity changed')
        require(paths['approval'].read_bytes()==z.read('approval/C10_ack.json'),'Parent approval differs')
        for n in names:
            if n.startswith('run/'):
                p=ensure_inside_repo(root,'artifacts/dayC/member1/day10/'+PARENT_ID+'/'+n[4:],must_exist=True)
                require(p.read_bytes()==z.read(n),'Saved parent evidence changed')
    expected={str(paths['lock']),str(paths['ledger'])}
    require(set(prior_test_artifacts(root))==expected,'Unknown prior test output/attempt')
    day10=Path(root)/'artifacts/dayC/member1/day10'
    require({p.parent.name for p in day10.glob('*/attempt_opened.json')}=={PARENT_ID},'Unknown parent attempt directory')
    verify_release(root,paths['policy'],paths['run'],paths['anchor'],paths['approval'],dry_run=True)
    frozen=read_strict_json(paths['run']);verify_registry(root,frozen['inputs'])
    verify_previous_recovery(root)
    return frozen


def verify_previous_recovery(root):
    paths={k:inside(root,item) for k,item in PREVIOUS.items()}
    with zipfile.ZipFile(paths['archive']) as z:
        names=z.namelist();require(len(names)==len(set(names)) and z.testzip() is None,'Invalid previous recovery archive')
        hashes=json.loads(z.read('output_hashes.json'));require(set(names)==set(hashes)|{'output_hashes.json'},'Unlisted previous recovery files')
        for name,h in hashes.items():require(hashlib.sha256(z.read(name)).hexdigest()==h,'Previous recovery archive tampered')
        for key,item in PARENT.items():require(z.read('parent/'+key+'/'+Path(item['path']).name)==inside(root,item).read_bytes(),'Original history linkage changed')
        require(z.read('recovery/recovery_attempt.lock')==paths['lock'].read_bytes() and z.read('recovery/recovery_attempt_ledger.jsonl')==paths['ledger'].read_bytes(),'Previous recovery live history changed')
        require(paths['lock'].read_text().strip()==PREVIOUS_RECOVERY_ID,'Wrong previous recovery lock')
        events=[json.loads(l) for l in paths['ledger'].read_text().splitlines()]
        require(len(events)==2 and [e['event'] for e in events]==['RECOVERY_ATTEMPT_OPENED','RECOVERY_ATTEMPT_FAILED'],'Previous recovery not one failed attempt')
        require(all(e['run_id']==PREVIOUS_RECOVERY_ID and e['parent_run_id']==PARENT_ID for e in events),'Previous recovery identity differs')
        failure=json.loads(z.read('run/failure.json'))
        require(events[1]['mode_counts']==failure['mode_counts']==dict.fromkeys(MODES,0) and events[1]['error']==failure['error']=='KeyboardInterrupt: ','Previous recovery scored or failure differs')
        require(not any('/predictions/' in n or n.endswith(('completion.json','saved_output_audit.json')) for n in names),'Previous recovery completed/scored')
        for key,name in [('context','frozen_recovery_context.json'),('anchor','recovery_anchor.json'),('approval','recovery_approval.json')]:require(paths[key].read_bytes()==z.read('recovery/'+name),'Previous operational identity changed')
        for name in names:
            if name.startswith('run/'):
                path=ensure_inside_repo(root,'artifacts/dayC/member1/recovery/'+PREVIOUS_RECOVERY_ID+'/'+name[4:],must_exist=True)
                require(path.read_bytes()==z.read(name),'Previous recovery run evidence changed')
    context=read_strict_json(paths['context']);anchor=read_strict_json(paths['anchor'])
    require(anchor['context']=={'path':PREVIOUS['context']['path'],'sha256':PREVIOUS['context']['sha256']},'Previous context anchor differs')
    proposal=inside(root,context['proposal']);c9=inside(root,context['amendment_c9_approval'])
    require_human_approval(c9,'C9',context_sha256=sha256_file(proposal))
    require(context['contract']==read_strict_json(proposal),'Previous C9 contract differs')
    old_digest=digest_object({'recovery_context_sha256':PREVIOUS['context']['sha256'],'recovery_anchor_sha256':PREVIOUS['anchor']['sha256']})
    require_human_approval(paths['approval'],'C10',context_sha256=old_digest)
    receipt=read_strict_json(paths['approval'])
    require(receipt['execution_package_sha256']==PREVIOUS['package']['sha256'] and events[0]['approval_sha256']==PREVIOUS['approval']['sha256'],'Previous C10 binding differs')
    require(events[0]['context_sha256']==old_digest and events[0]['package_sha256']==PREVIOUS['package']['sha256'] and events[0]['package_manifest_sha256']==receipt['execution_package_manifest_sha256'] and events[0]['parent_archive_sha256']==PARENT['archive']['sha256'],'Previous execution authorization bindings differ')
    previous_dir=Path(root)/'artifacts/dayC/member1/recovery'
    require(not list(previous_dir.glob('*/completion.json')) and not list(previous_dir.glob('*/predictions/*_test.jsonl')),'Unexpected previous recovery completion/output')
    require({p.parent.name for p in (Path(root)/'artifacts/dayC/member1/recovery').glob('*/failure.json')}=={PREVIOUS_RECOVERY_ID},'Unknown recovery history')


def verify_sources(root,sources):
    require(set(sources)==set(SOURCES),'Recovery source set differs')
    for name,digest in sources.items():require_sha256(ensure_inside_repo(root,name,must_exist=True),digest)


def scientific_contract(frozen):
    return {k:frozen[k] for k in ('checkpoint_sha256','mode_order','fixed_threshold','heldout_cohort','feature_path_policy','runtime','device','backend','batch_size','layers','precision','expected_count','known_test_label_summary_access')}


def path_resolution(root):
    value=read_strict_json(Path(root)/SOURCES[3])
    require(value=={'version': 'R2_PATH_RESOLUTION_ONLY', 'logical_root': '/content/afml_drive/MyDrive/AFML_Project/dimodif_emb', 'physical_root': '/content/afml_drive/.shortcut-targets-by-id/1yKf-S0Yyi8ZU7uLx6JbYKZpL95Kn3Mb7/dimodif_emb', 'scope': 'read_only_equivalent_cache_root', 'manifest_modified': False, 'original_scientific_C9_modified': False, 'additional_attempt_allowance_changed': False}, 'Unapproved physical cache-root resolution')
    return value


def effective_execution_contract(frozen,resolution):
    # Copy only operational lookup policy. Never rewrite the original frozen JSON.
    from copy import deepcopy
    require(frozen['feature_path_policy']['cache_root']==resolution['logical_root'],'Original logical root differs')
    effective=deepcopy(frozen)
    effective['feature_path_policy']['cache_root']=resolution['physical_root']
    return effective


def check_resolution_metadata(root):
    resolution=path_resolution(root)
    logical=Path(resolution['logical_root']);physical=Path(resolution['physical_root'])
    require(logical.resolve(strict=True)==physical.resolve(strict=True),'Drive alias retargeted; stop before claim')


def validate_proposal(root,proposal):
    require(proposal['schema_id']=='afml-c10-recovery-proposal/1' and proposal['status']=='PROPOSED_NOT_APPROVED','Not a recovery proposal')
    require(proposal['parent']==PARENT and proposal['parent_run_id']==PARENT_ID and proposal['recovery_run_id']==RECOVERY_ID,'Wrong recovery/parent identity')
    require(type(proposal['max_additional_attempts']) is int and proposal['max_additional_attempts']==1,'Only one additional attempt may be approved')
    require(proposal.get('previous_recovery')==PREVIOUS and proposal.get('prior_failed_attempts')==2 and proposal.get('overall_attempt_limit')==3,'Both failures and explicit three-total-attempt limit required')
    frozen=verify_parent(root);require(proposal['scientific_contract']==scientific_contract(frozen),'Scientific/cohort/runtime contract changed')
    require(proposal.get('cache_path_resolution')==path_resolution(root),'Missing/wrong path-only operational resolution')
    require(frozen['feature_path_policy']['cache_root']==path_resolution(root)['logical_root'],'Original logical root differs')
    verify_sources(root,proposal['operational_sources'])
    proof=read_strict_json(inside(root,proposal['test_proof']))
    require(proof['status']=='PASS' and proof['passed']>0 and proof['failed']==proof['errors']==proof['skipped']==0,'Fresh passing recovery tests required')
    require(proof['source_sha256']==proposal['operational_sources'],'Recovery source changed after tests')
    inside(root,proof['junit']);inside(root,proof['log'])
    return frozen


def propose(root,proof_path,out):
    frozen=verify_parent(root)
    item={'path':str(Path(proof_path).relative_to(root)),'sha256':sha256_file(proof_path)}
    result={'schema_id':'afml-c10-recovery-proposal/1','status':'PROPOSED_NOT_APPROVED','parent':PARENT,
        'parent_run_id':PARENT_ID,'recovery_run_id':RECOVERY_ID,'max_additional_attempts':1,'previous_recovery':PREVIOUS,'prior_failed_attempts':2,'overall_attempt_limit':3,
        'scientific_contract':scientific_contract(frozen),'cache_path_resolution':path_resolution(root),'operational_sources':{n:sha256_file(Path(root)/n) for n in SOURCES},
        'test_proof':item,'user_reported_cache_observation':{'expected_pairs':2000,'present_pairs':2000,'missing_features':0,'missing_metadata':0,'scope':'Existence only; historical cause and original per-file integrity not established'},
        'limits':['Original canonical metadata was accessed; zero scores generated.','Original failed attempt is never erased or counted as unused.','No historical per-test-file tensor hash inventory supplied.','One further attempt only after fresh approval acknowledging two failures; no automatic retry, fourth attempt or post-test tuning.','All original scientific limitations, owner releases and precision semantics remain binding.']}
    validate_proposal(root,result);write_new_json(out,result);return result


def finalize(root,proposal_path,approval,out):
    # Existing C9 approval phrase, explicitly bound to this operational amendment.
    require_human_approval(approval,'C9',context_sha256=sha256_file(proposal_path))
    proposal=read_strict_json(proposal_path);validate_proposal(root,proposal)
    require_new_file(out);Path(out).mkdir(parents=True,exist_ok=False)
    record={'schema_id':'afml-c10-recovery-frozen/1','status':'FROZEN_OPERATIONAL_AMENDMENT','proposal':{'path':str(Path(proposal_path).relative_to(root)),'sha256':sha256_file(proposal_path)},
        'amendment_c9_approval':{'path':str(Path(approval).relative_to(root)),'sha256':sha256_file(approval)},'contract':proposal}
    context=Path(out)/'frozen_recovery_context.json';digest=write_new_json(context,record)
    anchor=Path(out)/'recovery_anchor.json';write_new_json(anchor,{'context':{'path':str(context.relative_to(root)),'sha256':digest},'proposal_sha256':sha256_file(proposal_path),'approval_sha256':sha256_file(approval)})
    return context,anchor


def verify_amendment(root,context,anchor):
    record=read_strict_json(context);a=read_strict_json(anchor)
    require(a['context']=={'path':str(Path(context).relative_to(root)),'sha256':sha256_file(context)},'Recovery anchor/context differs')
    require(record['schema_id']=='afml-c10-recovery-frozen/1' and record['status']=='FROZEN_OPERATIONAL_AMENDMENT','Recovery not frozen')
    proposal_path=inside(root,record['proposal']);approval=inside(root,record['amendment_c9_approval'])
    require(sha256_file(proposal_path)==a['proposal_sha256'] and sha256_file(approval)==a['approval_sha256'],'Amendment anchor changed')
    require_human_approval(approval,'C9',context_sha256=sha256_file(proposal_path))
    require(record['contract']==read_strict_json(proposal_path),'Frozen amendment differs from approved proposal')
    frozen=validate_proposal(root,record['contract'])
    digest=digest_object({'recovery_context_sha256':sha256_file(context),'recovery_anchor_sha256':sha256_file(anchor)})
    return effective_execution_contract(frozen,record['contract']['cache_path_resolution']),digest


def verify_recovery_approval(root,approval,context_digest,archive,manifest_digest):
    record=read_strict_json(approval)
    # JSON schema is validated as well as semantic bindings. Parent constants are
    # source-pinned: test fixture receipts cannot validate against real parent bytes.
    from jsonschema import validate
    validate(record,read_strict_json(Path(root)/SOURCES[2]))
    require_human_approval(approval,'C10',context_sha256=context_digest)
    require(record['approval_kind']=='C10_INCIDENT_ADDITIONAL_RECOVERY' and record['parent_run_id']==PARENT_ID and record['recovery_run_id']==RECOVERY_ID,'Missing incident-specific C10 approval')
    require(record['parent_evidence_sha256']=={k:v['sha256'] for k,v in PARENT.items()},'Recovery approval parent hashes differ')
    require(record['previous_recovery_evidence_sha256']=={k:v['sha256'] for k,v in PREVIOUS.items()} and record['prior_failed_attempts_acknowledged']==2 and record['overall_attempt_limit']==3 and record['manual_interruption_acknowledged'] is True,'Fresh approval must acknowledge both failures and one further attempt')
    require(type(record['max_additional_attempts']) is int and record['max_additional_attempts']==1,'Additional-attempt allowance differs')
    require(record['execution_package_sha256']==sha256_file(archive) and record['execution_package_manifest_sha256']==manifest_digest,'Recovery package not approved')
    require(bool(record['acknowledgement_source'].strip()),'Missing actual human acknowledgement source')
    return record


def recovery_paths(root):
    base=Path(root)/RECOVERY_DIR
    return base/'recovery_attempt.lock',base/'recovery_attempt_ledger.jsonl',base/RECOVERY_ID


def claim_recovery(root,bindings):
    lock,ledger,out=recovery_paths(root);lock.parent.mkdir(parents=True,exist_ok=True)
    require_new_file(lock);require_new_file(ledger);require_new_file(out)
    with lock.open('x') as f:f.write(RECOVERY_ID+'\n');f.flush();os.fsync(f.fileno())
    with ledger.open('x') as f:f.flush();os.fsync(f.fileno())
    append_event(ledger,{'event':'RECOVERY_ATTEMPT_OPENED','run_id':RECOVERY_ID,'parent_run_id':PARENT_ID,**bindings})
    return ledger,out


def verify_package(root):
    from scripts.dayC_member1.colab_day10 import verify_package as verify
    return verify(Path(root))


def check_cache_root_metadata(policy):
    # Directory/ancestor metadata only; no test directory listing or file access.
    base=Path(policy['cache_root'])
    require(base.is_dir(),'Approved cache root unavailable; stop before claim')
    require(not any(p.is_symlink() for p in (base,*base.parents)),
        'Frozen cache resolver rejects this root alias; stop before claim')


def recovery_preflight(root,context,anchor,approval,archive):
    # No canonical row parsing, feature access or writes until actual C10 validates.
    manifest=verify_package(root);frozen,digest=verify_amendment(root,context,anchor)
    verify_recovery_approval(root,approval,digest,archive,manifest)
    from scripts.dayC_member1.run_day8 import observed_runtime
    from scripts.dayC_member1.colab_query_pilot import check_environment
    check_environment(observed_runtime(),frozen['runtime'])
    check_resolution_metadata(root)
    check_cache_root_metadata(frozen['feature_path_policy'])
    lock,ledger,out=recovery_paths(root)
    require(not any(p.exists() or p.is_symlink() for p in (lock,ledger,out)),'Further recovery already claimed; no fourth attempt')
    return frozen,{'context_sha256':digest,'package_sha256':sha256_file(archive),'package_manifest_sha256':manifest,'approval_sha256':sha256_file(approval),'parent_archive_sha256':PARENT['archive']['sha256']}


def _execute_recovery_with_progress(root,context,anchor,approval,archive,report):
    with report.phase('authorization_runtime_and_parent_verification'):
        frozen,bindings=recovery_preflight(root,context,anchor,approval,archive)
    ledger,out=claim_recovery(root,bindings);counts=dict.fromkeys(MODES,0)
    try:
        out.mkdir(exist_ok=False)
        report.attach(out/'progress.jsonl')
        write_new_json(out/'authorization_bindings.json',bindings)
        with (out/'recovery_approval.json').open('xb') as f:f.write(Path(approval).read_bytes())
        with report.phase('canonical_cohort_integrity'):
            rows=read_frozen_cohort(root,frozen)
        acquired={}
        with report.phase('cache_integrity_preflight',total_clips=len(rows)):
            for index,row in enumerate(rows,1):
                report.update(clip_index=index,verified_clips=index-1,operation='feature_and_metadata_hash_shape_finiteness_checks')
                if index==1 or index%25==0:report.emit('CACHE_CHECK_STARTED')
                path=resolve_test_feature(root,row,frozen['feature_path_policy']);acquired[row['clip_id']]=inspect_cache_identity(row,path)
                report.update(verified_clips=index)
                if index==1 or index%25==0 or index==len(rows):report.emit('CACHE_CHECK_COMPLETED')
        write_new_json(out/'cache_acquisition.json',{'status':'PASS','records':list(acquired.values()),'scope':'C10-time acquisition identity; no historical per-file tensor hash inventory'})
        with report.phase('model_initialization_and_frozen_state_verification'):
            from scripts.dayB_inference import create_adapter,load_features
            from avfuse.member1_fusion.dayc_precision_audit import check_model_invariants
            from types import SimpleNamespace
            import torch
            registry=frozen['inputs'];upstream=(Path(root)/registry['model_source']['path']).parent.parent
            torch.manual_seed(42);adapter=create_adapter(SimpleNamespace(upstream=str(upstream),device=frozen['device']))
            require(adapter.checkpoint_path.resolve()==(Path(root)/registry['checkpoint']['path']).resolve(),'Checkpoint path differs')
            require(adapter.config_path.resolve()==(Path(root)/registry['model_config']['path']).resolve(),'Model config differs')
            before=check_model_invariants(adapter,checkpoint_sha256=frozen['checkpoint_sha256'])
        def resolve(row):
            path=resolve_test_feature(root,row,frozen['feature_path_policy']);identity=acquired[row['clip_id']]
            require_sha256(path,identity['feature_sha256']);require_sha256(path.parent/'meta.json',identity['metadata_sha256']);return path
        def progress(index,completed):
            counts.update(completed);report.update(clip_index=index,mode_counts=dict(completed),operation='four_mode_scoring')
            if all(value==index for value in completed.values()) and (index==1 or index%25==0 or index==len(rows)):
                report.emit('SCORING_PROGRESS')
        with report.phase('four_mode_scoring'):
            outputs,identities=score_complete_cohort(rows,adapter,load_features,resolve,RECOVERY_ID,progress)
        report.update(phase='post_scoring_integrity_and_publication',operation='verify_original_model_and_cache_hashes')
        report.emit('PHASE_STARTED')
        after=check_model_invariants(adapter,checkpoint_sha256=frozen['checkpoint_sha256']);require(before['state_dict_sha256']==after['state_dict_sha256'],'Frozen model changed')
        for row in rows:resolve(row)
        verify_parent(root)
        hashes=publish_complete_predictions(out,outputs,ids={r['clip_id'] for r in rows},checkpoint=frozen['checkpoint_sha256'],run_id=RECOVERY_ID)
        write_new_json(out/'completion.json',{'status':'COMPLETE','run_id':RECOVERY_ID,'mode_counts':counts,'prediction_sha256':hashes,'state_dict_before_sha256':before['state_dict_sha256'],'state_dict_after_sha256':after['state_dict_sha256']})
        append_event(ledger,{'event':'RECOVERY_ATTEMPT_COMPLETED','run_id':RECOVERY_ID,'parent_run_id':PARENT_ID,'mode_counts':counts,'prediction_sha256':hashes})
        inventory={str(p.relative_to(out)):sha256_file(p) for p in out.rglob('*') if p.is_file() and p.name!='progress.jsonl'}
        write_new_json(out/'run_manifest.json',{'files':inventory})
        return out
    except BaseException as exc:
        failure={'event':'RECOVERY_ATTEMPT_FAILED','run_id':RECOVERY_ID,'parent_run_id':PARENT_ID,'mode_counts':counts,'error':f'{type(exc).__name__}: {exc}','automatic_retry_permitted':False}
        if out.exists():write_new_json(out/'failure.json',failure)
        append_event(ledger,failure);raise


def execute_recovery(root,context,anchor,approval,archive):
    with ProgressReporter(output=lambda message:print(message,flush=True)) as report:
        out=_execute_recovery_with_progress(root,context,anchor,approval,archive,report)
    write_new_json(out/'progress_identity.json',{'sha256':sha256_file(out/'progress.jsonl'),'scope':'Final progress log after heartbeat thread shutdown; not benchmark timing'})
    return out


def audit_recovery(root,context,anchor,approval,archive):
    manifest=verify_package(root);frozen,digest=verify_amendment(root,context,anchor)
    verify_recovery_approval(root,approval,digest,archive,manifest)
    lock,ledger,out=recovery_paths(root)
    require(lock.read_text().strip()==RECOVERY_ID,'Recovery lock corrupted')
    events=[json.loads(l) for l in ledger.read_text().splitlines()]
    require(len(events)==2 and [e['event'] for e in events]==['RECOVERY_ATTEMPT_OPENED','RECOVERY_ATTEMPT_COMPLETED'],'Recovery not exactly one successful attempt')
    require(all(e['run_id']==RECOVERY_ID and e['parent_run_id']==PARENT_ID for e in events),'Recovery ledger identity mismatch')
    require(events[0]['context_sha256']==digest and events[0]['approval_sha256']==sha256_file(approval) and events[0]['package_sha256']==sha256_file(archive) and events[0]['package_manifest_sha256']==manifest,'Recovery ledger authorization mismatch')
    require(not (out/'failure.json').exists(),'Failed recovery is not success')
    complete=read_strict_json(out/'completion.json');require(complete['status']=='COMPLETE' and complete['mode_counts']==events[1]['mode_counts']==dict.fromkeys(MODES,frozen['expected_count']),'Recovery counts differ')
    require(complete['state_dict_before_sha256']==complete['state_dict_after_sha256'],'Recovered model state changed')
    for n,h in read_strict_json(out/'run_manifest.json')['files'].items():require_sha256(ensure_inside_repo(out,n,must_exist=True),h)
    require_sha256(out/'progress.jsonl',read_strict_json(out/'progress_identity.json')['sha256'])
    rows=read_frozen_cohort(root,frozen);ids={r['clip_id'] for r in rows}
    acquired=read_strict_json(out/'cache_acquisition.json')['records'];require(len(acquired)==len(ids) and {r['clip_id'] for r in acquired}==ids,'Recovery cache coverage differs')
    by_id={r['clip_id']:r for r in acquired}
    for row in rows:
        path=resolve_test_feature(root,row,frozen['feature_path_policy']);require_sha256(path,by_id[row['clip_id']]['feature_sha256']);require_sha256(path.parent/'meta.json',by_id[row['clip_id']]['metadata_sha256'])
    from avfuse.member2_data.predictions import read_predictions
    require({p.name for p in (out/'predictions').iterdir()}=={m+'_test.jsonl' for m in MODES},'Wrong recovery prediction file set')
    hashes={}
    for mode in MODES:
        path=out/'predictions'/(mode+'_test.jsonl');validate_saved_rows(read_predictions(path),mode=mode,ids=ids,checkpoint=frozen['checkpoint_sha256'],run_id=RECOVERY_ID)
        hashes[mode]=sha256_file(path);require(hashes[mode]==complete['prediction_sha256'][mode]==events[1]['prediction_sha256'][mode],'Recovery output hash differs')
    return {'previous_recovery_failed_attempts':1,'total_attempts':3,'status':'READY_FOR_MEMBER2_SCORING','run_id':RECOVERY_ID,'count_per_mode':len(ids),'prediction_sha256':hashes,'parent_run_id':PARENT_ID,'parent_failed_attempts':1,'recovery_attempts':1,'test_metrics_computed':False,'historical_metadata_access':True}


def build_package(root,context,anchor,out):
    frozen,digest=verify_amendment(root,context,anchor)
    require(not any(p.exists() for p in recovery_paths(root)[:2]),'Recovery already claimed')
    require_new_file(out)
    for suffix in ('_prepare.py','_run.py'):require_new_file(Path(out).with_name(Path(out).stem+suffix))
    payload={}
    with zipfile.ZipFile(inside(root,PARENT['package'])) as z:
        prefix='AFML_DayC_Day10_Frozen/'
        for name in z.namelist():
            require(name.startswith(prefix) and '..' not in Path(name).parts,'Invalid original package name')
            relative=name[len(prefix):]
            if relative not in {'PACKAGE_MANIFEST.json','PACKAGE_MANIFEST.sha256'}:payload[relative]=z.read(name)
    # Preserve the exact original package as incident evidence, not an executable substitute.
    for item in PARENT.values():payload[item['path']]=inside(root,item).read_bytes()
    with zipfile.ZipFile(inside(root,PARENT['archive'])) as z:
        for name in z.namelist():
            if name.startswith('run/'):payload['artifacts/dayC/member1/day10/'+PARENT_ID+'/'+name[4:]]=z.read(name)
    record=read_strict_json(context);proposal=record['contract'];proof_path=inside(root,proposal['test_proof']);proof=read_strict_json(proof_path)
    for item in PREVIOUS.values():payload[item['path']]=inside(root,item).read_bytes()
    previous_context=read_strict_json(inside(root,PREVIOUS['context']))
    for item in [previous_context['proposal'],previous_context['amendment_c9_approval']]:payload[item['path']]=inside(root,item).read_bytes()
    with zipfile.ZipFile(inside(root,PREVIOUS['archive'])) as prior:
        for name in prior.namelist():
            if name.startswith('run/'):payload['artifacts/dayC/member1/recovery/'+PREVIOUS_RECOVERY_ID+'/'+name[4:]]=prior.read(name)
    files=[Path(context),Path(anchor),inside(root,record['proposal']),inside(root,record['amendment_c9_approval']),proof_path,
           inside(root,proof['junit']),inside(root,proof['log']),*(Path(root)/n for n in SOURCES)]
    for path in files:payload[str(path.relative_to(root))]=path.read_bytes()
    payload['RECOVERY_LAUNCH.json']=(json.dumps({'context':str(Path(context).relative_to(root)),'anchor':str(Path(anchor).relative_to(root))},sort_keys=True)+'\n').encode()
    manifest={'schema_id':'avfuse-day10-frozen-package/1.0','recovery_package':True,'files':{n:hashlib.sha256(b).hexdigest() for n,b in sorted(payload.items())}}
    raw=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode();manifest_sha=hashlib.sha256(raw).hexdigest()
    payload['PACKAGE_MANIFEST.json']=raw;payload['PACKAGE_MANIFEST.sha256']=(manifest_sha+'\n').encode()
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for n,b in sorted(payload.items()):z.writestr(PACKAGE_PREFIX+'/'+n,b)
    with zipfile.ZipFile(out) as z:
        require(z.testzip() is None,'Recovery package CRC failed')
        for n,b in payload.items():require(z.read(PACKAGE_PREFIX+'/'+n)==b,'Recovery package bytes changed')
    package_sha=sha256_file(out)
    for action in ('prepare','run'):
        script=colab_cell(Path(out).name,package_sha,action)
        with Path(out).with_name(Path(out).stem+'_'+action+'.py').open('x') as f:f.write(script)
    return {'archive':str(out),'execution_package_sha256':package_sha,'execution_package_manifest_sha256':manifest_sha,'context_sha256':digest,
        'parent_run_id':PARENT_ID,'recovery_run_id':RECOVERY_ID,'parent_evidence_sha256':{k:v['sha256'] for k,v in PARENT.items()},'max_additional_attempts':1,'previous_recovery_evidence_sha256':{k:v['sha256'] for k,v in PREVIOUS.items()},'prior_failed_attempts_acknowledged':2,'overall_attempt_limit':3,'manual_interruption_acknowledged':True}


def colab_cell(filename,digest,action):
    return f"""# User-managed recovery; no mounts/installs, original runtime must be preserved.
from pathlib import Path
import hashlib, os, stat, subprocess, sys, zipfile
from google.colab import files
archive=Path('/content')/{filename!r}
root=Path('/content')/{PACKAGE_PREFIX!r}
assert hashlib.sha256(archive.read_bytes()).hexdigest()=={digest!r}
if {action!r}=='prepare':
    assert not root.exists(), 'Preserve existing recovery workspace; never reset it'
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        assert len(names)==len(set(names)) and z.testzip() is None
        assert all(not Path(n).is_absolute() and '..' not in Path(n).parts and Path(n).parts[0]==root.name for n in names)
        assert all(not stat.S_ISLNK(i.external_attr>>16) for i in z.infolist())
        z.extractall('/content')
else:
    assert root.is_dir()
cmd=[sys.executable,'-B',str(root/'scripts/dayC_member1/additional_recovery_day10.py'),'--action',{action!r},'--repo-root',str(root)]
if {action!r}=='run':cmd+=['--approval-file','/content/C10_recovery_approval.json','--package-archive',str(archive)]
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(root/'src')+os.pathsep+str(root),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
env['PYTHONUNBUFFERED']='1'
lines=[]
with subprocess.Popen(cmd,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1) as process:
    for line in process.stdout:
        print(line,end='',flush=True);lines.append(line)
    code=process.wait()
print('Exit code:',code)
review=root/'artifacts/dayC/member1/recovery_reviews'/{RECOVERY_ID!r}/'DAYC_C10_RECOVERY_review.zip'
markers=[line[len('RETURN REVIEW ZIP: '):].strip() for line in ''.join(lines).splitlines() if line.startswith('RETURN REVIEW ZIP: ')]
if len(markers)==1 and Path(markers[0]).resolve()==review.resolve() and review.is_file():
    print('REVIEW_SHA256:',hashlib.sha256(review.read_bytes()).hexdigest());files.download(str(review))
else:
    print('No new review archive emitted by this invocation. Existing archives will not be downloaded.')
print('STOP. Prepare does not authorize inference; run never authorizes a fourth attempt.')
"""


def prepare_remote(root,context,anchor):
    verify_package(root)
    # Exact original checkpoint staging only; no cached tensor reads.
    from scripts.dayC_member1.colab_day10 import stage_checkpoint
    stage_checkpoint(Path(root),read_strict_json(Path(root)/'DAY10_LAUNCH.json'))
    frozen,digest=verify_amendment(root,context,anchor)
    from scripts.dayC_member1.run_day8 import observed_runtime
    from scripts.dayC_member1.colab_query_pilot import check_environment
    check_environment(observed_runtime(),frozen['runtime'])
    check_resolution_metadata(root)
    check_cache_root_metadata(frozen['feature_path_policy'])
    require(not any(p.exists() for p in recovery_paths(root)[:2]),'Recovery already claimed')
    return {'status':'RECOVERY_METADATA_PREFLIGHT_PASS','context_sha256':digest,'package_manifest_sha256':sha256_file(Path(root)/'PACKAGE_MANIFEST.json'),'test_data_accessed':False,'attempt_claimed':False}


def package_handoff(root,out,audit,frozen):
    from scripts.dayC_member1.package_day10_handoff import build_public_payload,add_accepted_query_validation
    payload=build_public_payload(out,audit,frozen);add_accepted_query_validation(root,payload,frozen)
    payload['incident_recovery_audit.json']=(json.dumps(audit,sort_keys=True,indent=2)+'\n').encode()
    for name,key in [('parent_failed_ledger.jsonl','ledger'),('parent_failed_lock.txt','lock')]:payload['history/'+name]=inside(root,PARENT[key]).read_bytes()
    payload['history/previous_recovery_ledger.jsonl']=inside(root,PREVIOUS['ledger']).read_bytes();payload['history/previous_recovery_lock.txt']=inside(root,PREVIOUS['lock']).read_bytes()
    lock,ledger,_=recovery_paths(root);payload['history/recovery_ledger.jsonl']=ledger.read_bytes();payload['history/recovery_lock.txt']=lock.read_bytes()
    hashes={n:hashlib.sha256(b).hexdigest() for n,b in payload.items()};payload['output_hashes.json']=(json.dumps(hashes,sort_keys=True,indent=2)+'\n').encode()
    dest=Path(out)/'member1_recovery_handoff.zip';require_new_file(dest)
    with zipfile.ZipFile(dest,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for n,b in payload.items():z.writestr(n,b)
    return dest


def return_review(root,context,anchor,approval,error=None):
    destination=Path(root)/'artifacts/dayC/member1/recovery_reviews'/RECOVERY_ID
    destination.mkdir(parents=True,exist_ok=False);out=destination/'DAYC_C10_RECOVERY_review.zip';payload={}
    for key,item in PREVIOUS.items():payload['previous_recovery/'+key+'/'+Path(item['path']).name]=inside(root,item).read_bytes()
    for key,item in PARENT.items():payload['parent/'+key+'/'+Path(item['path']).name]=inside(root,item).read_bytes()
    for name,path in [('frozen_recovery_context.json',context),('recovery_anchor.json',anchor),('recovery_approval.json',approval)]:
        if path is not None and Path(path).is_file():payload['recovery/'+name]=Path(path).read_bytes()
    lock,ledger,run=recovery_paths(root)
    for path in (lock,ledger):
        if path.is_file():payload['recovery/'+path.name]=path.read_bytes()
    if run.is_dir():
        for path in run.rglob('*'):
            if path.is_file():
                require(not path.is_symlink() and path.suffix not in {'.npz','.pth','.pt'},'Unexpected recovery output')
                payload['run/'+str(path.relative_to(run))]=path.read_bytes()
    payload['review_status.json']=(json.dumps({'status':'FAILED' if error else 'COMPLETE','error':error,'parent_run_id':PARENT_ID,'recovery_run_id':RECOVERY_ID,'no_automatic_retry':True},sort_keys=True)+'\n').encode()
    hashes={n:hashlib.sha256(b).hexdigest() for n,b in payload.items()};payload['output_hashes.json']=(json.dumps(hashes,sort_keys=True)+'\n').encode()
    with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for n,b in payload.items():z.writestr(n,b)
    return out
