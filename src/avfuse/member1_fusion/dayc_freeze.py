"""Member 1 proposal/freeze identity guards. No test-row or feature access."""
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from avfuse.member1_fusion.dayc_guards import (
    ensure_inside_repo,read_strict_json,require_human_approval,require_new_file,
    require_sha256,sha256_file,write_new_json)
from avfuse.member1_fusion.dayc_validation import MODES
from avfuse.member1_fusion.dayc_test_contract import (canonical_release,cohort_contract,validate_path_policy,OFFICIAL_M2_SHA)

CHECKPOINT='e1e59006d1bb6ad36a3859af5486a6649ed41bba23258b7c41e24f873be7948e'
PURPOSE='one frozen full four-mode held-out DiMoDif inference for original Day 10'
PREPARATION_TEXT="no I need you to speedrun through B3 and we'll test B2 and B3 together for any auditing. Implement B3 part as a whole block and after that we'll review and audit"
SCORING_FILES=(
    'scripts/dayC_member1/build_day10_package.py','scripts/dayC_member1/colab_day10.py',
    'src/avfuse/member1_fusion/dayc_test_contract.py',
    'src/avfuse/member1_fusion/precision.py','src/avfuse/member1_fusion/attention.py',
    'src/avfuse/member1_fusion/dimodif_intervention.py','src/avfuse/member1_fusion/dimodif_adapter.py',
    'src/avfuse/member1_fusion/dayc_precision_audit.py','src/avfuse/member1_fusion/dayc_guards.py',
    'src/avfuse/member1_fusion/dayc_freeze.py','src/avfuse/member1_fusion/dayc_heldout.py',
    'src/avfuse/member1_fusion/model_adapter.py','src/avfuse/shared/contracts.py',
    'src/avfuse/member2_data/cache.py','src/avfuse/member1_fusion/dayc_validation.py',
    'src/avfuse/member1_fusion/dayc_cache_mapping.py','scripts/dayC_member1/run_day8.py',
    'scripts/dayC_member1/colab_query_pilot.py',
    'scripts/dayB_inference.py','scripts/dayC_member1/freeze_day9.py',
    'scripts/dayC_member1/preflight.py','src/avfuse/member1_fusion/dayc_research.py',
    'src/avfuse/member2_data/evaluate.py','src/avfuse/member2_data/paired.py','src/avfuse/member2_data/seal.py',
    'scripts/dayC_member1/run_day10_test.py',
    'scripts/dayC_member1/verify_day10_release.py','scripts/dayC_member1/audit_day10.py',
    'scripts/dayC_member1/package_day10_handoff.py',
    'src/avfuse/member2_data/manifest.py','src/avfuse/member2_data/predictions.py',
    'src/avfuse/member2_data/splits.py','src/avfuse/shared/jsonl.py',
    'contracts/schemas/manifest.schema.json','contracts/schemas/prediction.schema.json')
BASE_REQUIRED=('checkpoint','model_config','model_source','canonical_manifest','validation_manifest',
    'ready_val_manifest','calibration','precision_modes','ready_test_manifest','b2_research_packet',
    'metrics','paired','per_type','benchmark_short','benchmark_typical','benchmark_long',
    'member2_release','member3_release','source_test_report','test_junit','test_log',*('predictions_'+m for m in MODES))


def require(ok,message):
    if not ok:raise ValueError(message)


def digest_object(obj):
    return hashlib.sha256((json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()).hexdigest()


def finite_tree(value):
    if isinstance(value,dict):
        for child in value.values():finite_tree(child)
    elif isinstance(value,(list,tuple)):
        for child in value:finite_tree(child)
    elif isinstance(value,float):require(math.isfinite(value),'Nonfinite freeze evidence')
    elif isinstance(value,str):
        require(not re.search(r'<[^>]+>',value),'Specification placeholder is not real evidence')


def check_preparation_authorization(record,context_sha256):
    require(record.get('gate')=='B3_SOURCE_PREPARATION' and record.get('user_supplied') is True
        and record.get('instruction_verbatim')==PREPARATION_TEXT
        and record.get('C8A_context_sha256')==context_sha256
        and record.get('permits_source_preparation_only') is True
        and record.get('does_not_authorize_freeze_or_test') is True,'Missing actual B3 source-preparation instruction')


def approved_runtime(context):
    release=context['pilot_release'];environment=deepcopy(release['environment'])
    for name,change in release['runtime_amendment']['allowed_version_changes'].items():
        environment['versions'][name]=change['to']
    return environment


def validate_owner_release(record,owner,expected):
    evidence=deepcopy(record)
    if owner=='member2' and canonical_release(record) and 'test_cache' in record:
        # This is the owner's explicit filename convention, not a missing asset
        # placeholder. Permit only that exact template, without changing raw bytes.
        require(record['test_cache']['path']=='Drive avfuse/dimodif_emb/test/<clip_id>/features.npz + meta.json',
            'Unexpected owner cache template')
        evidence['test_cache']['path']='Drive avfuse/dimodif_emb/test/{clip_id}/features.npz + meta.json'
    finite_tree(evidence)
    require(record.get('owner')==owner and record.get('external_owner_acknowledgement') is True,
            'Missing external '+owner+' acknowledgement')
    for key in ('release_id','acknowledgement_source','approved_at'):
        require(isinstance(record.get(key),str) and bool(record[key].strip()),'Missing owner release '+key)
    when=datetime.fromisoformat(record['approved_at'].replace('Z','+00:00'))
    require(when.utcoffset() is not None and when<=datetime.now(timezone.utc),'Invalid release timestamp')
    require(record.get('four_modes')==list(MODES),'Owner release must retain all four modes')
    require(record.get('checkpoint_sha256')==CHECKPOINT,'Wrong owner checkpoint')
    if owner=='member2':
        require(record.get('canonical_manifest_sha256')==expected['canonical_manifest_sha256'],'Member2 manifest differs')
        require(record.get('protocol_sha256')==expected['calibration_sha256']
            and record.get('fixed_threshold')==expected['fixed_threshold'],'Final threshold/calibration differs')
        require(record.get('threshold_selected_on')=='val' and type(record.get('n_boot')) is int and record.get('n_boot')==1000 and type(record.get('seed')) is int and record.get('seed')==0,
                'Member2 threshold/bootstrap protocol differs')
        require(record.get('score_orientation') in {'probability_of_fake','probability_of_fake (checked: ok for all val configs)'} and record.get('sealed_test') is True,'Missing test seal/orientation')
        require((type(record.get('test_model_predictions_used_for_policy_selection')) is int and record['test_model_predictions_used_for_policy_selection']==0
            or record.get('test_model_predictions_used_for_policy_selection') is False)
            and record.get('known_test_label_summary_access_acknowledged') is True,'Missing test-exposure disclosure')
        require(record.get('stratification_scope') in {'analysis_subsets','analysis_subsets: per fake type, that type + all real; whole groups resampled with replacement; category counts vary per replicate'} and bool(record.get('source_group_definition')),
                'Member2 must explicitly acknowledge the actual cluster/bootstrap method')
        identity=record.get('ready_test_manifest',{})
        require(isinstance(identity.get('path'),str) and re.fullmatch('[0-9a-f]{64}',identity.get('sha256','')),
                'Missing opaque ready-test manifest identity')
        require(type(identity.get('expected_count')) is int and identity['expected_count']>0,'Missing full held-out cohort count')
        require(bool(record.get('one_time_evaluation_purpose')) and bool(record.get('limitations')),'Incomplete Member2 release')
        cohort_contract(record)
    elif owner=='member3':
        require(record.get('runtime')==expected['runtime'],'Member3 runtime differs from approved arrangement')
        for key,value in {'device':'cpu','backend':'sdpa','batch_size':1,'warmup':10,'repeats':50,
                          'synchronization':'not_required_cpu','cached_model_scope':'cached_feature_model',
                          'peak_memory_method':'not_measured','conversion_overhead_included':True,
                          'raw_samples_retained':True,'preprocessing_reported_separately':True}.items():
            require(record.get(key)==value,'Member3 benchmark contract differs: '+key)
        require(all(type(record.get(k)) is int for k in ('batch_size','warmup','repeats')),'Benchmark counts must be integers')
        require(record.get('approved_workloads')==expected['approved_workloads'],'Member3 workloads changed')
        require(set(record.get('benchmark_evidence_sha256',[]))==set(expected['benchmark_evidence_sha256']),
                'Member3 release does not acknowledge the actual four-mode timings')
        require(bool(record.get('isolated_operator_scope')) and record['isolated_operator_scope']!='cached_feature_model','Operator/model timing scopes not separated')
        require(record.get('peak_memory_bytes') is None,'CPU peak memory must remain unmeasured/null')
    else:raise ValueError('Unknown owner')
    return record


def prior_test_artifacts(root):
    root=Path(root);day10=root/'artifacts/dayC/member1/day10'
    files=list((root/'artifacts/predictions').glob('*_test.jsonl'))
    if day10.exists():files+=list(day10.glob('*/predictions/*_test.jsonl'))
    files += [p for p in (day10/'test_attempt.lock',day10/'test_attempt_ledger.jsonl') if os.path.lexists(p)]
    return sorted(str(p) for p in files)


def verify_registry(root,registry,*,complete=True):
    require(isinstance(registry,dict),'Expected hash registry')
    if complete:
        missing=set(BASE_REQUIRED)|{'source:'+p for p in SCORING_FILES}
        require(not (missing-set(registry)),'Missing freeze dependencies: '+', '.join(sorted(missing-set(registry))))
    for name,item in registry.items():
        require(set(item)=={'path','sha256'},'Registry entries require exact path/hash')
        if name.startswith('source:'):
            require(item['path']==name.removeprefix('source:'),'Source registry redirected to another file')
        path=ensure_inside_repo(root,Path(item['path']),must_exist=True)
        # Opaque byte hashing only; never parse ready-test rows here.
        require_sha256(path,item['sha256'])
    if complete:
        require(registry['checkpoint']['sha256']==CHECKPOINT,'Wrong checkpoint identity')
    return registry


def expected_owner_context(root,context,registry):
    calibration=read_strict_json(ensure_inside_repo(root,context['experiment_spec']['local_inputs']['calibration']['path'],must_exist=True))
    item=context['experiment_spec']['local_inputs']['dayb_context']
    g5path=ensure_inside_repo(root,item['path'],must_exist=True);require_sha256(g5path,item['sha256'])
    g5=read_strict_json(g5path)
    return {'canonical_manifest_sha256':context['pilot_release']['canonical_manifest_sha256'],
        'calibration_sha256':context['experiment_spec']['local_inputs']['calibration']['sha256'],
        'fixed_threshold':calibration['threshold'],'runtime':approved_runtime(context),
        'approved_workloads':g5['approved_workloads'],
        'benchmark_evidence_sha256':[registry[k]['sha256'] for k in ('benchmark_short','benchmark_typical','benchmark_long') if k in registry]}


def build_draft(*,context,registry,evidence_packet=None,member2_release=None,member3_release=None,
                expected=None,c8b_accepted=False,interpretation=None,feature_path_policy=None):
    """Pure preparation: missing evidence stays null with explicit blockers, never fabricated."""
    finite_tree(registry);blockers=[];table=None;findings=None
    if not c8b_accepted:blockers.append('C8B scientific acceptance not yet supplied; combined B2/B3 source preparation only')
    if evidence_packet is None:blockers.append('Actual B2 four-mode statistics/research packet not returned')
    else:
        from avfuse.member1_fusion.dayc_research import validate_day9_decision
        findings=validate_day9_decision(evidence_packet);table=evidence_packet['decision_table']
    if member2_release is None:blockers.append('WAITING FOR MEMBER 2 final threshold/seal/evaluation acknowledgement')
    elif expected is not None:validate_owner_release(member2_release,'member2',expected)
    if member3_release is None:blockers.append('WAITING FOR MEMBER 3 benchmark/runtime acknowledgement')
    elif expected is not None:validate_owner_release(member3_release,'member3',expected)
    required=set(BASE_REQUIRED)|{'source:'+p for p in SCORING_FILES}
    if required-set(registry):blockers.append('Freeze identity registry incomplete: '+', '.join(sorted(required-set(registry))))
    if interpretation is None:blockers.append('Lead interpretation not proposed/approved; no policy selected')
    if member2_release and canonical_release(member2_release):
        if feature_path_policy is None:blockers.append('Held-out cache/runtime path policy must be proposed before C9')
        else:
            validate_path_policy(feature_path_policy)
            require(feature_path_policy['kind']=='split_clip_cache','Canonical cohort requires explicit original cache resolution')
        if registry.get('ready_test_manifest')!=registry.get('canonical_manifest'):
            blockers.append('Canonical test selector must bind the same opaque canonical manifest entry')
    elif feature_path_policy is not None:validate_path_policy(feature_path_policy)
    from avfuse.member1_fusion.dayc_research import LIMITATIONS
    limitations=LIMITATIONS+[
        'Trailing-silence shortcut warning: validation AUROC about 0.682; previously inspected test summary about 0.660; matched-control result outstanding.',
        'No test-model predictions or labels may be used to select/tune the precision policy.',
        'Lead must explicitly approve proceeding with historical test-label summary exposure disclosed.']
    proposal={'schema_id':'avfuse-member1-precision-policy/1.0','status':'PROPOSED_NOT_APPROVED',
        'preparation_state':'BLOCKED_PENDING_DEPENDENCIES' if blockers else 'READY_FOR_C9_REVIEW',
        'decision_split':'val','test_score_accesses_for_policy_selection':0,
        'test_access_basis':'required owner/lead attestation; no test access is executed by proposal code',
        'checkpoint_sha256':CHECKPOINT,'selected_interpretation':interpretation,
        'H2':findings['H2'] if findings else None,'H3':findings['H3'] if findings else None,
        'numeric_damage_and_ci':{'D':table['damage'],'CI':table['damage_ci']} if table else None,
        'restoration_direct_contrast_and_ci':{'H3_direct':table['H3_direct'],'CI':table['H3_ci']} if table else None,
        'decision_table':table,'measured_costs':evidence_packet['evidence']['costs'] if evidence_packet else None,
        'fixed_threshold':expected['fixed_threshold'] if expected and member2_release else None,
        'four_frozen_modes':list(MODES),'limitations':limitations,'evidence_hashes':deepcopy(registry),
        'member2_release_id':member2_release['release_id'] if member2_release else None,
        'member3_release_id':member3_release['release_id'] if member3_release else None,
        'requires_approvals':['member1_lead','member2_evaluation','member3_runtime'],
        'blockers':blockers,'ready_for_freeze':not blockers,'heldout_inference_authorized':False}
    if member2_release:
        proposal['heldout_cohort']=cohort_contract(member2_release)
        proposal['owner_scientific_corrections']=deepcopy(member2_release['limitations'])
        if canonical_release(member2_release):
            proposal['limitations']=[v for v in limitations if v not in {LIMITATIONS[1],LIMITATIONS[2]} and not v.startswith('Trailing-silence shortcut warning:')]
            proposal['limitations']+=['Member2 reports zero source-level LAV-DF train overlap; speaker separation remains unverified.',
                'Member2 reports validation silence control without detector drop for the three earlier configurations; duration uncontrolled; no query-specific control established.',
                'No pre-existing per-test-file tensor SHA inventory supplied; complete C10-time checks establish acquisition identity, not independent original-byte equivalence.']
    proposal['feature_path_policy']=deepcopy(feature_path_policy or {'kind':'project_local_only'})
    finite_tree(proposal);return proposal


def validate_complete_proposal(root,proposal,context):
    finite_tree(proposal)
    require(proposal.get('schema_id')=='avfuse-member1-precision-policy/1.0','Wrong policy schema')
    require(proposal.get('status')=='PROPOSED_NOT_APPROVED' and proposal.get('ready_for_freeze') is True
        and not proposal.get('blockers'),'Proposal dependencies unresolved')
    require(proposal['decision_split']=='val' and proposal['checkpoint_sha256']==CHECKPOINT,'Non-validation/wrong checkpoint proposal')
    require(proposal['four_frozen_modes']==list(MODES),'Missing fourth mode')
    require(proposal.get('selected_interpretation') in {'reference_preferred','no_restoration_justified',
        'restoration_conditionally_supported','inconclusive'},'Missing/invalid lead-reviewed interpretation')
    from avfuse.member1_fusion.dayc_research import LIMITATIONS
    require(set(LIMITATIONS)-{LIMITATIONS[1],LIMITATIONS[2]}<=set(proposal.get('limitations',[])),'Mandatory limitations omitted')
    validate_path_policy(proposal.get('feature_path_policy',{'kind':'project_local_only'}))
    require(type(proposal.get('test_score_accesses_for_policy_selection')) is int and proposal.get('test_score_accesses_for_policy_selection')==0,'Test-based policy selection forbidden')
    registry=verify_registry(root,proposal['evidence_hashes'])
    for path,expected_sha in context['pilot_source_sha256'].items():
        key='source:'+path
        if key in registry:require(registry[key]['sha256']==expected_sha,'Approved scoring/source semantics changed: '+path)
    for name in ('checkpoint','model_config','upstream_source','canonical_manifest','validation_manifest',
                 'ready_val_manifest','calibration','precision_modes',*('predictions_'+m for m in MODES[:3])):
        target='model_source' if name=='upstream_source' else name
        require(registry[target]['sha256']==context['experiment_spec']['local_inputs'][name]['sha256'],
                'Approved original identity changed: '+name)
    from avfuse.member1_fusion.dayc_precision_audit import expected_mode_boundaries
    definitions=read_strict_json(ensure_inside_repo(root,registry['precision_modes']['path'],must_exist=True))
    for mode in MODES:
        require({k:definitions[mode][k] for k in ('Q','K','V','attention_output')}==expected_mode_boundaries(mode),
                'Frozen precision definitions differ')
    checks=read_strict_json(ensure_inside_repo(root,registry['source_test_report']['path'],must_exist=True))
    require(checks.get('status')=='PASS' and checks.get('execution_exit_code')==0
        and checks.get('failed')==checks.get('errors')==checks.get('skipped')==0 and checks.get('passed',0)>0,
        'Fresh passing source tests required before freeze')
    require(checks.get('junit_sha256')==registry['test_junit']['sha256'] and checks.get('log_sha256')==registry['test_log']['sha256'],
            'Test report is not bound to actual saved output')
    require(checks.get('source_sha256')=={p:registry['source:'+p]['sha256'] for p in SCORING_FILES},
            'Source changed after the recorded pre-freeze tests')
    require(not prior_test_artifacts(root),'Prior held-out attempt/predictions already exist')
    expected=expected_owner_context(root,context,registry)
    for owner in ('member2','member3'):
        record=read_strict_json(ensure_inside_repo(root,registry[owner+'_release']['path'],must_exist=True))
        validate_owner_release(record,owner,expected)
        require(record['release_id']==proposal[owner+'_release_id'],'Owner release ID changed')
    require(proposal['fixed_threshold']==expected['fixed_threshold'],'Threshold differs')
    packet=read_strict_json(ensure_inside_repo(root,registry['b2_research_packet']['path'],must_exist=True))
    from avfuse.member1_fusion.dayc_research import validate_day9_decision
    verdict=validate_day9_decision(packet)
    require(set(packet['evidence']['source_artifact_sha256'].values()).issubset({v['sha256'] for v in registry.values()}),
            'B2 raw metric/paired/cost provenance missing from registry')
    for mode in MODES:
        require(registry['predictions_'+mode]['sha256']==packet['evidence']['prediction_sha256'][mode],
                'Prediction/evidence identity mismatch')
    owner2=read_strict_json(ensure_inside_repo(root,registry['member2_release']['path'],must_exist=True))
    require(proposal.get('heldout_cohort')==cohort_contract(owner2),'Cohort selector differs from owner release')
    require(proposal.get('owner_scientific_corrections')==owner2['limitations'],'Owner corrections changed')
    if canonical_release(owner2):
        require(registry['member2_release']['sha256']==OFFICIAL_M2_SHA,'Official release bytes changed')
        require(registry['ready_test_manifest']==registry['canonical_manifest'],'Canonical cohort alias differs')
        require(proposal['feature_path_policy']['kind']=='split_clip_cache','Canonical cohort needs original cache mapping')
    else:
        require(set(LIMITATIONS)<=set(proposal['limitations']),'Historical limitations omitted')
    require(registry['ready_test_manifest']['sha256']==owner2['ready_test_manifest']['sha256'],
            'Opaque test-manifest identity differs from owner release')
    require(proposal['decision_table']==packet['decision_table'],'Decision table changed')
    if proposal['selected_interpretation']=='restoration_conditionally_supported':
        require(verdict['H3']=='supported','Restoration policy cannot claim support absent direct/positive/coherent evidence')
    require(proposal['H2']==verdict['H2'] and proposal['H3']==verdict['H3'],'Decision interpretation changed')
    return registry,expected


def finalize_proposal(root,proposal_path,c9_approval,c8b_approval,anchor_dir,context_path):
    root=Path(root).resolve(strict=True)
    proposal_path=ensure_inside_repo(root,proposal_path,must_exist=True)
    context_path=ensure_inside_repo(root,context_path,must_exist=True)
    proposal=read_strict_json(proposal_path);context=read_strict_json(context_path)
    # Missing/stale human C9 stops before dependency reads or any final writes.
    require_human_approval(ensure_inside_repo(root,c9_approval,must_exist=True),'C9',context_sha256=sha256_file(proposal_path))
    candidates=[root/'assets/c8a_approval.json',*sorted((root/'artifacts/dayC/member1/approvals').glob('C8A_ack_*.json'))]
    actual_c8a=False
    for candidate in candidates:
        if not candidate.is_file():continue
        try:require_human_approval(candidate,'C8A',context_sha256=sha256_file(context_path));actual_c8a=True;break
        except (OSError,ValueError):continue
    require(actual_c8a,'Actual C8A context approval missing or stale')
    registry=proposal['evidence_hashes']
    b2=ensure_inside_repo(root,registry['b2_research_packet']['path'],must_exist=True)
    require_human_approval(ensure_inside_repo(root,c8b_approval,must_exist=True),'C8B',context_sha256=sha256_file(b2))
    registry,expected=validate_complete_proposal(root,proposal,context)
    # Entire required input set is rehashed immediately before committing any freeze file.
    verify_registry(root,registry)
    config=root/'configs/dayC/member1';config.mkdir(parents=True,exist_ok=True)
    policy_path=config/'frozen_precision_policy.json';run_path=config/'frozen_test_run.json'
    require_new_file(policy_path);require_new_file(run_path)
    anchor_dir=ensure_inside_repo(root,anchor_dir);anchor_dir.relative_to(root/'artifacts/dayC/member1/day9')
    require_new_file(anchor_dir);anchor_dir.mkdir(parents=True,exist_ok=False)
    c9path=ensure_inside_repo(root,c9_approval,must_exist=True)
    write_new_json(anchor_dir/'freeze_attempt_started.json',{'proposal_sha256':sha256_file(proposal_path),'C9_approval_sha256':sha256_file(c9path)})
    now=datetime.now(timezone.utc).isoformat()
    policy=deepcopy(proposal);policy.update(status='FROZEN',freeze_timestamp_utc=now,
        approved_c9_context_sha256=sha256_file(proposal_path),c9_approval={'path':str(c9path.relative_to(root)),'sha256':sha256_file(c9path)},
        source_proposal={'path':str(proposal_path.relative_to(root)),'sha256':sha256_file(proposal_path)})
    m2=read_strict_json(root/registry['member2_release']['path'])
    frozen_run={'schema_id':'avfuse-member1-frozen-test-run/1.0','status':'FROZEN','freeze_timestamp_utc':now,
        'mode_order':list(MODES),'split':'test','cohort':'100% of signed canonical held-out split; no subset/drops',
        'expected_count':m2['ready_test_manifest']['expected_count'],'batch_size':1,'backend':'sdpa',
        'device':'cpu','runtime':expected['runtime'],'checkpoint_sha256':CHECKPOINT,
        'precision':'symmetric_dynamic_per_tensor_fake_INT8_in_floating_point','layers':5,
        'score_orientation':'probability_of_fake','fixed_threshold':proposal['fixed_threshold'],
        'inputs':deepcopy(registry),'feature_path_policy':deepcopy(proposal.get('feature_path_policy',{'kind':'project_local_only'})),
        'heldout_cohort':deepcopy(proposal.get('heldout_cohort',{'kind':'ready_manifest','split':'test','expected_count':m2['ready_test_manifest']['expected_count']})),
        'one_attempt_rule':True,'no_post_test_tuning':True,'known_test_label_summary_access':True,
        'member2_release_id':proposal['member2_release_id'],'member3_release_id':proposal['member3_release_id'],
        'purpose':PURPOSE,'approved_c9_context_sha256':sha256_file(proposal_path),
        'c9_approval':policy['c9_approval'],'source_proposal':policy['source_proposal']}
    # If interrupted, any partial immutable freeze is retained and cannot launch without a complete anchor.
    policy_sha=write_new_json(policy_path,policy);run_sha=write_new_json(run_path,frozen_run)
    anchor={'schema_id':'avfuse-member1-freeze-anchor/1.0','status':'FREEZE_COMPLETE',
        'policy':{'path':str(policy_path.relative_to(root)),'sha256':policy_sha},
        'run':{'path':str(run_path.relative_to(root)),'sha256':run_sha},
        'proposal_sha256':sha256_file(proposal_path),'c9_approval_sha256':sha256_file(c9path)}
    write_new_json(anchor_dir/'freeze_anchor.json',anchor)
    return anchor


def validate_frozen_contract(policy,run):
    finite_tree(policy);finite_tree(run)
    require(policy.get('schema_id')=='avfuse-member1-precision-policy/1.0' and run.get('schema_id')=='avfuse-member1-frozen-test-run/1.0','Wrong frozen schema')
    require(policy.get('status')==run.get('status')=='FROZEN','Missing actual freeze')
    require(policy.get('checkpoint_sha256')==run.get('checkpoint_sha256')==CHECKPOINT,'Wrong frozen checkpoint')
    require(policy.get('four_frozen_modes')==run.get('mode_order')==list(MODES),'Frozen mode order differs')
    require(run.get('split')=='test' and run.get('batch_size')==1 and run.get('backend')=='sdpa'
        and run.get('device')=='cpu' and run.get('layers')==5,'Frozen execution contract differs')
    require(run.get('score_orientation')=='probability_of_fake' and run.get('one_attempt_rule') is True
        and run.get('no_post_test_tuning') is True and run.get('known_test_label_summary_access') is True,
        'Missing score/one-time/test-exposure contract')
    require(run.get('fixed_threshold')==policy.get('fixed_threshold'),'Frozen threshold mismatch')
    require(type(run.get('expected_count')) is int and run['expected_count']>0,'Missing full cohort count')
    require(run.get('precision')=='symmetric_dynamic_per_tensor_fake_INT8_in_floating_point','Precision semantics differ')
    validate_path_policy(run.get('feature_path_policy'))
    require(run['feature_path_policy']==policy.get('feature_path_policy',{'kind':'project_local_only'}),'Frozen cache path policy differs from approved proposal')
    if 'heldout_cohort' in policy:require(run.get('heldout_cohort')==policy['heldout_cohort'],'Frozen cohort policy differs')
    require(run.get('inputs')==policy.get('evidence_hashes'),'Frozen dependency registry differs')


def verify_release(root,policy_path,run_path,anchor_path,approval_file=None,member2_release=None,member3_release=None,*,dry_run=False):
    root=Path(root).resolve(strict=True)
    policy_path=ensure_inside_repo(root,policy_path,must_exist=True);run_path=ensure_inside_repo(root,run_path,must_exist=True)
    anchor_path=ensure_inside_repo(root,anchor_path,must_exist=True)
    policy=read_strict_json(policy_path);run=read_strict_json(run_path);anchor=read_strict_json(anchor_path)
    require(anchor.get('status')=='FREEZE_COMPLETE','Incomplete freeze anchor')
    require(anchor['policy']=={'path':str(policy_path.relative_to(root)),'sha256':sha256_file(policy_path)}
        and anchor['run']=={'path':str(run_path.relative_to(root)),'sha256':sha256_file(run_path)},'Frozen file/anchor mismatch')
    validate_frozen_contract(policy,run)
    c9=run['c9_approval'];c9path=ensure_inside_repo(root,c9['path'],must_exist=True);require_sha256(c9path,c9['sha256'])
    require_human_approval(c9path,'C9',context_sha256=run['approved_c9_context_sha256'])
    require(anchor['proposal_sha256']==policy['approved_c9_context_sha256']==run['approved_c9_context_sha256'],'C9 proposal context differs')
    require(anchor['c9_approval_sha256']==sha256_file(c9path) and policy['c9_approval']==run['c9_approval'],
            'C9 acknowledgement/anchor differs')
    require(policy['source_proposal']==run['source_proposal'],'Source proposal reference differs')
    source=ensure_inside_repo(root,run['source_proposal']['path'],must_exist=True)
    require_sha256(source,run['source_proposal']['sha256'])
    require(sha256_file(source)==run['approved_c9_context_sha256'],'Original proposal changed')
    original=read_strict_json(source)
    for key,value in original.items():
        if key!='status':require(policy.get(key)==value,'Frozen policy differs from C9-approved proposal')
    c10context=digest_object({'frozen_policy_sha256':sha256_file(policy_path),'frozen_run_sha256':sha256_file(run_path),
                              'freeze_anchor_sha256':sha256_file(anchor_path)})
    if approval_file is None:
        if dry_run:return {'status':'BLOCKED_PENDING_C10','C10_context_sha256':c10context,'test_rows_read':False,'test_features_read':False}
        raise PermissionError('Missing explicit C10 authorization')
    approval=ensure_inside_repo(root,approval_file,must_exist=True)
    require_human_approval(approval,'C10',context_sha256=c10context)
    if dry_run:return {'status':'AUTHORIZED_METADATA_ONLY','C10_context_sha256':c10context,'test_rows_read':False,'test_features_read':False}
    verify_registry(root,run['inputs'])
    require(not prior_test_artifacts(root),'A held-out output or attempt already exists')
    for owner,path in (('member2',member2_release),('member3',member3_release)):
        require(path is not None,'Missing '+owner+' release')
        actual=ensure_inside_repo(root,path,must_exist=True);entry=run['inputs'][owner+'_release']
        require(actual==ensure_inside_repo(root,entry['path'],must_exist=True),'Owner release path differs')
        require_sha256(actual,entry['sha256'])
        record=read_strict_json(actual)
        require(record.get('external_owner_acknowledgement') is True and record.get('release_id')==run[owner+'_release_id'],
                'Owner acknowledgement differs')
    return {'status':'READY','C10_context_sha256':c10context,'policy':policy,'run':run,
        'hashes':{'frozen_policy_sha256':sha256_file(policy_path),'frozen_run_sha256':sha256_file(run_path),
                 'freeze_anchor_sha256':sha256_file(anchor_path),'approved_c10_ack_sha256':sha256_file(approval)},
        'test_rows_read':False,'test_features_read':False}
