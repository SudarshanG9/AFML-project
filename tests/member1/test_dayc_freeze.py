"""Synthetic Day9 receipts and immutable-write tests; no actual project freeze."""
from datetime import datetime,timezone
import json
from pathlib import Path
import pytest
from avfuse.member1_fusion import dayc_freeze as f
from avfuse.member1_fusion.dayc_guards import write_new_json,sha256_file


def ack(path,gate,context):
    from avfuse.member1_fusion.dayc_guards import GATE_ACKS
    write_new_json(path,{'gate':gate,'ack_text':GATE_ACKS[gate],'context_sha256':context,
                         'approved_at':datetime.now(timezone.utc).isoformat(),'user_supplied':True,'scope':'TEST_FIXTURE'})


def test_source_only_draft_never_selects_or_freezes_with_missing_evidence():
    result=f.build_draft(context={},registry={})
    assert result['status']=='PROPOSED_NOT_APPROVED' and not result['ready_for_freeze']
    assert result['selected_interpretation'] is result['H2'] is result['H3'] is None
    assert any('MEMBER 2' in s for s in result['blockers'])
    assert any('MEMBER 3' in s for s in result['blockers'])
    assert any('C8B' in s for s in result['blockers'])
    assert not result['heldout_inference_authorized']


def test_actual_preparation_receipt_cannot_become_c9_or_c10():
    record={'gate':'B3_SOURCE_PREPARATION','user_supplied':True,'instruction_verbatim':f.PREPARATION_TEXT,
            'C8A_context_sha256':'a'*64,'permits_source_preparation_only':True,'does_not_authorize_freeze_or_test':True}
    f.check_preparation_authorization(record,'a'*64)
    record['does_not_authorize_freeze_or_test']=False
    with pytest.raises(ValueError):f.check_preparation_authorization(record,'a'*64)


@pytest.mark.parametrize('value',[{'loss':float('nan')},{'loss':'<REAL_VALUES>'},{'ci':[0,float('inf')]}])
def test_nonfinite_or_specification_placeholders_refused(value):
    with pytest.raises(ValueError):f.finite_tree(value)


@pytest.mark.parametrize('owner',['member2','member3'])
def test_unsigned_owner_handoff_cannot_be_inferred(owner):
    with pytest.raises(ValueError,match='acknowledgement'):f.validate_owner_release({'owner':owner},owner,{})


def test_source_registry_cannot_redirect_to_different_file(tmp_path):
    p=tmp_path/'other.py';p.write_text('TEST_FIXTURE')
    with pytest.raises(ValueError,match='redirected'):
        f.verify_registry(tmp_path,{'source:expected.py':{'path':'other.py','sha256':sha256_file(p)}},complete=False)


def test_wrong_digest_and_incomplete_registry_refused(tmp_path):
    p=tmp_path/'original';p.write_text('TEST_FIXTURE')
    with pytest.raises(ValueError):f.verify_registry(tmp_path,{'example':{'path':'original','sha256':'a'*64}},complete=False)
    with pytest.raises(ValueError,match='Missing freeze'):f.verify_registry(tmp_path,{})


@pytest.mark.parametrize('case',['missing','stale'])
def test_finalize_without_actual_c9_writes_no_configs(tmp_path,case):
    proposal=tmp_path/'proposal.json';write_new_json(proposal,{'TEST_FIXTURE':True})
    context=tmp_path/'context.json';write_new_json(context,{})
    c9=tmp_path/'c9.json'
    if case=='stale':ack(c9,'C9','a'*64)
    with pytest.raises((OSError,PermissionError)):
        f.finalize_proposal(tmp_path,proposal,c9,tmp_path/'c8b.json',tmp_path/'new',context)
    assert not (tmp_path/'configs/dayC/member1/frozen_precision_policy.json').exists()


@pytest.fixture
def committed_fixture(tmp_path,monkeypatch):
    root=tmp_path/'TEST_FIXTURE';root.mkdir()
    context=root/'context.json';write_new_json(context,{})
    approvals=root/'artifacts/dayC/member1/approvals';approvals.mkdir(parents=True)
    ack(approvals/'C8A_ack_TEST_FIXTURE.json','C8A',sha256_file(context))
    b2=root/'b2.json';write_new_json(b2,{'TEST_FIXTURE':True})
    owner=root/'owner.json';write_new_json(owner,{'ready_test_manifest':{'expected_count':2}})
    registry={'b2_research_packet':{'path':'b2.json','sha256':sha256_file(b2)},
              'member2_release':{'path':'owner.json','sha256':sha256_file(owner)}}
    proposal=root/'proposal.json'
    write_new_json(proposal,{'schema_id':'avfuse-member1-precision-policy/1.0','synthetic_fixture':True,'status':'PROPOSED_NOT_APPROVED','ready_for_freeze':True,
        'blockers':[],'checkpoint_sha256':f.CHECKPOINT,'four_frozen_modes':list(f.MODES),'decision_split':'val',
        'fixed_threshold':0.5,'evidence_hashes':registry,'member2_release_id':'TEST_FIXTURE_M2',
        'member3_release_id':'TEST_FIXTURE_M3','selected_interpretation':'inconclusive'})
    c9=root/'c9.json';ack(c9,'C9',sha256_file(proposal))
    c8b=root/'c8b.json';ack(c8b,'C8B',sha256_file(b2))
    # Isolate only the atomic freeze writer; production completeness gates are tested separately.
    monkeypatch.setattr(f,'validate_complete_proposal',lambda *a:(registry,{'runtime':{'device':'cpu'}}))
    monkeypatch.setattr(f,'verify_registry',lambda *a,**kw:registry)
    anchor_dir=root/'artifacts/dayC/member1/day9/TEST_FIXTURE_FREEZE'
    result=f.finalize_proposal(root,proposal,c9,c8b,anchor_dir,context)
    return root,proposal,c9,c8b,context,anchor_dir,result


def test_freeze_writer_creates_both_files_and_separate_anchor_exclusively(committed_fixture):
    root,proposal,c9,c8b,context,directory,result=committed_fixture
    assert result['status']=='FREEZE_COMPLETE'
    policy=root/result['policy']['path'];run=root/result['run']['path']
    assert sha256_file(policy)==result['policy']['sha256'] and sha256_file(run)==result['run']['sha256']
    assert json.loads(run.read_text())['mode_order']==list(f.MODES)
    original=policy.read_bytes()
    with pytest.raises(FileExistsError):f.finalize_proposal(root,proposal,c9,c8b,directory,context)
    assert policy.read_bytes()==original


def test_dry_release_without_c10_opens_no_test_manifest_or_features(committed_fixture,monkeypatch):
    root,_,_,_,_,directory,result=committed_fixture
    # No test-manifest path even exists in this isolated metadata fixture.
    monkeypatch.setattr(f,'verify_registry',lambda *a,**k:pytest.fail('pre-C10 dependency/manifest inspection'))
    report=f.verify_release(root,root/result['policy']['path'],root/result['run']['path'],directory/'freeze_anchor.json',dry_run=True)
    assert report['status']=='BLOCKED_PENDING_C10' and not report['test_rows_read'] and not report['test_features_read']
    assert not (root/'artifacts/dayC/member1/day10').exists()


@pytest.mark.parametrize('defect',['mode','checkpoint','threshold','precision'])
def test_frozen_contract_drift_is_rejected(committed_fixture,defect):
    root,_,_,_,_,_,result=committed_fixture
    policy=json.loads((root/result['policy']['path']).read_text());run=json.loads((root/result['run']['path']).read_text())
    if defect=='mode':run['mode_order'].pop()
    elif defect=='checkpoint':run['checkpoint_sha256']='0'*64
    elif defect=='threshold':run['fixed_threshold']=0.7
    elif defect=='precision':run['precision']='different'
    with pytest.raises(ValueError):f.validate_frozen_contract(policy,run)


def owner2_fixture():
    expected={'canonical_manifest_sha256':'a'*64,'calibration_sha256':'b'*64,'fixed_threshold':0.5}
    record={'owner':'member2','external_owner_acknowledgement':True,'release_id':'TEST_FIXTURE_M2',
        'acknowledgement_source':'TEST_FIXTURE','approved_at':datetime.now(timezone.utc).isoformat(),
        'four_modes':list(f.MODES),'checkpoint_sha256':f.CHECKPOINT,**expected,
        'protocol_sha256':'b'*64,'threshold_selected_on':'val','n_boot':1000,'seed':0,
        'score_orientation':'probability_of_fake','sealed_test':True,'test_model_predictions_used_for_policy_selection':0,
        'known_test_label_summary_access_acknowledged':True,'stratification_scope':'analysis_subsets',
        'source_group_definition':'TEST_FIXTURE connected source/identity groups',
        'ready_test_manifest':{'path':'TEST_FIXTURE_ready.jsonl','sha256':'c'*64,'expected_count':3},
        'one_time_evaluation_purpose':'TEST_FIXTURE only','limitations':['TEST_FIXTURE']}
    return record,expected


@pytest.mark.parametrize('defect',[None,'threshold','missing_mode','checkpoint','test_tuning','summary_disclosure','seed_type'])
def test_final_member2_contract_is_explicit_and_bound(defect):
    record,expected=owner2_fixture()
    if defect=='threshold':record['fixed_threshold']=0.6
    elif defect=='missing_mode':record['four_modes'].pop()
    elif defect=='checkpoint':record['checkpoint_sha256']='d'*64
    elif defect=='test_tuning':record['test_model_predictions_used_for_policy_selection']=1
    elif defect=='summary_disclosure':record['known_test_label_summary_access_acknowledged']=False
    elif defect=='seed_type':record['seed']=False
    if defect is None:f.validate_owner_release(record,'member2',expected)
    else:
        with pytest.raises(ValueError):f.validate_owner_release(record,'member2',expected)


def test_stale_c10_does_not_read_test_dependencies(committed_fixture,monkeypatch):
    root,_,_,_,_,directory,result=committed_fixture
    c10=root/'TEST_FIXTURE_C10.json';ack(c10,'C10','a'*64)
    monkeypatch.setattr(f,'verify_registry',lambda *a,**kw:pytest.fail('data before matching C10'))
    with pytest.raises(PermissionError):
        f.verify_release(root,root/result['policy']['path'],root/result['run']['path'],directory/'freeze_anchor.json',c10)


@pytest.mark.parametrize('defect',[None,'memory','scope','counts','timings'])
def test_member3_release_binds_protocol_and_actual_timings(defect):
    expected={'runtime':{'device':'cpu'},'approved_workloads':[{'role':'TEST_FIXTURE'}],
              'benchmark_evidence_sha256':['b'*64,'c'*64,'d'*64]}
    record={'owner':'member3','external_owner_acknowledgement':True,'release_id':'TEST_FIXTURE_M3',
        'acknowledgement_source':'TEST_FIXTURE','approved_at':datetime.now(timezone.utc).isoformat(),
        'four_modes':list(f.MODES),'checkpoint_sha256':f.CHECKPOINT,'runtime':expected['runtime'],
        'device':'cpu','backend':'sdpa','batch_size':1,'warmup':10,'repeats':50,'synchronization':'not_required_cpu',
        'cached_model_scope':'cached_feature_model','peak_memory_method':'not_measured','peak_memory_bytes':None,
        'conversion_overhead_included':True,'raw_samples_retained':True,'preprocessing_reported_separately':True,
        'approved_workloads':expected['approved_workloads'],'benchmark_evidence_sha256':expected['benchmark_evidence_sha256'],
        'isolated_operator_scope':'TEST_FIXTURE_operator'}
    if defect=='memory':record['peak_memory_bytes']=0
    elif defect=='scope':record['isolated_operator_scope']='cached_feature_model'
    elif defect=='counts':record['repeats']=49
    elif defect=='timings':record['benchmark_evidence_sha256']=['f'*64]
    if defect is None:f.validate_owner_release(record,'member3',expected)
    else:
        with pytest.raises(ValueError):f.validate_owner_release(record,'member3',expected)


@pytest.mark.parametrize('defect',[None,'proof_missing','different_patch','scoring_change'])
def test_only_exact_preapproved_display_repair_with_proof_is_compatible(defect):
    name='src/avfuse/member3_runtime/cli.py'
    old='c10bbe905f6d78b57590d2f2d851ec5a33c7bde914d981af661ae81ecb3b92a0'
    new='8bdff2f3604bbf7726b831c0bb6423361570e8520974b65ab10efdf1ed352dde'
    context={'pilot_source_sha256':{name:old,'scoring.py':'a'*64}}
    registry={'source:'+name:{'sha256':new},'source:scoring.py':{'sha256':'a'*64}}
    checks={'additional_source_sha256':{name:new}}
    if defect=='proof_missing':checks={}
    if defect=='different_patch':registry['source:'+name]['sha256']='b'*64
    if defect=='scoring_change':registry['source:scoring.py']['sha256']='b'*64
    if defect is None:f.verify_pilot_source_compatibility(context,registry,checks)
    else:
        with pytest.raises(ValueError):f.verify_pilot_source_compatibility(context,registry,checks)
