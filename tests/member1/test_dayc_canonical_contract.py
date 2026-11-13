"""Synthetic canonical-selector/cache fixtures only; actual sealed rows stay closed."""
from copy import deepcopy
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
import pytest
from avfuse.member1_fusion import dayc_test_contract as c,dayc_freeze as f,dayc_heldout as h
from avfuse.member1_fusion.dayc_guards import sha256_file,write_new_jsonl


def owner():
    return {'owner':'member2','external_owner_acknowledgement':True,'release_id':'M2-REL-20261113-01',
        'acknowledgement_source':'TEST_FIXTURE','approved_at':datetime.now(timezone.utc).isoformat(),
        'four_modes':list(f.MODES),'checkpoint_sha256':f.CHECKPOINT,'canonical_manifest_sha256':c.CANONICAL_SHA,
        'protocol_sha256':'b'*64,'fixed_threshold':0.5,'threshold_selected_on':'val','n_boot':1000,'seed':0,
        'score_orientation':'probability_of_fake (checked: ok for all val configs)','sealed_test':True,
        'test_model_predictions_used_for_policy_selection':False,'known_test_label_summary_access_acknowledged':True,
        'stratification_scope':'analysis_subsets: per fake type, that type + all real; whole groups resampled with replacement; category counts vary per replicate',
        'source_group_definition':'TEST_FIXTURE groups',
        'ready_test_manifest':{'path':"Drive avfuse/m2_handoff/manifest_whole.jsonl (rows with split == 'test'; no separate test file, not regenerated)",'sha256':c.CANONICAL_SHA,'expected_count':2000},
        'one_time_evaluation_purpose':'TEST_FIXTURE','limitations':['TEST_FIXTURE owner finding']}


@pytest.mark.parametrize('defect',[None,'true','float_zero','orientation','strata','count','hash'])
def test_exact_official_field_semantics_without_rewriting(defect):
    record=owner();expected={'canonical_manifest_sha256':c.CANONICAL_SHA,'calibration_sha256':'b'*64,'fixed_threshold':0.5}
    if defect=='true':record['test_model_predictions_used_for_policy_selection']=True
    if defect=='float_zero':record['test_model_predictions_used_for_policy_selection']=0.0
    if defect=='orientation':record['score_orientation']='probability_of_real'
    if defect=='strata':record['stratification_scope']='any grouping'
    if defect=='count':record['ready_test_manifest']['expected_count']=1999
    if defect=='hash':record['ready_test_manifest']['sha256']='c'*64
    original=deepcopy(record)
    if defect is None:
        f.validate_owner_release(record,'member2',expected);assert c.cohort_contract(record)['kind']=='canonical_split'
    else:
        with pytest.raises(ValueError):f.validate_owner_release(record,'member2',expected)
    assert record==original


def policy(root):
    return {'kind':'split_clip_cache','cache_root':str(root),'split':'test','feature_filename':'features.npz','metadata_filename':'meta.json'}


@pytest.fixture
def cache(tmp_path):
    root=tmp_path/'TEST_FIXTURE_CACHE';folder=root/'test/TEST_FIXTURE';folder.mkdir(parents=True)
    feature=folder/'features.npz';np.savez(feature,video_features=np.zeros((3,768),np.float32),audio_features=np.zeros((4,768),np.float32))
    (folder/'meta.json').write_text(json.dumps({'status':'ok','clip_id':'TEST_FIXTURE'}))
    row={'clip_id':'TEST_FIXTURE','split':'test','feature_path':'untouched/placeholder.pt'}
    return root,row,feature


@pytest.mark.parametrize('defect',[None,'traversal','val','symlink','parent_symlink','missing_meta','dtype','nan','shape','status','wrong_meta_id'])
def test_readonly_mapping_full_cache_checks(cache,defect):
    root,row,path=cache;meta=path.parent/'meta.json'
    if defect=='traversal':row['clip_id']='../TEST_FIXTURE'
    if defect=='val':row['split']='val'
    if defect=='symlink':
        other=path.with_name('other.npz');path.rename(other);path.symlink_to(other)
    if defect=='parent_symlink':
        original=path.parent;renamed=original.with_name('moved');original.rename(renamed);original.symlink_to(renamed,target_is_directory=True)
    if defect=='missing_meta':meta.unlink()
    if defect in {'dtype','nan','shape'}:
        video=np.zeros((3,768),np.float64 if defect=='dtype' else np.float32)
        if defect=='nan':video[0,0]=np.nan
        if defect=='shape':video=video[:,:767]
        np.savez(path,video_features=video,audio_features=np.zeros((4,768),np.float32))
    if defect=='status':meta.write_text('{"status":"failed"}')
    if defect=='wrong_meta_id':meta.write_text('{"status":"ok","clip_id":"OTHER"}')
    original=dict(row)
    if defect is None:
        before=path.read_bytes();resolved=h.resolve_test_feature(root,row,policy(root));assert resolved==path
        record=c.inspect_cache_identity(row,resolved);assert record['metadata_sha256']==sha256_file(meta)
        assert path.read_bytes()==before
    else:
        with pytest.raises((ValueError,OSError)):c.inspect_cache_identity(row,h.resolve_test_feature(root,row,policy(root)))
    assert row==original


def test_policy_construction_opens_no_external_directory(monkeypatch):
    monkeypatch.setattr(Path,'is_dir',lambda *a:pytest.fail('data access during proposal'))
    monkeypatch.setattr(Path,'resolve',lambda *a,**kw:pytest.fail('data access during proposal'))
    assert c.validate_path_policy(policy(Path('/content/UNMOUNTED_TEST_FIXTURE')))['split']=='test'


def test_canonical_selector_uses_same_manifest_once_and_no_replacement(tmp_path,monkeypatch):
    from avfuse.member2_data import manifest
    path=tmp_path/'canonical.jsonl'
    rows=[{'clip_id':name,'dataset':'TEST_FIXTURE','source_id':name,'identity_id':name,'split':split,
        'label':0,'manipulation_type':'real','feature_path':'unchanged.pt'} for name,split in [('TEST_FIXTURE_VAL','val'),('TEST_FIXTURE_TEST','test')]]
    write_new_jsonl(path,rows);digest=sha256_file(path)
    entry={'path':path.name,'sha256':digest};calls=[];original=manifest.read_manifest
    def read(p):calls.append(p);return original(p)
    monkeypatch.setattr(manifest,'read_manifest',read)
    frozen={'inputs':{'canonical_manifest':entry,'ready_test_manifest':entry},'expected_count':1,
        'heldout_cohort':{'kind':'canonical_split','split':'test','expected_count':1,'manifest_sha256':digest}}
    result=h.read_frozen_cohort(tmp_path,frozen)
    assert len(calls)==1 and [r['clip_id'] for r in result]==['TEST_FIXTURE_TEST']
    assert sha256_file(path)==digest and len(list(tmp_path.iterdir()))==1
    frozen['heldout_cohort']['expected_count']=2
    with pytest.raises(ValueError):h.read_frozen_cohort(tmp_path,frozen)


def test_canonical_draft_explicitly_blocks_missing_mapping_and_never_infers_owner3():
    record=owner();expected={'canonical_manifest_sha256':c.CANONICAL_SHA,'calibration_sha256':'b'*64,'fixed_threshold':0.5}
    result=f.build_draft(context={},registry={},member2_release=record,expected=expected,c8b_accepted=True,interpretation='reference_preferred')
    assert not result['ready_for_freeze'] and any('path policy' in v for v in result['blockers'])
    assert any('MEMBER 3' in v for v in result['blockers'])
    assert result['heldout_cohort']['kind']=='canonical_split'


def test_handoff_reuses_exact_1200_validation_bytes_and_rejects_drift(tmp_path):
    from scripts.dayC_member1.package_day10_handoff import add_accepted_query_validation
    path=tmp_path/'restore_query_val.jsonl'
    rows=[{'clip_id':'TEST_FIXTURE_'+str(i),'split':'val','run_id':'TEST_FIXTURE','config':'restore_query','score_fake':0.5,'checkpoint_sha256':f.CHECKPOINT} for i in range(1200)]
    write_new_jsonl(path,rows);original=path.read_bytes()
    frozen={'checkpoint_sha256':f.CHECKPOINT,'inputs':{'predictions_restore_query':{'path':path.name,'sha256':sha256_file(path)}}}
    payload={};add_accepted_query_validation(tmp_path,payload,frozen)
    assert payload['predictions/restore_query_val.jsonl']==original and path.read_bytes()==original
    path.write_bytes(original+b'\n')
    with pytest.raises(ValueError):add_accepted_query_validation(tmp_path,{},frozen)


@pytest.mark.parametrize('defect',['relative','traversal','root','wrong_split','extra'])
def test_frozen_policy_rejects_ambiguous_roots_or_templates(defect):
    value=policy(Path('/content/TEST_FIXTURE'))
    if defect=='relative':value['cache_root']='relative'
    if defect=='traversal':value['cache_root']='/content/../test'
    if defect=='root':value['cache_root']='/'
    if defect=='wrong_split':value['split']='val'
    if defect=='extra':value['override_feature_path']=True
    with pytest.raises(ValueError):c.validate_path_policy(value)


def test_changed_mapping_cannot_match_frozen_policy():
    # Metadata-only pure check: no directories or files need exist.
    from avfuse.member1_fusion.dayc_freeze import validate_frozen_contract
    approved=policy(Path('/content/TEST_FIXTURE'))
    p={'schema_id':'avfuse-member1-precision-policy/1.0','status':'FROZEN','checkpoint_sha256':f.CHECKPOINT,
        'four_frozen_modes':list(f.MODES),'fixed_threshold':0.5,'feature_path_policy':approved,'evidence_hashes':{}}
    r={'schema_id':'avfuse-member1-frozen-test-run/1.0','status':'FROZEN','checkpoint_sha256':f.CHECKPOINT,
        'mode_order':list(f.MODES),'split':'test','batch_size':1,'backend':'sdpa','device':'cpu','layers':5,
        'score_orientation':'probability_of_fake','one_attempt_rule':True,'no_post_test_tuning':True,
        'known_test_label_summary_access':True,'fixed_threshold':0.5,'expected_count':1,
        'precision':'symmetric_dynamic_per_tensor_fake_INT8_in_floating_point','feature_path_policy':deepcopy(approved),'inputs':{}}
    validate_frozen_contract(p,r)
    r['feature_path_policy']['cache_root']='/content/OTHER'
    with pytest.raises(ValueError):validate_frozen_contract(p,r)


def test_original_acknowledged_release_is_accepted_without_modifying_bytes():
    # Public owner metadata only; never parse canonical/test cohort rows.
    path=Path(__file__).resolve().parents[2]/'docs/member1/dayC/releases/M2-REL-20261113-01/m2_release.json'
    original=path.read_bytes();record=json.loads(original)
    expected={'canonical_manifest_sha256':c.CANONICAL_SHA,
        'calibration_sha256':'358c037242f6980fed2a8d0ff0ef830f17b4789676b391c8f6dff245842b37dd',
        'fixed_threshold':0.6627789735794068}
    f.validate_owner_release(record,'member2',expected)
    assert sha256_file(path)==c.OFFICIAL_M2_SHA and path.read_bytes()==original
    record['test_cache']['path']='Drive/<UNRESOLVED>/features.npz'
    with pytest.raises(ValueError):f.validate_owner_release(record,'member2',expected)
