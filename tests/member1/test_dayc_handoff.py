"""Synthetic whitelisted handoff tests; no real test predictions generated."""
from pathlib import Path
import json
import pytest
from avfuse.member1_fusion.dayc_guards import write_new_jsonl,sha256_file
from avfuse.member1_fusion.dayc_validation import MODES
from scripts.dayC_member1.package_day10_handoff import build_public_payload


@pytest.fixture
def fixture(tmp_path):
    root=tmp_path/'TEST_FIXTURE';pred=root/'predictions';pred.mkdir(parents=True)
    hashes={}
    for mode in MODES:
        rows=[{'run_id':'TEST_FIXTURE','clip_id':'TEST_FIXTURE_CLIP','split':'test','config':mode,'score_fake':0.4,'checkpoint_sha256':'a'*64}]
        path=pred/(mode+'_test.jsonl');write_new_jsonl(path,rows);hashes[mode]=sha256_file(path)
    audit={'status':'READY_FOR_MEMBER2_SCORING','run_id':'TEST_FIXTURE','count_per_mode':1,'prediction_sha256':hashes}
    frozen={'checkpoint_sha256':'a'*64,'fixed_threshold':0.5,'inputs':{
        name:{'path':'private/do_not_export','sha256':'b'*64} for name in ('canonical_manifest','calibration','member2_release','member3_release')}}
    return root,audit,frozen


def test_payload_contains_only_scores_public_hashes_and_owner_instructions(fixture):
    root,audit,frozen=fixture;payload=build_public_payload(root,audit,frozen)
    assert len([n for n in payload if n.endswith('.jsonl')])==4
    assert not any(Path(n).suffix in {'.pth','.npz','.pt','.mp4','.wav'} for n in payload)
    assert not any(b'private/do_not_export' in b for b in payload.values())
    for name,content in payload.items():
        if name.endswith('.jsonl'):
            row=json.loads(content);assert set(row)=={'run_id','clip_id','split','config','score_fake','checkpoint_sha256'}
    assert json.loads(payload['frozen_public_references.json'])['test_metrics_computed_by_member1'] is False


@pytest.mark.parametrize('defect',['not_audited','changed','labels','different_ids'])
def test_payload_refuses_unverified_or_changed_scores(fixture,defect):
    root,audit,frozen=fixture
    if defect=='not_audited':audit['status']='FAILED'
    else:
        path=root/'predictions/restore_query_test.jsonl';rows=[json.loads(path.read_text())]
        if defect=='changed':path.write_text('changed')
        else:
            if defect=='labels':rows[0]['label']=0
            else:rows[0]['clip_id']='OTHER_TEST_FIXTURE'
            path.write_text(json.dumps(rows[0])+'\n');audit['prediction_sha256']['restore_query']=sha256_file(path)
    with pytest.raises((ValueError,PermissionError)):build_public_payload(root,audit,frozen)
