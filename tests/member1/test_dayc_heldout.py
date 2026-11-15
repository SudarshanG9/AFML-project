"""Tiny TEST_FIXTURE engine tests only; never open actual held-out assets."""
from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import sys
import torch
import pytest
from avfuse.member1_fusion import dayc_heldout as h
from avfuse.member1_fusion.dayc_validation import MODES

SHA='a'*64


@pytest.fixture
def fixture(tmp_path):
    root=tmp_path/'TEST_FIXTURE';root.mkdir();rows=[]
    for n in range(3):
        folder=root/'cache'/('TEST_FIXTURE_'+str(n));folder.mkdir(parents=True)
        path=folder/'features.npz';path.write_bytes(b'TEST_FIXTURE_NOT_REAL_FEATURES')
        rows.append({'clip_id':'TEST_FIXTURE_'+str(n),'split':'test','feature_path':str(path)})
    return root,rows


def fake_adapter(defect=None):
    class Adapter:
        checkpoint_sha256=SHA
        calls=[]
        def predict_from_cached_features(self,features,mode):
            self.calls.append((features['x'].item(),mode))
            if defect=='nan' and mode=='restore_query':return torch.tensor([float('nan')])
            if defect=='error' and mode=='restore_value_output':raise RuntimeError('TEST_FIXTURE injected error')
            if defect=='mutation':features['x'].add_(1)
            return torch.tensor([0.25])
    return Adapter()


def test_one_feature_load_and_exactly_one_call_per_clip_mode(fixture):
    root,rows=fixture;adapter=fake_adapter();loads=[]
    def load(path):loads.append(path);return {'x':torch.tensor(float(len(loads)))}
    outputs,identities=h.score_complete_cohort(rows,adapter,load,lambda r:h.resolve_test_feature(root,r),'TEST_FIXTURE_RUN')
    assert len(loads)==3 and len(adapter.calls)==12
    assert len(set(adapter.calls))==12 and set(outputs)==set(MODES)
    assert len(identities)==3 and all(len(values)==3 for values in outputs.values())
    out=root/'out';out.mkdir()
    hashes=h.publish_complete_predictions(out,outputs,ids={r['clip_id'] for r in rows},checkpoint=SHA,run_id='TEST_FIXTURE_RUN')
    assert set(hashes)==set(MODES) and len(list((out/'predictions').glob('*.jsonl')))==4
    with pytest.raises(FileExistsError):h.publish_complete_predictions(out,outputs,ids={r['clip_id'] for r in rows},checkpoint=SHA,run_id='TEST_FIXTURE_RUN')


@pytest.mark.parametrize('defect',['nan','error','mutation'])
def test_engine_failure_never_publishes_predictions(fixture,defect):
    root,rows=fixture;adapter=fake_adapter(defect)
    with pytest.raises((ValueError,RuntimeError)):
        h.score_complete_cohort(rows,adapter,lambda _: {'x':torch.tensor(1.)},lambda r:h.resolve_test_feature(root,r),'TEST_FIXTURE')
    assert not (root/'predictions').exists()


def test_second_attempt_rejected_even_without_outputs(fixture):
    root,_=fixture;ledger,_=h.claim_attempt(root,'TEST_FIXTURE',{})
    original=ledger.read_bytes()
    with pytest.raises(FileExistsError):h.claim_attempt(root,'TEST_FIXTURE_OTHER',{})
    assert ledger.read_bytes()==original
    h.append_event(ledger,{'event':'TEST_ATTEMPT_FAILED','status':'FAILED','run_id':'TEST_FIXTURE'})
    with pytest.raises(FileExistsError):h.claim_attempt(root,'TEST_FIXTURE_NEW',{})


@pytest.mark.parametrize('defect',['missing','symlink','outside','wrong_id','wrong_split'])
def test_bad_cache_never_loads(fixture,defect):
    root,rows=fixture;row=dict(rows[0]);p=Path(row['feature_path'])
    if defect=='missing':p.unlink()
    elif defect=='symlink':
        other=p.with_name('other.npz');p.rename(other);p.symlink_to(other)
    elif defect=='outside':row['feature_path']=str(root.parent/'outside.npz')
    elif defect=='wrong_id':row['clip_id']='DIFFERENT'
    elif defect=='wrong_split':row['split']='val'
    with pytest.raises((ValueError,OSError)):h.resolve_test_feature(root,row)


@pytest.mark.parametrize('defect',['duplicate','split','mode','checkpoint','extra','missing'])
def test_saved_score_contract_rejects_bad_rows(defect):
    rows=[{'clip_id':'TEST_FIXTURE','run_id':'TEST_FIXTURE_RUN','split':'test','config':'reference','score_fake':0.5,'checkpoint_sha256':SHA}]
    ids={'TEST_FIXTURE'}
    if defect=='duplicate':rows.append(dict(rows[0]))
    elif defect=='split':rows[0]['split']='val'
    elif defect=='mode':rows[0]['config']='restore_query'
    elif defect=='checkpoint':rows[0]['checkpoint_sha256']='b'*64
    elif defect=='extra':rows[0]['label']=0
    elif defect=='missing':rows=[]
    with pytest.raises(ValueError):h.validate_saved_rows(rows,mode='reference',ids=ids,checkpoint=SHA,run_id='TEST_FIXTURE_RUN')


def test_cohort_reconciliation_refuses_subset_and_cross_split():
    rows=[{'clip_id':'TEST_FIXTURE','dataset':'TEST_FIXTURE','source_id':'source','identity_id':'id','split':'test','label':0,'manipulation_type':'real','feature_path':'a'}]
    assert len(h.reconcile_test_metadata(rows,rows,1))==1
    with pytest.raises(ValueError):h.reconcile_test_metadata(rows,rows,2)
    leaking=rows+[dict(rows[0],clip_id='TEST_FIXTURE_VAL',split='val')]
    with pytest.raises(ValueError):h.reconcile_test_metadata(leaking,rows,1)


def test_failed_release_prevents_attempt_claim_or_row_reads(fixture,monkeypatch):
    root,_=fixture
    monkeypatch.setattr(h,'verify_release',lambda *a,**kw:(_ for _ in ()).throw(PermissionError('missing C10')))
    monkeypatch.setattr(h,'claim_attempt',lambda *a,**kw:pytest.fail('claim before authorization'))
    args=SimpleNamespace(repo_root=root,run_id='TEST_FIXTURE',out_dir=root/'artifacts/dayC/member1/day10/TEST_FIXTURE',
        frozen_policy=root/'policy',frozen_run=root/'run',freeze_anchor=root/'anchor',approval_file=root/'approval',member2_release=root/'m2',member3_release=root/'m3')
    with pytest.raises(PermissionError):h.execute_heldout(args)
    assert not (root/'artifacts/dayC/member1/day10').exists()


def test_failed_staging_never_publishes_any_final_name(fixture,monkeypatch):
    root,rows=fixture;adapter=fake_adapter()
    outputs,_=h.score_complete_cohort(rows,adapter,lambda _: {'x':torch.tensor(1.)},lambda r:h.resolve_test_feature(root,r),'TEST_FIXTURE')
    original=h.write_new_jsonl;calls=[]
    def fail_on_third(path,values):
        calls.append(path)
        if len(calls)==3:raise OSError('TEST_FIXTURE injected staging failure')
        return original(path,values)
    monkeypatch.setattr(h,'write_new_jsonl',fail_on_third)
    out=root/'out';out.mkdir()
    with pytest.raises(OSError):h.publish_complete_predictions(out,outputs,ids={r['clip_id'] for r in rows},checkpoint=SHA,run_id='TEST_FIXTURE')
    assert not (out/'predictions').exists() and (out/'.predictions_staging').is_dir()


def test_post_claim_missing_cache_records_failed_attempt_without_model(fixture,monkeypatch):
    from scripts.dayC_member1 import run_day8,colab_query_pilot
    root,rows=fixture
    canonical=[]
    for row in rows:canonical.append(dict(row,dataset='TEST_FIXTURE',source_id=row['clip_id'],identity_id=row['clip_id'],label=0,manipulation_type='real'))
    c=root/'canonical.jsonl';r=root/'ready.jsonl'
    c.write_text(''.join(json.dumps(x)+'\n' for x in canonical));r.write_bytes(c.read_bytes())
    model=root/'model/src/models.py';model.parent.mkdir(parents=True);model.write_text('TEST_FIXTURE')
    Path(rows[0]['feature_path']).unlink()
    registry={'canonical_manifest':{'path':'canonical.jsonl'},'ready_test_manifest':{'path':'ready.jsonl'},'model_source':{'path':'model/src/models.py'}}
    release={'status':'READY','run':{'inputs':registry,'device':'cpu','runtime':{},'expected_count':3},'hashes':{}}
    monkeypatch.setattr(h,'verify_release',lambda *a,**kw:release)
    monkeypatch.setattr(run_day8,'observed_runtime',lambda:{})
    monkeypatch.setattr(colab_query_pilot,'check_environment',lambda *a,**kw:None)
    from scripts import dayB_inference
    monkeypatch.setattr(dayB_inference,'create_adapter',lambda *a:pytest.fail('Model before full path verification'))
    args=SimpleNamespace(repo_root=root,run_id='TEST_FIXTURE',out_dir=root/'artifacts/dayC/member1/day10/TEST_FIXTURE',
        frozen_policy=root/'p',frozen_run=root/'run',freeze_anchor=root/'a',approval_file=root/'ack',member2_release=root/'m2',
        member3_release=root/'m3',device='cpu',canonical_manifest=c,ready_test_manifest=r,upstream=root/'model')
    with pytest.raises(FileNotFoundError):h.execute_heldout(args)
    entries=[json.loads(line) for line in (root/'artifacts/dayC/member1/day10/test_attempt_ledger.jsonl').read_text().splitlines()]
    assert [x['event'] for x in entries]==['TEST_ATTEMPT_OPENED','TEST_ATTEMPT_FAILED']
    assert (args.out_dir/'attempt_failed.json').exists() and not (args.out_dir/'predictions').exists()
