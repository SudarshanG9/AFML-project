"""Synthetic 1200-cohort and owner-schema guards; no real inference."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import pytest
from avfuse.member1_fusion.dayc_validation import verify_prediction_rows,verify_validation_metadata,verify_matched_benchmarks,MODES
from avfuse.member1_fusion.dayc_guards import require_sha256
from scripts.dayC_member1 import run_day8

SHA='a'*64


def cohort():
    types=('real','audio_fake','video_fake','both_fake')
    return [{'clip_id':f'{i:016x}','dataset':'TEST_FIXTURE','source_id':str(i),'identity_id':str(i),
             'label':int(i>=300),'split':'val','manipulation_type':types[i//300],
             'feature_path':f'/synthetic/val/{i:016x}/features.npz'} for i in range(1200)]


def predictions():
    return [{'clip_id':r['clip_id'],'run_id':'TEST_FIXTURE_run','split':'val','config':'restore_query',
             'score_fake':0.3,'checkpoint_sha256':SHA} for r in cohort()]


@pytest.mark.parametrize('defect',[None,'1199','1201','duplicate','mode','split','checkpoint','nan','range','labels','id','run'])
def test_query_output_guards(defect):
    rows=predictions();ids={r['clip_id'] for r in rows}
    if defect=='1199':rows.pop()
    elif defect=='1201':rows.append(dict(rows[0],clip_id='extra'))
    elif defect=='duplicate':rows[1]=dict(rows[0])
    elif defect=='mode':rows[0]['config']='low_precision'
    elif defect=='split':rows[0]['split']='test'
    elif defect=='checkpoint':rows[0]['checkpoint_sha256']='b'*64
    elif defect=='nan':rows[0]['score_fake']=float('nan')
    elif defect=='range':rows[0]['score_fake']=1.1
    elif defect=='labels':rows[0]['label']=0
    elif defect=='id':rows[0]['clip_id']='extra'
    elif defect=='run':rows[0]['run_id']='different'
    if defect is None:assert verify_prediction_rows(rows,'restore_query',ids,SHA)['count']==1200
    else:
        with pytest.raises((ValueError,RuntimeError)):verify_prediction_rows(rows,'restore_query',ids,SHA)


@pytest.mark.parametrize('defect',[None,'1199','1201','duplicate','imbalance','cross_split','test','metadata'])
def test_validation_cohort_guards(defect):
    canonical=cohort();ready=deepcopy(canonical)
    if defect=='1199':ready.pop()
    elif defect=='1201':ready.append(dict(ready[0],clip_id='extra'))
    elif defect=='duplicate':ready[1]=dict(ready[0])
    elif defect=='imbalance':canonical[0]['manipulation_type']=ready[0]['manipulation_type']='video_fake'
    elif defect=='cross_split':canonical.append(dict(canonical[0],clip_id='TEST_FIXTURE_test',split='test'))
    elif defect=='test':ready[0]['split']='test'
    elif defect=='metadata':ready[0]['source_id']='different'
    if defect is None:assert len(verify_validation_metadata(canonical,ready))==1200
    else:
        with pytest.raises((ValueError,RuntimeError)):verify_validation_metadata(canonical,ready)


def test_old_prediction_hash_drift_fails(tmp_path):
    p=tmp_path/'baseline.jsonl';p.write_text('original')
    with pytest.raises(ValueError):require_sha256(p,'0'*64)


def test_missing_c8a_stops_before_runtime_or_cache(tmp_path,monkeypatch):
    (tmp_path/'configs/dayC/member1').mkdir(parents=True)
    (tmp_path/'configs/dayC/member1/C8A_context.json').write_text('{}')
    monkeypatch.setattr(run_day8,'observed_runtime',lambda:pytest.fail('runtime touched before C8A'))
    with pytest.raises(FileNotFoundError):
        run_day8.preflight(tmp_path,tmp_path/'missing.json',Path('/old/ready'),Path('/old/canonical'),
                          Path('/old/upstream'),tmp_path/'artifacts/predictions/restore_query_val.jsonl',tmp_path/'new')


def test_existing_output_fails_before_runtime_or_cache(tmp_path,monkeypatch):
    monkeypatch.setattr(run_day8,'load_release',lambda *_:({},{}))
    output=tmp_path/'artifacts/predictions/restore_query_val.jsonl';output.parent.mkdir(parents=True);output.write_text('preserve')
    monkeypatch.setattr(run_day8,'observed_runtime',lambda:pytest.fail('runtime touched for existing target'))
    with pytest.raises(FileExistsError):
        run_day8.preflight(tmp_path,tmp_path/'ack',Path('/ready'),Path('/canonical'),Path('/upstream'),output,tmp_path/'new')
    assert output.read_text()=='preserve'


def records():
    from avfuse.member3_runtime.artifacts import build_benchmark_record
    environment={'python_version':'TEST_FIXTURE','torch_version':'TEST_FIXTURE','os_platform':'TEST_FIXTURE',
        'device_metadata':{'type':'cpu','name':None,'properties':None},'git_commit':None,'git_dirty':False,'environment_report_ref':None}
    output=[]
    for mode in MODES:
        measurement={'repeats':50,'warmup':10,'timing_samples_ms':[10.0]*50,'p50_ms':10.0,'p95_ms':10.0,
            'mean_ms':10.0,'median_ms':10.0,'device':'cpu','peak_memory_bytes':None,
            'percentile_method':'linear','synchronization':'not_required_cpu','memory_method':'not_measured'}
        output.append(build_benchmark_record(measurement,run_id='TEST_FIXTURE',experiment_id='TEST_FIXTURE',config=mode,
            scope='cached_feature_model',backend='sdpa',batch_size=1,execution_dtype='float32',
            input_boundary='cached_encoder_tokens',input_shapes={'video_tokens':[1,192,768],'audio_tokens':[1,192,768]},
            environment=environment,checkpoint={'kind':'checkpoint','sha256':SHA},
            input_identity={'kind':'feature_cache_content','sha256':'b'*64},
            conversion_overhead_included_in_latency=mode!='reference'))
    return output


@pytest.mark.parametrize('defect',[None,'missing_query','duplicate','device','scope','workload','warmup','repeats','raw','memory'])
def test_matched_benchmarks_fail_on_incomparable_records(defect):
    rows=records()
    if defect=='missing_query':rows.pop()
    elif defect=='duplicate':rows[-1]=deepcopy(rows[0])
    elif defect=='device':rows[0]['device']='cuda:0'
    elif defect=='scope':rows[0]['scope']='operator'
    elif defect=='workload':rows[0]['workload']['input_identity_sha256']='c'*64
    elif defect=='warmup':
        for r in rows:r['warmup']=9
    elif defect=='repeats':rows[0]['repeats']=49
    elif defect=='raw':rows[0]['timing_samples_ms']=[]
    elif defect=='memory':rows[0]['peak_memory_bytes']=0
    if defect is None:assert verify_matched_benchmarks(rows,SHA)['E_value_ms']==0
    else:
        with pytest.raises(ValueError):verify_matched_benchmarks(rows,SHA)


def test_subprocess_failure_keeps_failed_attempt(tmp_path,monkeypatch):
    evidence=tmp_path/'artifacts/dayC/member1/day8/TEST_FIXTURE';output=tmp_path/'artifacts/predictions/restore_query_val.jsonl'
    (tmp_path/'assets').mkdir();(tmp_path/'assets/accepted_pilot.json').write_text('{}')
    (tmp_path/'configs/dayC/member1').mkdir(parents=True);(tmp_path/'configs/dayC/member1/C8A_context.json').write_text('{}')
    ack=tmp_path/'ack.json';ack.write_text('{}')
    monkeypatch.setattr(run_day8,'preflight',lambda *_:({'pilot_source_sha256':{}},{'checkpoint_sha256':SHA,'validation_cache_mapping':{}}, {},set(),[],{},{}))
    def fail(*args):raise subprocess.CalledProcessError(1,['TEST_FIXTURE'])
    monkeypatch.setattr(run_day8,'launch_checked',fail)
    with pytest.raises(subprocess.CalledProcessError):
        run_day8.execute(tmp_path,ack,Path('/ready'),Path('/canonical'),Path('/upstream'),output,evidence)
    assert json.loads((evidence/'attempt_status.json').read_text())['status']=='FAILED'
    assert (evidence.parent/'full_validation_execution.lock').exists()
    assert not output.exists()


def test_read_only_bridge_runs_original_1200_loop_unchanged(tmp_path,monkeypatch):
    import hashlib,io,sys
    import numpy as np
    import torch
    from scripts.dayC_member1 import day8_child as child
    from scripts import dayB_inference as original
    rows=cohort();newroot=tmp_path/'cache';inventory=[]
    buffer=io.BytesIO();np.savez(buffer,video_features=np.ones((1,2),dtype=np.float32),audio_features=np.ones((1,2),dtype=np.float32))
    content=buffer.getvalue();metadata=b'{"TEST_FIXTURE":true}'
    for row in rows:
        folder=newroot/'val'/row['clip_id'];folder.mkdir(parents=True)
        (folder/'features.npz').write_bytes(content);(folder/'meta.json').write_bytes(metadata)
        inventory.append({'clip_id':row['clip_id'],'feature_sha256':hashlib.sha256(content).hexdigest(),
                          'metadata_sha256':hashlib.sha256(metadata).hexdigest()})
    ready=tmp_path/'ready.jsonl';ready.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    g4=tmp_path/'g4.json';g4.write_text(json.dumps({'cache_inventory':inventory}))
    release={'ready_manifest':str(ready),'ready_manifest_sha256':hashlib.sha256(ready.read_bytes()).hexdigest(),
             'g4_report':str(g4),'g4_report_sha256':hashlib.sha256(g4.read_bytes()).hexdigest(),
             'validation_cache_mapping':{'original_root':'/synthetic','mapped_root':str(newroot)},
             'environment':{},'runtime_amendment':{'allowed_version_changes':{}}}
    monkeypatch.setattr(child,'load_release',lambda *_:({},release))
    monkeypatch.setattr(child,'observed_runtime',lambda:{})
    monkeypatch.setattr(child,'check_environment',lambda *_:None)
    monkeypatch.setattr(child,'__file__',str(tmp_path/'scripts/dayC_member1/day8_child.py'))
    calls=[]
    class Adapter:
        checkpoint_sha256=SHA
        def predict_from_cached_features(self,features,mode):
            assert mode=='restore_query' and set(features)=={'audio_tokens','video_tokens'}
            calls.append(mode);return torch.tensor([0.3])
    monkeypatch.setattr(original,'create_adapter',lambda _:Adapter())
    monkeypatch.setattr(original,'load_features',original.load_features)
    original_loop=original.run_validation
    output=tmp_path/'output.jsonl'
    monkeypatch.setattr(sys,'argv',['child','validation','--approval-file',str(tmp_path/'ack'),'--',
        '--manifest',str(ready),'--mode','restore_query','--device','cpu','--repo-root',str(tmp_path),'--out',str(output),'--run-id','TEST_FIXTURE'])
    child.main()
    assert original.run_validation is original_loop
    assert len(calls)==1200
    result=[json.loads(line) for line in output.read_text().splitlines()]
    assert len(result)==1200 and {r['clip_id'] for r in result}=={r['clip_id'] for r in rows}
    assert ready.read_text()==''.join(json.dumps(r)+'\n' for r in rows)
