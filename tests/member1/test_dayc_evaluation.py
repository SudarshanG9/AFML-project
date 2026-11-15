"""Synthetic API delegation and explicit resampling-scope tests; no real statistics run."""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import pytest
from avfuse.member1_fusion.dayc_research import verify_analysis_resampling
from scripts.dayC_member1.finish_day8_evaluation import owner_stratum_statistics


def fixture_rows():
    types=('real','audio_fake','video_fake','both_fake')
    manifest=[];predictions=[]
    for n in range(1200):
        clip=f'{n:016x}';label=int(n>=300)
        manifest.append({'clip_id':clip,'dataset':'TEST_FIXTURE','source_id':str(n),'identity_id':str(n),
            'split':'val','label':label,'manipulation_type':types[n//300],'feature_path':'TEST_FIXTURE'})
        for mode in ('reference','low_precision','restore_value_output','restore_query'):
            predictions.append({'clip_id':clip,'run_id':'TEST_FIXTURE','split':'val','config':mode,
                'score_fake':0.8 if label else 0.2,'checkpoint_sha256':'a'*64})
    return manifest,predictions


def test_delegates_all_strata_contrasts_and_missing_metrics_to_original_apis(monkeypatch):
    from avfuse.member2_data import paired,evaluate
    calls=[];ci_calls=[]
    def paired_fake(fn,y,a,b,groups,n_boot,*,seed):
        calls.append((len(y),n_boot,seed,len(groups)))
        va=float(fn(y,a));vb=float(fn(y,b));delta=vb-va
        return {'a':va,'b':vb,'delta':delta,'ci_low':delta-0.01,'ci_high':delta+0.01,'excludes_zero':False}
    def ci_fake(fn,y,s,groups,n_boot,*,seed):
        ci_calls.append((len(y),n_boot,seed))
        return {'value':float(fn(y,s)),'ci_low':0.0,'ci_high':1.0}
    monkeypatch.setattr(paired,'paired_delta',paired_fake)
    monkeypatch.setattr(evaluate,'cluster_ci',ci_fake)
    strata,extra=owner_stratum_statistics(*fixture_rows(),0.5)
    assert set(strata)=={'overall','audio_fake','video_fake','both_fake'}
    assert len(calls)==80 and all(n_boot==1000 and seed==0 and n==groups for n,n_boot,seed,groups in calls)
    assert sum(n==1200 for n,_,_,_ in calls)==20 and sum(n==600 for n,_,_,_ in calls)==60
    assert len(ci_calls)==24 and all(t==(600,1000,0) for t in ci_calls)
    assert set(extra)=={'audio_fake','video_fake','both_fake'}
    for block in strata.values():
        assert len(block['contrasts'])==4
        assert any(c['a']=='restore_value_output' and c['b']=='restore_query' for c in block['contrasts'])


@pytest.mark.parametrize('n_boot,seed',[(999,0),(1000,1),(50,0)])
def test_never_downscales_or_changes_bootstrap_seed(n_boot,seed):
    with pytest.raises(ValueError):owner_stratum_statistics([],[],0.5,n_boot=n_boot,seed=seed)


def method_fixture():
    root=Path(__file__).resolve().parents[2]
    context=json.loads((root/'configs/dayC/member1/C8A_context.json').read_text())
    keys=['src/avfuse/member2_data/'+n for n in ('paired.py','evaluate.py','splits.py')]
    method={'kind':'analysis_subsets_with_original_source_cluster_bootstrap','n_boot':1000,'seed':0,
        'analysis_strata':['overall','audio_fake','video_fake','both_fake'],'fixed_category_counts_per_resample':False,
        'engine_sha256':{k:context['pilot_source_sha256'][k] for k in keys}}
    pairs=[('reference','low_precision'),('low_precision','restore_value_output'),
           ('low_precision','restore_query'),('restore_value_output','restore_query')]
    result={name:{'n':1200 if name=='overall' else 600,'contrasts':[
        {'a':a,'b':b,'metrics':{m:{'ci_low':-0.1,'ci_high':0.1} for m in ('auroc','auprc','bce','recall','real_fpr')}}
        for a,b in pairs]} for name in method['analysis_strata']}
    return {'resampling':method,'per_type_paired':result},context


@pytest.mark.parametrize('defect',[None,'fixed_counts_claim','wrong_engine','count','missing_stratum','missing_contrast','missing_metric','seed','nan'])
def test_method_scope_and_full_result_coverage_fail_closed(defect):
    paired,context=method_fixture()
    if defect=='fixed_counts_claim':paired['resampling']['fixed_category_counts_per_resample']=True
    elif defect=='wrong_engine':paired['resampling']['engine_sha256']['src/avfuse/member2_data/paired.py']='0'*64
    elif defect=='count':paired['per_type_paired']['audio_fake']['n']=599
    elif defect=='missing_stratum':paired['per_type_paired'].pop('video_fake')
    elif defect=='missing_contrast':paired['per_type_paired']['overall']['contrasts'].pop()
    elif defect=='missing_metric':paired['per_type_paired']['overall']['contrasts'][0]['metrics'].pop('bce')
    elif defect=='seed':paired['resampling']['seed']=1
    elif defect=='nan':paired['per_type_paired']['overall']['contrasts'][0]['metrics']['bce']['ci_low']=float('nan')
    if defect is None:
        result=verify_analysis_resampling(paired,context)
        assert result['analysis_stratification_verified'] and not result['fixed_category_counts_per_resample']
        assert 'pending' in result['scientific_method_review']
    else:
        with pytest.raises(ValueError):verify_analysis_resampling(paired,context)


@pytest.mark.parametrize('defect',[None,'parent_escape','absolute','missing_binding','symlink'])
def test_archived_input_bindings_are_explicit_exact_read_only_paths(tmp_path,defect):
    from scripts.dayC_member1.finish_day8_evaluation import bound_input
    path=tmp_path/'saved.json';path.write_text('TEST_FIXTURE')
    spec={'execution_project_relative':'execution_project','input_path_bindings':{'/original/saved.json':'saved.json'}}
    if defect=='parent_escape':spec['input_path_bindings']['/original/saved.json']='../saved.json'
    elif defect=='absolute':spec['input_path_bindings']['/original/saved.json']=str(path)
    elif defect=='missing_binding':spec['input_path_bindings'].clear()
    elif defect=='symlink':
        link=tmp_path/'linked.json';link.symlink_to(path);spec['input_path_bindings']['/original/saved.json']='linked.json'
    if defect is None:
        assert bound_input(spec,'/original/saved.json',base=tmp_path)==path and path.read_text()=='TEST_FIXTURE'
    else:
        with pytest.raises((PermissionError,KeyError)):bound_input(spec,'/original/saved.json',base=tmp_path)
