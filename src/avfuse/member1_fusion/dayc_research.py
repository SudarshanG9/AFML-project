"""Member 1 read-only checks of owner-supplied statistics; never computes new CIs."""
import math
from pathlib import Path
from avfuse.member1_fusion.dayc_guards import read_strict_json,sha256_file
from avfuse.member1_fusion.dayc_validation import MODES,verify_prediction_rows,verify_matched_benchmarks
from avfuse.member2_data.predictions import read_predictions
from avfuse.member2_data.manifest import read_manifest
from avfuse.shared.jsonl import read_jsonl

LIMITATIONS = [
    'Validation checkpoint-selection bias; validation findings are not held-out generalization.',
    'Source grouping is not independently verified speaker separation; checkpoint-training overlap is unverified.',
    'Silence/duration shortcuts remain possible.',
    'Previously disclosed test-label-derived summary inspection occurred; do not claim completely untouched test labels.',
    'Fake INT8 executes in floating point; no real integer kernel acceleration claim.',
    'The all-five-layer intervention is cumulative; deeper upstream hidden states can differ.',
    'BCE intervals do not automatically include benchmark timing uncertainty.',
    'Approved dependency drift does not establish original-versus-current population/runtime equivalence.',
]


def require(ok,message):
    if not ok:raise ValueError(message)


def interval(value):
    low,high=value['ci_low'],value['ci_high']
    require(all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in (low,high))
            and low<=high,'Invalid confidence interval')
    return [low,high]


def verify_analysis_resampling(paired, context):
    """Verify explicit analysis strata and unchanged owner engine, not fixed-count resampling."""
    method=paired.get('resampling',{})
    required={'overall','audio_fake','video_fake','both_fake'}
    require(method.get('kind')=='analysis_subsets_with_original_source_cluster_bootstrap',
            'Missing precise resampling method')
    require(method.get('n_boot')==1000 and method.get('seed')==0,'Bootstrap protocol changed')
    require(set(method.get('analysis_strata',[]))==required,'Missing analysis stratum')
    require(method.get('fixed_category_counts_per_resample') is False,
            'Do not label the original owner engine fixed-count stratified resampling')
    for name in ('paired.py','evaluate.py','splits.py'):
        path='src/avfuse/member2_data/'+name
        require(method.get('engine_sha256',{}).get(path)==context['pilot_source_sha256'][path],
                'Member 2 engine source changed')
    strata=paired.get('per_type_paired',{})
    require(set(strata)==required,'Missing per-stratum paired results')
    expected={('reference','low_precision'),('low_precision','restore_value_output'),
              ('low_precision','restore_query'),('restore_value_output','restore_query')}
    for name,block in strata.items():
        require(block.get('n')==(1200 if name=='overall' else 600),'Wrong paired stratum cohort')
        contrasts=block.get('contrasts',[])
        require(len(contrasts)==4 and {(c['a'],c['b']) for c in contrasts}==expected,
                'Missing/duplicate paired contrast')
        for contrast in contrasts:
            require(set(contrast['metrics'])=={'auroc','auprc','bce','recall','real_fpr'},'Missing paired metric')
            for metric in contrast['metrics'].values():interval(metric)
    return {'analysis_stratification_verified':True,'fixed_category_counts_per_resample':False,
            'method':'original Member 2 source-cluster bootstrap within overall and each fake-vs-real analysis',
            'scientific_method_review':'C8B lead review pending; no resampling algorithm changed'}


def read_four_mode_validation_evidence(*,metric_json,paired_json,prediction_paths,benchmark_jsonl,canonical_manifest):
    metric=read_strict_json(metric_json);paired=read_strict_json(paired_json)
    manifest=read_manifest(canonical_manifest);val=[r for r in manifest if r['split']=='val']
    require(len(val)==1200 and len({r['clip_id'] for r in val})==1200,'Wrong validation cohort')
    from collections import Counter
    require(Counter(r['manipulation_type'] for r in val)=={t:300 for t in ('real','audio_fake','video_fake','both_fake')},'Unbalanced validation')
    from avfuse.member2_data.splits import split_audit
    require(split_audit(manifest)['passed'],'Cross-split source leakage')
    require(metric['manifest_sha256']==paired['manifest_sha256']==sha256_file(canonical_manifest),'Wrong manifest provenance')
    require(metric['split']=='val' and metric['n_boot']==1000 and metric['seed']==0,'Wrong bootstrap/split protocol')
    require(set(metric['configs'])==set(MODES),'Missing metric mode')
    threshold=metric['threshold'];require(isinstance(threshold,(int,float)) and math.isfinite(threshold),'Missing threshold')
    root=Path(__file__).resolve().parents[3]
    context=read_strict_json(root/'configs/dayC/member1/C8A_context.json')
    spec=context['experiment_spec'];proto_item=spec['local_inputs']['calibration']
    protocol_path=root/proto_item['path']
    require(sha256_file(protocol_path)==proto_item['sha256'],'Changed original calibration')
    protocol=read_strict_json(protocol_path)
    require(protocol['threshold']==threshold and protocol['manifest_sha256']==metric['manifest_sha256'],'Wrong fixed threshold')
    expected=protocol['checkpoint_sha256'];ids={r['clip_id'] for r in val}
    require(set(prediction_paths)==set(MODES),'Missing prediction arm')
    hashes={}
    for mode,path in prediction_paths.items():
        rows=read_predictions(path);verify_prediction_rows(rows,mode,ids,expected);hashes[mode]=sha256_file(path)
        if mode!='restore_query':
            require(hashes[mode]==spec['local_inputs']['predictions_'+mode]['sha256'],'Changed original predictions')
        result=metric['configs'][mode]
        require(result['n']==1200 and result['n_real']==300 and result['n_fake']==900,'Metric cohort mismatch')
        require(result['threshold']==threshold,'Per-mode threshold changed')
        require(set(result['per_type'])=={'audio_fake','video_fake','both_fake'},'Missing fake-vs-real stratum')
        for kind,block in result['per_type'].items():
            require(block['n_real']==block['n_fake']==300,'Empty/wrong fake-vs-real stratum')
        # Delegate BCE to the existing Member 2 implementation to cross-check sign/input identity.
        from avfuse.member2_data.predictions import join
        from avfuse.member2_data.evaluate import bce
        joined=join(manifest,rows,'val')
        actual=float(bce([r['label'] for r in joined],[r['score_fake'] for r in joined]))
        require(math.isclose(actual,result['bce']['value'],rel_tol=1e-12,abs_tol=1e-12),'Metric prediction provenance/BCE mismatch')
        interval(result['bce'])
    costs=verify_matched_benchmarks(read_jsonl(benchmark_jsonl),expected,run_dir=Path(benchmark_jsonl).parent)
    restoration=paired['restoration'];require(restoration['n']==1200 and restoration['split']=='val','Wrong paired cohort')
    losses={m:metric['configs'][m]['bce']['value'] for m in MODES}
    require(math.isclose(restoration['quantization_damage'],losses['low_precision']-losses['reference'],abs_tol=1e-12),'Damage sign/value mismatch')
    rows={r['config']:r for r in restoration['rows']};require(set(rows)==set(MODES[2:]),'Missing restoration contrast')
    for mode,row in rows.items():
        require(math.isclose(row['R'],losses['low_precision']-losses[mode],abs_tol=1e-12),'Recovery sign/value mismatch')
        interval(row)
    interval(restoration['damage_ci'])
    direct=[c for c in metric['paired'] if {c['a'],c['b']}=={'restore_query','restore_value_output'}]
    require(len(direct)==1,'Direct paired value/query contrast missing')
    direct=direct[0];ci=direct['metrics']['bce'];interval(ci)
    require(direct['n']==1200 and direct['split']=='val','Direct contrast cohort mismatch')
    direction=1 if direct['a']=='restore_value_output' else -1
    require(math.isclose(direction*ci['delta'],losses['restore_query']-losses['restore_value_output'],abs_tol=1e-12),'H3 direct sign/value mismatch')
    h3_ci=[direction*ci['ci_low'],direction*ci['ci_high']];h3_ci.sort()
    blockers=[]
    method_review=verify_analysis_resampling(paired,context)
    return {'status':'RESEARCH_REVIEW_REQUIRED','blockers':blockers,'resampling_review':method_review,'losses':losses,
        'damage_ci':interval(restoration['damage_ci']),
        'recovery_ci':{m:interval(rows[m]) for m in MODES[2:]},'h3_ci':h3_ci,'costs':costs,
        'cost_reliability_approved':False,'prediction_sha256':hashes,'threshold':threshold,
        'per_type_results':paired['per_type_paired'],
        'source_artifact_sha256':{str(p):sha256_file(p) for p in (metric_json,paired_json,benchmark_jsonl,canonical_manifest)}}


def build_h2_h3_decision_table(evidence):
    losses=evidence['losses'];D=losses['low_precision']-losses['reference']
    Rv=losses['low_precision']-losses['restore_value_output'];Rq=losses['low_precision']-losses['restore_query']
    direct=losses['restore_query']-losses['restore_value_output']
    ci=evidence['damage_ci'];hci=evidence['h3_ci']
    costs=evidence.get('costs',{});Ev=costs.get('E_value_ms');Eq=costs.get('E_query_ms')
    reliable=evidence.get('cost_reliability_approved') is True
    validcost=reliable and all(isinstance(x,(int,float)) and math.isfinite(x) and x>0 for x in (Ev,Eq))
    recoveries={}
    for mode,value in (('restore_value_output',Rv),('restore_query',Rq)):
        rci=evidence['recovery_ci'][mode]
        recoveries[mode]={'R':value,'ci':rci,'share_recovered':value/D if D>0 else None,
                         'positive_recovery_supported':D>0 and ci[0]>0 and value>0 and rci[0]>0}
    return {'damage':D,'damage_ci':ci,'positive_damage_supported':D>0 and ci[0]>0,
        'damage_interpretation':'no observed positive damage to restore; motivation unsubstantiated' if D<=0 else
                                 ('positive damage supported' if ci[0]>0 else 'positive point estimate; damage evidence inconclusive'),
        'recoveries':recoveries,'H3_direct':direct,'H3_ci':hci,
        'value_better_than_query_supported':direct>0 and hci[0]>0,
        'H3_verdict':'value has lower BCE' if direct>0 and hci[0]>0 else
                     ('query has lower BCE' if direct<0 and hci[1]<0 else 'inconclusive; no equivalence claim'),
        'E_value_ms':Ev,'E_query_ms':Eq,'Rv_per_ms':Rv/Ev if validcost else None,
        'Rq_per_ms':Rq/Eq if validcost else None,
        'cost_interpretation':'measured descriptive normalization; timing uncertainty not included in BCE CIs' if validcost else
                              'undefined: requires positive, comparable, owner-reviewed reliable extra costs',
        'equal_cost_established':False,'policy_selected':False,'evidence_status':evidence.get('status','UNVERIFIED')}


def describe_limitations(evidence):
    return LIMITATIONS+list(evidence.get('blockers',[]))


def validate_day9_decision(packet):
    """Source-only validation of actual B2 findings. Does not select/freeze a policy."""
    require(packet.get('status')=='RESEARCH_REVIEW_REQUIRED' and packet.get('policy_frozen') is False,
            'Actual unfrozen B2 research packet required')
    evidence=packet['evidence'];table=packet['decision_table']
    require(evidence.get('status')=='RESEARCH_REVIEW_REQUIRED' and not evidence.get('blockers'),
            'Incomplete four-mode statistical evidence')
    require(table==build_h2_h3_decision_table(evidence),'Scientific decision table differs from evidence')
    require(set(evidence['losses'])=={'reference','low_precision','restore_value_output','restore_query'},'Missing mode')
    require(set(evidence.get('per_type_results',{}))=={'overall','audio_fake','video_fake','both_fake'},
            'Missing per-stratum scientific evidence')
    for value in evidence['losses'].values():
        require(isinstance(value,(float,int)) and math.isfinite(value) and value>=0,'Invalid BCE')
    for ci in [evidence['damage_ci'],evidence['h3_ci'],*evidence['recovery_ci'].values()]:
        interval({'ci_low':ci[0],'ci_high':ci[1]})
    coherent=True
    for kind in ('audio_fake','video_fake','both_fake'):
        block=evidence['per_type_results'][kind]
        require(block['n']==600,'Wrong manipulation stratum count')
        direct=[c for c in block['contrasts'] if c['a']=='restore_value_output' and c['b']=='restore_query']
        require(len(direct)==1,'Missing direct per-stratum Q-minus-V contrast')
        bce=direct[0]['metrics']['bce'];interval(bce)
        coherent=coherent and bce['delta']>=0 and bce['ci_high']>=0
    D=table['damage'];h2='supported' if table['positive_damage_supported'] else ('not_supported' if D<=0 else 'inconclusive')
    value=table['recoveries']['restore_value_output']
    measured=evidence.get('costs',{}).get('status')=='PASS'
    supported=table['positive_damage_supported'] and value['positive_recovery_supported'] and table['value_better_than_query_supported'] and coherent and measured
    h3='supported' if supported else ('not_supported' if D<=0 or table['H3_ci'][1]<=0 else 'inconclusive')
    return {'H2':h2,'H3':h3,'per_type_direction_coherent':coherent,'cost_assessment_available':measured,
            'per_cost_superiority_claim_permitted':False,'policy_selected':False,
            'limitations':describe_limitations(evidence)}
