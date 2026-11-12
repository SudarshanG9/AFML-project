"""User-run B2 evaluation only: unchanged Member 2 APIs, no inference or timing rerun."""
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
import zipfile

HERE=Path(__file__).resolve().parent


def bound_input(spec,logical_path,*,base=None):
    """Resolve exact archived metadata/score bytes without modifying any original manifest."""
    base=Path(base) if base is not None else HERE
    mapping=spec.get('input_path_bindings',{})
    if str(logical_path) in mapping:
        relative=Path(mapping[str(logical_path)])
        if relative.is_absolute() or '..' in relative.parts:raise PermissionError('Unsafe archived input binding')
        requested=base/relative
        if requested.is_symlink():raise PermissionError('Symlink archived input')
        actual=requested.resolve(strict=True)
        if not actual.is_relative_to(base.resolve()):raise PermissionError('Archived input escapes evaluation capsule')
        return actual
    if spec.get('execution_project_relative') is not None:
        raise KeyError('Missing explicit archived input binding: '+str(logical_path))
    return Path(logical_path)


def finite_tree(value):
    import math
    if isinstance(value,dict):
        for child in value.values():finite_tree(child)
    elif isinstance(value,list):
        for child in value:finite_tree(child)
    elif isinstance(value,float) and not math.isfinite(value):
        raise ValueError('Nonfinite statistic/CI; preserve and stop')


def owner_stratum_statistics(manifest,predictions,threshold,*,n_boot=1000,seed=0):
    """Reuse accepted Day B per-type orchestration with unchanged owner functions."""
    if n_boot!=1000 or seed!=0:raise ValueError('Exactly 1000 source bootstrap replicates, seed 0 required')
    import numpy as np
    from avfuse.member2_data.paired import aligned,paired_delta
    from avfuse.member2_data.evaluate import _auroc,_auprc,bce,recall_at,fpr_at,cluster_ci
    from avfuse.member2_data.splits import group_clusters
    modes=('reference','low_precision','restore_value_output','restore_query')
    y,ids,types,scores=aligned(manifest,predictions,modes,'val')
    if len(ids)!=1200:raise ValueError('Full 1200 cohort required')
    cluster_map=group_clusters(manifest);groups=np.array([cluster_map[i] for i in ids])
    functions={'auroc':_auroc,'auprc':_auprc,'bce':bce,
               'recall':lambda yy,ss:recall_at(yy,ss,threshold),'real_fpr':lambda yy,ss:fpr_at(yy,ss,threshold)}
    pairs=(('reference','low_precision'),('low_precision','restore_value_output'),
           ('low_precision','restore_query'),('restore_value_output','restore_query'))
    result={};additional={}
    for name in ('overall','audio_fake','video_fake','both_fake'):
        mask=np.ones(len(y),dtype=bool) if name=='overall' else ((types=='real')|(types==name))
        expected=1200 if name=='overall' else 600
        if int(mask.sum())!=expected or int((y[mask]==0).sum())!=300:
            raise ValueError('Wrong fake-vs-real analysis stratum')
        print('SOURCE_BOOTSTRAP_STRATUM='+name+' n='+str(expected)+' n_boot=1000 seed=0',flush=True)
        block=[]
        for first,second in pairs:
            metrics={k:paired_delta(fn,y[mask],scores[first][mask],scores[second][mask],groups[mask],n_boot,seed=seed)
                     for k,fn in functions.items()}
            block.append({'a':first,'b':second,'metrics':metrics})
        result[name]={'n':expected,'n_real':300,'n_fake':expected-300,
                      'n_source_clusters':len(set(groups[mask].tolist())),'contrasts':block}
        # Owner evaluate_config already emits per-type AUROC/AUPRC/recall. Supply its missing BCE/FPR CIs via owner cluster_ci.
        if name!='overall':
            additional[name]={mode:{k:cluster_ci(fn,y[mask],scores[mode][mask],groups[mask],n_boot,seed=seed)
                                   for k,fn in (('bce',bce),('real_fpr',functions['real_fpr']))} for mode in modes}
    finite_tree(result);finite_tree(additional)
    return result,additional


def main():
    spec=json.loads((HERE/'EVALUATION_INPUTS.json').read_text())
    if spec.get('execution_project_relative') is not None:
        relative=Path(spec['execution_project_relative'])
        if relative.is_absolute() or '..' in relative.parts:raise PermissionError('Unsafe execution project location')
        root=(HERE/relative).resolve(strict=True)
        if not root.is_relative_to(HERE.resolve()):raise PermissionError('Execution project escapes capsule')
    else:
        root=Path(spec['original_root']).resolve(strict=True)
    sys.path[:0]=[str(root/'src'),str(root)]
    from avfuse.member1_fusion.dayc_guards import read_strict_json,write_new_json,sha256_file,require_sha256
    from avfuse.member1_fusion.dayc_validation import verify_prediction_rows,verify_matched_benchmarks,MODES
    from scripts.dayC_member1.run_day8 import load_release,observed_runtime
    from scripts.dayC_member1.colab_query_pilot import check_environment,verify_package
    output=root/'artifacts/dayC/member1/day8'
    run=output/('B2_EVALUATION_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    run.mkdir(parents=True,exist_ok=False);stage='evaluation_preflight'
    print('EVALUATION_RUN_DIR='+str(run),flush=True)
    additional_artifacts={}
    try:
        verify_package(root)
        require_sha256(root/'PACKAGE_MANIFEST.json',spec['package_manifest_sha256'])
        require_sha256(root/'configs/dayC/member1/C8A_context.json',spec['C8A_context_sha256'])
        context,release=load_release(root,root/'assets/c8a_approval.json')
        runtime=observed_runtime()
        check_environment(runtime,release['environment'],release['runtime_amendment']['allowed_version_changes'])
        for path,expected in spec['inputs'].items():require_sha256(bound_input(spec,path),expected)
        predictions=[root/'artifacts/predictions'/ (mode+'_val.jsonl') for mode in MODES]
        canonical=bound_input(spec,release['canonical_manifest']);calibration=root/'artifacts/reports/dayB_validation_calibration.json'
        from avfuse.member2_data.manifest import read_manifest
        from avfuse.member2_data.predictions import read_predictions
        from avfuse.shared.jsonl import read_jsonl
        whole=read_manifest(canonical);val=[r for r in whole if r['split']=='val'];ids={r['clip_id'] for r in val}
        for mode,path in zip(MODES,predictions):verify_prediction_rows(read_predictions(path),mode,ids,release['checkpoint_sha256'])
        typical=next(bound_input(spec,p) for p in spec['inputs'] if p.endswith('/four_modes_typical.jsonl'))
        for path in spec['inputs']:
            if Path(path).name in ('four_modes_short.jsonl','four_modes_typical.jsonl','four_modes_long.jsonl'):
                verify_matched_benchmarks(read_jsonl(bound_input(spec,path)),release['checkpoint_sha256'],run_dir=bound_input(spec,path).parent)
        protocol=read_strict_json(calibration)
        if protocol['manifest_sha256']!=release['canonical_manifest_sha256']:raise ValueError('Frozen calibration provenance differs')
        write_new_json(run/'execution_context.json',{'scope':'validation evaluation only; saved scores/timings',
            'runtime':runtime,'input_sha256':spec['inputs'],'actual_input_bindings':{p:str(bound_input(spec,p)) for p in spec['inputs']},'C8A_context_sha256':spec['C8A_context_sha256'],
            'no_inference_or_benchmark_executed':True,'test_features_accessed':False,
            'evaluation_script_sha256':sha256_file(Path(__file__)),
            'research_reader_sha256':sha256_file(HERE/'src/avfuse/member1_fusion/dayc_research.py')})
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
            PYTHONPATH=str(root/'src')+os.pathsep+str(root),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
            OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',MPLBACKEND='Agg',
            MPLCONFIGDIR=str(run/'runtime_cache/matplotlib'),XDG_CACHE_HOME=str(run/'runtime_cache'))
        common=['--manifest',str(canonical),'--predictions',*map(str,predictions)]
        fixed=['--protocol',str(calibration),'--n-boot','1000','--seed','0']
        commands=[('prediction_checks',[str(root/'scripts/m2_check_predictions.py'),*common,'--n-boot','1000','--out',str(run/'reports/prediction_checks.json')]),
            ('metrics',[str(root/'scripts/m2_evaluate.py'),'evaluate',*common,*fixed,'--split','val','--out',str(run/'reports/four_mode_validation')]),
            ('paired',[str(root/'scripts/m2_paired.py'),*common,*fixed,'--benchmarks',str(typical),'--scope','cached_feature_model','--out',str(run/'reports/four_mode_paired')])]
        for name,args in commands:
            stage='member2_'+name;print('EVALUATION_STAGE='+stage,flush=True)
            write_new_json(run/(name+'_command.json'),{'command':[sys.executable,'-B',*args]})
            with (run/(name+'.log')).open('x') as log:
                subprocess.run([sys.executable,'-B',*args],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        stage='existing_owner_per_stratum_APIs'
        all_predictions=read_predictions(*predictions)
        paired_strata,extra_metrics=owner_stratum_statistics(whole,all_predictions,protocol['threshold'])
        engine={name:sha256_file(root/name) for name in
            ('src/avfuse/member2_data/paired.py','src/avfuse/member2_data/evaluate.py','src/avfuse/member2_data/splits.py')}
        method={'kind':'analysis_subsets_with_original_source_cluster_bootstrap','n_boot':1000,'seed':0,
            'analysis_strata':['overall','audio_fake','video_fake','both_fake'],
            'fixed_category_counts_per_resample':False,'engine_sha256':engine,
            'accepted_dayB_precedent':spec['accepted_method_precedent'],
            'disclosure':'Analysis-stratified fake-vs-real subsets; whole connected source/identity groups resampled with replacement. Category counts fluctuate within replicates. No bootstrap algorithm or random engine changed.'}
        write_new_json(run/'reports/per_stratum_owner_results.json',{'per_type_paired':paired_strata,'per_type_additional_metrics':extra_metrics,
            'resampling':method,'threshold':protocol['threshold'],'input_sha256':spec['inputs']})
        paired=read_strict_json(run/'reports/four_mode_paired/paired.json')
        finite_tree(paired)
        augmented=dict(paired,per_type_paired=paired_strata,resampling=method,
            original_member2_paired_sha256=sha256_file(run/'reports/four_mode_paired/paired.json'))
        write_new_json(run/'reports/paired_with_strata_binding.json',augmented)
        stage='member1_research_crosscheck'
        module_spec=importlib.util.spec_from_file_location('dayc_b2_research_reader',HERE/'src/avfuse/member1_fusion/dayc_research.py')
        module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
        metric_path=run/'reports/four_mode_validation/metrics.json'
        metrics=read_strict_json(metric_path);finite_tree(metrics)
        evidence=module.read_four_mode_validation_evidence(metric_json=metric_path,paired_json=run/'reports/paired_with_strata_binding.json',
            prediction_paths=dict(zip(MODES,predictions)),benchmark_jsonl=typical,canonical_manifest=canonical)
        table=module.build_h2_h3_decision_table(evidence)
        limits=module.describe_limitations(evidence)+[method['disclosure']]
        research=run/'research';research.mkdir()
        write_new_json(research/'day8_review_packet.json',{'status':'RESEARCH_REVIEW_REQUIRED','evidence':evidence,
            'decision_table':table,'limitations':limits,'policy_frozen':False,'C8B_approval_received':False})
        lines=['# Day C B2 full validation evidence','', 'Status: RESEARCH REVIEW REQUIRED; no policy selected.','',
            '| mode | AUROC | AUPRC | mean BCE | recall | real FPR |','|---|---:|---:|---:|---:|---:|']
        for mode in MODES:
            m=metrics['configs'][mode]
            lines.append('| '+mode+' | '+' | '.join(str(m[k]['value']) for k in ('auroc','auprc','bce','recall','real_fpr'))+' |')
        lines+=['','## Paired loss decision table','',json.dumps(table,indent=2),'','## Resampling disclosure','',method['disclosure'],
                '', 'This preserves the accepted Day B method; fixed-category-count resampling is not claimed. C8B lead review is required.',
                '', '## Limitations','']+['- '+s for s in limits]
        with (research/'day8_review_packet.md').open('x') as f:f.write('\n'.join(lines)+'\n')
        if spec.get('prepare_B3_source_draft') is True:
            stage='B3_source_proposal_preparation'
            print('B3_STAGE=validation-only decision and blocked freeze proposal; no test access',flush=True)
            freeze_spec=importlib.util.spec_from_file_location('dayc_b3_proposal_builder',HERE/'src/avfuse/member1_fusion/dayc_freeze.py')
            freeze_module=importlib.util.module_from_spec(freeze_spec);freeze_spec.loader.exec_module(freeze_module)
            preparation=read_strict_json(HERE/'assets/B3_preparation_authorization.json')
            freeze_module.check_preparation_authorization(preparation,spec['C8A_context_sha256'])
            packet=read_strict_json(research/'day8_review_packet.json')
            findings=module.validate_day9_decision(packet)
            registry={}
            for mode,path in zip(MODES,predictions):
                registry['predictions_'+mode]={'path':str(path.relative_to(root)),'sha256':sha256_file(path)}
            named={'calibration':calibration,'canonical_manifest':canonical,'b2_research_packet':research/'day8_review_packet.json',
                'metrics':metric_path,'paired':run/'reports/paired_with_strata_binding.json',
                'per_type':run/'reports/per_stratum_owner_results.json'}
            for name,path in named.items():registry[name]={'path':str(path.relative_to(root)),'sha256':sha256_file(path)}
            for role in ('short','typical','long'):
                path=next(bound_input(spec,p) for p in spec['inputs'] if p.endswith('/four_modes_'+role+'.jsonl'))
                registry['benchmark_'+role]={'path':str(path.relative_to(root)),'sha256':sha256_file(path)}
            # The pure draft is deliberately unapprovable until actual owner releases/identities and lead decisions exist.
            draft=freeze_module.build_draft(context=context,registry=registry,c8b_accepted=False)
            draft['blockers']=[b for b in draft['blockers'] if b!='Actual B2 four-mode statistics/research packet not returned']
            draft.update(H2=findings['H2'],H3=findings['H3'],decision_table=table,
                numeric_damage_and_ci={'D':table['damage'],'CI':table['damage_ci']},
                restoration_direct_contrast_and_ci={'H3_direct':table['H3_direct'],'CI':table['H3_ci']},
                measured_costs=evidence['costs'],fixed_threshold=protocol['threshold'],threshold_final_acknowledged=False)
            day9=output.parent/'day9'/(run.name+'_PROPOSAL');day9.mkdir(parents=True,exist_ok=False)
            proposal_sha=write_new_json(day9/'policy_proposal.json',draft)
            write_new_json(day9/'freeze_dependency_check.json',{'status':'BLOCKED_PENDING_REAL_RELEASES_AND_LEAD_REVIEW',
                'blockers':draft['blockers'],'test_rows_read':False,'test_features_read':False,'freeze_written':False})
            write_new_json(day9/'proposed_hash_registry.json',registry)
            with (day9/'policy_proposal.md').open('x') as stream:
                stream.write('# Unapproved B3 policy proposal\n\n'+json.dumps(findings,indent=2)+'\n\nNo policy selected, no freeze, no held-out access.\n\nProposal SHA256: '+proposal_sha+'\n')
            write_new_json(day9/'source_preparation_status.json',{'status':'B3_SOURCE_IMPLEMENTED_PENDING_COMBINED_REVIEW',
                'source_verification':read_strict_json(HERE/'assets/B3_source_verification.json'),
                'member2_final_release_supplied':False,'member3_final_release_supplied':False,
                'C8B_ack_fabricated':False,'C9_freeze_written':False,'C10_attempt_created':False})
            additional_artifacts={'day9_preparation/'+p.name:p for p in day9.iterdir() if p.is_file()}
            write_new_json(run/'day9_preparation_reference.json',{'actual_runtime_path':str(day9),
                'files':{name:sha256_file(path) for name,path in additional_artifacts.items()},
                'status':'UNAPPROVED_DRAFT_ONLY'})
        for path,expected in spec['inputs'].items():require_sha256(bound_input(spec,path),expected)
        verify_package(root)
        write_new_json(run/'evaluation_completion.json',{'status':'B2_EVIDENCE_COMPLETE_PENDING_C8B_REVIEW','n':1200,
            'n_boot':1000,'seed':0,'threshold':protocol['threshold'],'test_features_accessed':False,
            'model_inference_executed':False,'benchmark_executed':False,'owner_source_unchanged':True,
            'policy_frozen':False,'C8B_approval_received':False,'method_disclosure':method['disclosure']})
    except BaseException as exc:
        write_new_json(run/'failure.json',{'stage':stage,'status':'FAILED_PRESERVE_NO_RETRY','error':f'{type(exc).__name__}: {exc}',
            'model_inference_executed':False,'benchmark_executed':False,'test_features_accessed':False})
        print('STOP:',type(exc).__name__,str(exc),flush=True)
    files=sorted(p for p in run.rglob('*') if p.is_file())
    hashes={str(p.relative_to(run)):sha256_file(p) for p in files}
    hashes.update({name:sha256_file(path) for name,path in additional_artifacts.items()})
    write_new_json(run/'output_hashes.json',hashes)
    bundle=run/('DAYC_B2_B3_review.zip' if spec.get('prepare_B3_source_draft') is True else 'DAYC_B2_evaluation_review.zip')
    with zipfile.ZipFile(bundle,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for path in files+[run/'output_hashes.json']:z.write(path,path.relative_to(run))
        for name,path in additional_artifacts.items():z.write(path,name)
    print('RETURN REVIEW ZIP: '+str(bundle),flush=True)
    print('STOP: return for joint B2/B3 review. No final freeze or held-out execution.' if spec.get('prepare_B3_source_draft') is True else 'STOP: return for B2 review; no policy freeze or held-out execution.',flush=True)
    if (run/'failure.json').exists():raise SystemExit(1)


if __name__=='__main__':main()
