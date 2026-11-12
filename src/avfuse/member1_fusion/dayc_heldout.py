"""One-way held-out execution and saved-output checks. No metric calculation."""
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import re
from avfuse.member1_fusion.dayc_guards import (
    ensure_inside_repo,require_new_file,sha256_file,require_sha256,read_strict_json,write_new_json,write_new_jsonl)
from avfuse.member1_fusion.dayc_freeze import (
    CHECKPOINT,PURPOSE,require,verify_release,verify_registry)
from avfuse.member1_fusion.dayc_validation import MODES


def safe_run_id(value):
    require(isinstance(value,str) and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,119}',value)
            and value not in {'.','..'},'Unsafe held-out run ID')
    return value


def append_event(path,event):
    event=dict(event,timestamp_utc=datetime.now(timezone.utc).isoformat())
    with Path(path).open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(event,sort_keys=True,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())


def claim_attempt(root,run_id,hashes):
    safe_run_id(run_id);day10=Path(root)/'artifacts/dayC/member1/day10';day10.mkdir(parents=True,exist_ok=True)
    lock=day10/'test_attempt.lock';ledger=day10/'test_attempt_ledger.jsonl'
    require_new_file(lock);require_new_file(ledger)
    with lock.open('x') as stream:
        stream.write(run_id+'\n');stream.flush();os.fsync(stream.fileno())
    # The persistent lock remains even if ledger creation or the process fails.
    with ledger.open('x') as stream:stream.flush();os.fsync(stream.fileno())
    event={'event':'TEST_ATTEMPT_OPENED','status':'STARTED','split':'test','purpose':PURPOSE,
           'run_id':run_id,'modes':list(MODES),**hashes}
    append_event(ledger,event)
    return ledger,event


def reconcile_test_metadata(canonical,ready,expected_count):
    # Called only after explicit C10 and immutable attempt claim.
    from avfuse.member2_data.splits import split_audit
    require(split_audit(canonical)['passed'],'Cross-split source/identity leakage')
    require(len({r['clip_id'] for r in canonical})==len(canonical),'Duplicate canonical ID')
    expected={r['clip_id']:r for r in canonical if r['split']=='test'}
    require(type(expected_count) is int and expected_count>0 and len(expected)==len(ready)==expected_count,
            'Full signed held-out cohort required; no subset/drops')
    require(all(r['split']=='test' for r in ready),'Ready manifest must contain only test rows')
    require(len({r['clip_id'] for r in ready})==expected_count and {r['clip_id'] for r in ready}==set(expected),
            'Ready held-out IDs differ')
    for row in ready:
        require({k:v for k,v in row.items() if k!='feature_path'}==
                {k:v for k,v in expected[row['clip_id']].items() if k!='feature_path'},'Ready held-out metadata differs')
    return sorted(ready,key=lambda r:r['clip_id'])


def resolve_test_feature(root,row):
    require(row['split']=='test','Non-test row in held-out runner')
    require(isinstance(row['clip_id'],str) and re.fullmatch('[A-Za-z0-9_-]+',row['clip_id']),'Unsafe clip identity')
    requested=Path(row['feature_path']);raw=requested if requested.is_absolute() else Path(root)/requested
    require(not raw.is_symlink(),'Symlink feature file')
    path=ensure_inside_repo(root,raw,must_exist=True)
    require(path.is_file() and path.suffix in {'.npz','.pt'},'Missing/unsupported cached feature')
    require(row['clip_id'] in path.parts or path.stem==row['clip_id'],'Cache path not bound to clip ID')
    # Reject parent symlinks even if they happen to resolve inside the project.
    parent=raw.parent
    while parent!=Path(root):
        require(not parent.is_symlink(),'Symlink cache parent')
        require(parent!=parent.parent,'Feature path outside project');parent=parent.parent
    return path


def validate_saved_rows(rows,*,mode,ids,checkpoint,run_id):
    from avfuse.member2_data.predictions import validate_predictions
    validate_predictions(rows)
    require(len(rows)==len(ids)>0 and len({r['clip_id'] for r in rows})==len(ids)
        and {r['clip_id'] for r in rows}==set(ids),'Incomplete/duplicate held-out predictions')
    allowed={'run_id','clip_id','split','config','score_fake','checkpoint_sha256'}
    for row in rows:
        require(set(row)==allowed and row['split']=='test' and row['config']==mode and row['run_id']==run_id
            and row['checkpoint_sha256']==checkpoint,'Saved prediction identity/schema differs')
        score=row['score_fake'];require(not isinstance(score,bool) and math.isfinite(score) and 0<=score<=1,'Invalid saved probability')


def score_complete_cohort(rows,adapter,load_features,resolve_feature,run_id,progress=None):
    """Internal engine; production caller must authorize/claim first. Fixtures inject here, never via CLI."""
    import torch
    require(len({r['clip_id'] for r in rows})==len(rows)>0,'Duplicate/empty cohort')
    outputs={mode:[] for mode in MODES};identities=[]
    with torch.inference_mode():
        for index,row in enumerate(rows,start=1):
            path=resolve_feature(row);before=sha256_file(path)
            features=load_features(path);snapshot={key:t.detach().clone() for key,t in features.items()}
            if progress:progress(index,{m:len(values) for m,values in outputs.items()})
            for mode in MODES:
                score_tensor=adapter.predict_from_cached_features(features,mode)
                require(score_tensor.numel()==1,'Expected one original fake probability')
                score=float(score_tensor.reshape(-1)[0].item())
                require(math.isfinite(score) and 0<=score<=1,'Nonfinite/out-of-range held-out probability')
                require(set(features)==set(snapshot) and all(torch.equal(features[k],t) for k,t in snapshot.items()),'Input tensors modified')
                outputs[mode].append({'run_id':run_id,'clip_id':row['clip_id'],'split':'test','config':mode,
                    'score_fake':score,'checkpoint_sha256':adapter.checkpoint_sha256})
                if progress:progress(index,{m:len(values) for m,values in outputs.items()})
            require(sha256_file(path)==before,'Cached feature modified during scoring')
            identities.append({'clip_id':row['clip_id'],'feature_sha256':before})
            if progress:progress(index,{m:len(values) for m,values in outputs.items()})
    ids={r['clip_id'] for r in rows}
    for mode,values in outputs.items():validate_saved_rows(values,mode=mode,ids=ids,checkpoint=adapter.checkpoint_sha256,run_id=run_id)
    return outputs,identities


def publish_complete_predictions(out_dir,outputs,*,ids,checkpoint,run_id):
    # A directory rename publishes all four final names together after validation/staging.
    out_dir=Path(out_dir);target=out_dir/'predictions';stage=out_dir/'.predictions_staging'
    require(set(outputs)==set(MODES),'All four modes required before publication')
    for mode,rows in outputs.items():validate_saved_rows(rows,mode=mode,ids=ids,checkpoint=checkpoint,run_id=run_id)
    require_new_file(target);require_new_file(stage);stage.mkdir()
    for mode in MODES:write_new_jsonl(stage/(mode+'_test.jsonl'),outputs[mode])
    os.rename(stage,target)
    return {mode:sha256_file(target/(mode+'_test.jsonl')) for mode in MODES}


def execute_heldout(args):
    root=args.repo_root.resolve(strict=True);run_id=safe_run_id(args.run_id)
    out=ensure_inside_repo(root,args.out_dir)
    require(out==root/'artifacts/dayC/member1/day10'/run_id,'Output must match the one new run ID')
    require_new_file(out)
    release=verify_release(root,args.frozen_policy,args.frozen_run,args.freeze_anchor,args.approval_file,
                           args.member2_release,args.member3_release)
    frozen=release['run'];registry=frozen['inputs']
    require(args.device==frozen['device'],'Device differs from freeze')
    for requested,key in ((args.canonical_manifest,'canonical_manifest'),(args.ready_test_manifest,'ready_test_manifest')):
        require(ensure_inside_repo(root,requested,must_exist=True)==ensure_inside_repo(root,registry[key]['path'],must_exist=True),
                'Manifest path differs from freeze')
    upstream=ensure_inside_repo(root,args.upstream,must_exist=True)
    require(upstream==ensure_inside_repo(root,registry['model_source']['path'],must_exist=True).parent.parent,
            'Upstream root differs from frozen source')
    from scripts.dayC_member1.run_day8 import observed_runtime
    from scripts.dayC_member1.colab_query_pilot import check_environment
    runtime=observed_runtime();check_environment(runtime,frozen['runtime'])
    # Nothing above enumerates test IDs or opens cached tensors.
    ledger,event=claim_attempt(root,run_id,release['hashes']);counts=dict.fromkeys(MODES,0)
    try:
        out.mkdir(parents=True,exist_ok=False)
        with (out/'command.log').open('x') as stream:stream.write(json.dumps({'purpose':PURPOSE,'run_id':run_id,'modes':list(MODES)})+'\n')
        write_new_json(out/'attempt_opened.json',event)
        write_new_json(out/'release_check.json',{k:v for k,v in release.items() if k not in ('policy','run')})
        from avfuse.member2_data.manifest import read_manifest
        canonical=read_manifest(args.canonical_manifest);ready=read_manifest(args.ready_test_manifest)
        rows=reconcile_test_metadata(canonical,ready,frozen['expected_count'])
        for row in rows:resolve_test_feature(root,row)
        verify_registry(root,registry)
        import torch
        torch.manual_seed(42)
        from scripts.dayB_inference import create_adapter,load_features
        from avfuse.member1_fusion.dayc_precision_audit import check_model_invariants
        from types import SimpleNamespace
        adapter=create_adapter(SimpleNamespace(upstream=str(upstream),device=args.device))
        require(adapter.checkpoint_path.resolve()==ensure_inside_repo(root,registry['checkpoint']['path'],must_exist=True),
                'Adapter checkpoint path differs from freeze')
        require(adapter.config_path.resolve()==ensure_inside_repo(root,registry['model_config']['path'],must_exist=True),
                'Adapter configuration path differs from freeze')
        before=check_model_invariants(adapter,checkpoint_sha256=CHECKPOINT)
        write_new_json(out/'execution_context.json',{'runtime':runtime,'checkpoint_sha256':CHECKPOINT,
            'frozen_hashes':release['hashes'],'scope':'one full frozen four-mode inference; no labeled evaluation',
            'known_test_label_summary_access':True,'initial_state_dict_sha256':before['state_dict_sha256']})
        def progress(index,completed):
            counts.update(completed)
            with (out/'progress.log').open('a') as stream:stream.write(json.dumps({'current_clip_index':index,'mode_counts':completed})+'\n')
        outputs,identities=score_complete_cohort(rows,adapter,load_features,lambda row:resolve_test_feature(root,row),run_id,progress)
        after=check_model_invariants(adapter,checkpoint_sha256=CHECKPOINT)
        require(before['state_dict_sha256']==after['state_dict_sha256'],'Frozen model state changed')
        verify_registry(root,registry)
        hashes=publish_complete_predictions(out,outputs,ids={r['clip_id'] for r in rows},checkpoint=CHECKPOINT,run_id=run_id)
        verify_registry(root,registry)
        reports=out/'reports';reports.mkdir()
        for name,record in {
            'predictions_schema_validation.json':{'status':'PASS','four_modes':list(MODES)},
            'exact_cohort_coverage.json':{'status':'PASS','count':len(rows),'identical_ids_all_modes':True},
            'frozen_input_integrity.json':{'status':'PASS','registry':registry,'state_unchanged':True,'state_dict_before_sha256':before['state_dict_sha256'],'state_dict_after_sha256':after['state_dict_sha256']},
            'full_four_mode_completion.json':{'status':'COMPLETE','count_per_mode':len(rows),'prediction_sha256':hashes}}.items():
            write_new_json(reports/name,record)
        provenance=out/'provenance';provenance.mkdir()
        write_new_json(provenance/'manifest_and_cache_identity.json',{'features':identities,'manifests_byte_hashes':{
            k:registry[k]['sha256'] for k in ('canonical_manifest','ready_test_manifest')},'feature_loads':len(rows)})
        write_new_json(provenance/'output_hashes.json',hashes)
        for path,item in ((out/'provenance/model_checkpoint_sha256.json',{'sha256':CHECKPOINT}),
                          (out/'provenance/model_code_sha256.json',{k:v for k,v in registry.items() if k.startswith('source:')}),
                          (out/'provenance/freeze_and_approval_hashes.json',release['hashes'])):write_new_json(path,item)
        completed={'event':'TEST_ATTEMPT_COMPLETED','status':'COMPLETE','run_id':run_id,'mode_counts':counts,
                   'prediction_sha256':hashes,'no_retraining_or_recalibration':True}
        write_new_json(out/'attempt_completed.json',completed)
        append_event(ledger,completed)
        inventory={str(p.relative_to(out)):sha256_file(p) for p in out.rglob('*') if p.is_file()}
        write_new_json(out/'run_manifest.json',{'status':'COMPLETE','files':inventory,'run_id':run_id})
        return out
    except BaseException as exc:
        failure={'event':'TEST_ATTEMPT_FAILED','status':'FAILED','run_id':run_id,'mode_counts':counts,
                 'error':f'{type(exc).__name__}: {exc}','automatic_retry_permitted':False}
        if out.is_dir() and not (out/'attempt_failed.json').exists():write_new_json(out/'attempt_failed.json',failure)
        append_event(ledger,failure);raise


def audit_saved_outputs(root,run_dir,frozen_policy,frozen_run,freeze_anchor,approval_file):
    root=Path(root).resolve(strict=True);run_dir=ensure_inside_repo(root,run_dir,must_exist=True)
    meta=verify_release(root,frozen_policy,frozen_run,freeze_anchor,approval_file,dry_run=True)
    require(meta['status']=='AUTHORIZED_METADATA_ONLY','Audit requires actual C10')
    frozen=read_strict_json(ensure_inside_repo(root,frozen_run,must_exist=True));verify_registry(root,frozen['inputs'])
    ledger=root/'artifacts/dayC/member1/day10/test_attempt_ledger.jsonl'
    entries=[json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
    require(len(entries)==2 and [r['event'] for r in entries]==['TEST_ATTEMPT_OPENED','TEST_ATTEMPT_COMPLETED'],
            'Ledger is not exactly one completed attempt')
    require({r['run_id'] for r in entries}=={run_dir.name},'Ledger/run mismatch')
    require((ledger.parent/'test_attempt.lock').read_text().strip()==run_dir.name,'Attempt lock/run mismatch')
    for name,path in (('frozen_policy_sha256',frozen_policy),('frozen_run_sha256',frozen_run),
                      ('freeze_anchor_sha256',freeze_anchor),('approved_c10_ack_sha256',approval_file)):
        require(entries[0][name]==sha256_file(ensure_inside_repo(root,path,must_exist=True)),'Ledger frozen context mismatch')
    require(entries[1]['mode_counts']==dict.fromkeys(MODES,frozen['expected_count']),'Incomplete ledger counts')
    require(not (run_dir/'attempt_failed.json').exists(),'Failed attempt cannot be called complete')
    require(read_strict_json(run_dir/'attempt_completed.json')['status']=='COMPLETE','Missing completion marker')
    from avfuse.member2_data.manifest import read_manifest
    from avfuse.member2_data.predictions import read_predictions
    registry=frozen['inputs']
    ready=reconcile_test_metadata(read_manifest(root/registry['canonical_manifest']['path']),
        read_manifest(root/registry['ready_test_manifest']['path']),frozen['expected_count'])
    ids={r['clip_id'] for r in ready};preds=run_dir/'predictions'
    require({p.name for p in preds.iterdir()}=={m+'_test.jsonl' for m in MODES},'Prediction file set differs')
    cached=read_strict_json(run_dir/'provenance/manifest_and_cache_identity.json')['features']
    require(len(cached)==len(ids) and {r['clip_id'] for r in cached}==ids,'Cached identity inventory differs')
    expected_cache={r['clip_id']:r['feature_sha256'] for r in cached}
    for row in ready:require_sha256(resolve_test_feature(root,row),expected_cache[row['clip_id']])
    expected=read_strict_json(run_dir/'provenance/output_hashes.json');hashes={} 
    for mode in MODES:
        path=preds/(mode+'_test.jsonl');rows=read_predictions(path)
        validate_saved_rows(rows,mode=mode,ids=ids,checkpoint=CHECKPOINT,run_id=run_dir.name)
        hashes[mode]=sha256_file(path);require(hashes[mode]==expected[mode]==entries[1]['prediction_sha256'][mode],'Output hash differs')
    inventory=read_strict_json(run_dir/'run_manifest.json')
    for path,expected_sha in inventory['files'].items():require_sha256(ensure_inside_repo(run_dir,path,must_exist=True),expected_sha)
    return {'status':'READY_FOR_MEMBER2_SCORING','run_id':run_dir.name,'count_per_mode':len(ids),
        'prediction_sha256':hashes,'attempts_opened':1,'attempts_completed':1,'test_metrics_computed':False,
        'known_test_label_summary_access':True,'frozen_input_integrity':'PASS'}
