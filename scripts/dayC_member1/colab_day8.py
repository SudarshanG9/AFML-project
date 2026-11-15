"""Manual B2: one full query validation plus three unchanged four-mode benchmarks."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import uuid
import zipfile
from avfuse.member1_fusion.dayc_guards import read_strict_json,write_new_json,sha256_file,require_sha256
from avfuse.member1_fusion.dayc_cache_mapping import ValidationCacheResolver
from avfuse.member1_fusion.dayc_validation import MODES,verify_matched_benchmarks
from scripts.dayC_member1.colab_query_pilot import verify_package
from scripts.dayC_member1.run_day8 import execute,load_release,launch_checked,read_rows


def main():
    root=Path(__file__).resolve().parents[2]
    verify_package(root)
    approval=root/'assets/c8a_approval.json'
    context,release=load_release(root,approval)
    output=root/'artifacts/dayC/member1/day8'
    output.mkdir(parents=True,exist_ok=True)
    envelope=output/('B2_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    envelope.mkdir(exist_ok=False)
    stage='full_validation_preflight'
    try:
        print('B2_RUN_DIR='+str(envelope),flush=True)
        print('Verifying exact 1200-cohort cache; then one restore_query validation attempt.',flush=True)
        validation=execute(root,approval,Path(release['ready_manifest']),Path(release['canonical_manifest']),
                           Path(release['upstream']),root/'artifacts/predictions/restore_query_val.jsonl',envelope/'validation')
        print('QUERY_VALIDATION_COMPLETE=1200/1200',flush=True)
        mapping=release['validation_cache_mapping']
        resolver=ValidationCacheResolver(mapping['original_root'],mapping['mapped_root'])
        ready={r['clip_id']:r for r in read_rows(release['ready_manifest'])}
        context_g5=read_strict_json(release['g5_context'])
        expected_roles={'short','typical_anchor','long'}
        workloads=context_g5['approved_workloads']
        if len(workloads)!=3 or {w['role'] for w in workloads}!=expected_roles:
            raise ValueError('Original approved short/typical/long workload set changed')
        benchdir=envelope/'benchmarks';benchdir.mkdir()
        cost={}
        for workload in workloads:
            role=workload['role'];stage='matched_four_mode_benchmark_'+role
            row=ready[workload['clip_id']]
            if row['feature_path']!=workload['feature_path']:raise ValueError('Benchmark cohort identity mismatch')
            feature=resolver.resolve(row);require_sha256(feature,workload['feature_sha256'])
            filename_role='typical' if role=='typical_anchor' else role
            path=benchdir/('four_modes_'+filename_role+'.jsonl')
            command=[sys.executable,'-B',str(root/'scripts/dayC_member1/day8_child.py'),'benchmark',
                '--approval-file',str(approval),'--','--features',str(feature),'--upstream',release['upstream'],
                '--device','cpu','--modes',*MODES,'--warmup','10','--repeats','50',
                '--run-id',envelope.name+'_'+role,'--experiment-id','dayC-member1-day8-matched-val-'+role,'--out',str(path)]
            write_new_json(benchdir/(role+'_attempt_started.json'),{'command':command,'workload':workload})
            print('BENCHMARK_START='+role+'; four modes; warmup=10 repeats=50 each',flush=True)
            launch_checked(command,benchdir/(role+'.log'),root)
            records=read_rows(path)
            cost[role]=verify_matched_benchmarks(records,release['checkpoint_sha256'],run_dir=benchdir)
            shapes=records[0]['workload']['input_shapes']
            if not (shapes['video_tokens'][-2]==workload['video_tokens']
                    and shapes['audio_tokens'][-2]==workload['audio_tokens']):
                raise ValueError('Benchmark input shapes differ from approved workload')
            require_sha256(feature,workload['feature_sha256'])
            print('BENCHMARK_COMPLETE='+role,flush=True)
        # Preserve primary typical separately. M2/3 review supplies timing reliability.
        write_new_json(envelope/'matched_costs.json',{'status':'MEASURED_PENDING_MEMBER3_REVIEW','workloads':cost,
            'primary':'typical_anchor','cost_normalization_reliability_approved':False,
            'timing_uncertainty_included_in_BCE_CIs':False})
        for path,expected in read_strict_json(validation/'input_hashes.json').items():
            require_sha256(Path(path),expected)
        verify_package(root)
        write_new_json(envelope/'batch_status.json',{'status':'C8_3_COMPLETE_C8_4_MEASURED_PENDING_REVIEW_C8_5_BLOCKED',
            'validation_count':1200,'benchmarks':12,'test_access':False,
            'evaluation_blocker':'Member2 explicit stratified source-group resampling and per-stratum paired contrasts require owned handoff; unchanged evaluator does not provide these complete requirements.',
            'C8B_ready':False,'policy_frozen':False})
    except BaseException as exc:
        write_new_json(envelope/'failure.json',{'status':'FAILED_PRESERVE_NO_RETRY','stage':stage,
            'error':f'{type(exc).__name__}: {exc}','test_access':False})
        print('STOP:',type(exc).__name__,str(exc),flush=True)
    paths=sorted(p for p in envelope.rglob('*') if p.is_file())
    prediction=root/'artifacts/predictions/restore_query_val.jsonl'
    hashes={str(p.relative_to(envelope)):sha256_file(p) for p in paths}
    if prediction.exists():hashes['predictions/restore_query_val.jsonl']=sha256_file(prediction)
    write_new_json(envelope/'output_hashes.json',hashes)
    bundle=envelope/'DAYC_B2_review.zip'
    with zipfile.ZipFile(bundle,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for path in paths+[envelope/'output_hashes.json']:z.write(path,path.relative_to(envelope))
        if prediction.exists():z.write(prediction,'predictions/restore_query_val.jsonl')
    print('RETURN REVIEW ZIP: '+str(bundle),flush=True)
    print('STOP B2 evidence review; no Day9/10 or test work.',flush=True)
    if (envelope/'failure.json').exists():raise SystemExit(1)


if __name__=='__main__':main()
