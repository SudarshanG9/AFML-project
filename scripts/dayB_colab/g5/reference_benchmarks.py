"""Reference-only CPU cached-model timings on three approved validation inputs."""
import argparse
from pathlib import Path
from common import ROOT, CHECKPOINT, context, load, save, run_logged, review_bundle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    args = p.parse_args()
    run, c = context(args.run_dir)
    assert load(run / 'runtime_parity.json')['status'] == 'PASS'
    from avfuse.member3_runtime.artifacts import validate_benchmark_records
    from avfuse.shared.jsonl import read_jsonl
    private = []
    for item in c['approved_workloads']:
        role = item['role']
        name = 'dayB_model_reference.jsonl' if role == 'typical_anchor' else 'dayB_model_reference_' + role + '.jsonl'
        target = run / 'benchmarks' / name
        run_logged(run, 'benchmark_' + role, [ROOT / 'scripts/dayB_model_benchmark.py',
            '--features',item['feature_path'],'--upstream',c['upstream'],'--device','cpu',
            '--modes','reference','--warmup','10','--repeats','50',
            '--run-id',run.name + '-reference-' + role,
            '--experiment-id',('dayB-matched-validation-model' if role == 'typical_anchor'
                                else 'dayB-matched-validation-model-' + role),'--out',target], [target])
        records = read_jsonl(target)
        validate_benchmark_records(records)
        assert len(records) == 1
        record = records[0]
        assert record['config'] == 'reference' and record['device'] == 'cpu'
        assert record['warmup'] == 10 and record['repeats'] == len(record['timing_samples_ms']) == 50
        assert record['scope'] == 'cached_feature_model' and record['backend'] == 'sdpa'
        assert record['workload']['checkpoint_sha256'] == CHECKPOINT
        assert record['workload']['input_boundary'] == 'cached_encoder_tokens'
        assert record['peak_memory_bytes'] is None and record['memory_method'] == 'not_measured'
        assert record['precision']['real_int8_kernel'] is False
        private.append({'role':role,'clip_id':item['clip_id'], 'valid_tokens':item['valid_tokens'],
            'source_npz_sha256':item['feature_sha256'], 'feature_path':item['feature_path'],
            'benchmark_record':str(target),'p50_ms':record['p50_ms'],'p95_ms':record['p95_ms'],
            'directory_content_identity_sha256':record['workload']['input_identity_sha256'],
            'used_for_future_cost_anchor':role == 'typical_anchor'})
    save(run / 'private_benchmark_workloads.json', private)
    save(run / 'benchmark_stage.json', {'status':'VALIDATED_RECORDS_AWAITING_G5_REVIEW',
        'device':'cpu','matched_cost_anchor':'7cea6ad75b4ebf41',
        'model_scope':'Cached-feature preparation/padding + detector; excludes disk, decode and host/device transfer',
        'memory':'CPU peak memory not measured', 'test_feature_access':False,
        'next_action':'STOP AT G5; return evidence; no G6 authorized'})
    review_bundle(run, 'reference_benchmarks')

if __name__ == '__main__':
    main()
