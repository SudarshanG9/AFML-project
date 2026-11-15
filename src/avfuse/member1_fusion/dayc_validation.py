"""Full-cohort guards; delegates schemas, grouping and model execution to owners."""
from collections import Counter
import math
from avfuse.member2_data.predictions import validate_predictions
from avfuse.member2_data.splits import split_audit
from scripts.dayC_member1.colab_query_pilot import validate_cohort

MODES = ('reference','low_precision','restore_value_output','restore_query')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_validation_metadata(canonical, ready):
    require(split_audit(canonical)['passed'], 'Cross-split source or identity leakage')
    validation = [r for r in canonical if r['split'] == 'val']
    validate_cohort(validation, ready)
    require(len({r['clip_id'] for r in canonical}) == len(canonical), 'Duplicate canonical ID')
    return {r['clip_id'] for r in validation}


def verify_prediction_rows(rows, mode, ids, checkpoint):
    validate_predictions(rows)
    require(len(rows) == len(ids) == 1200, 'Exactly 1200 predictions required')
    require(len({r['clip_id'] for r in rows}) == 1200, 'Duplicate prediction ID')
    require({r['clip_id'] for r in rows} == set(ids), 'Prediction cohort differs')
    require({r['config'] for r in rows} == {mode}, 'Wrong prediction mode')
    require({r['split'] for r in rows} == {'val'}, 'Non-validation prediction')
    require({r['checkpoint_sha256'] for r in rows} == {checkpoint}, 'Wrong checkpoint')
    require(len({r['run_id'] for r in rows}) == 1, 'Mixed run IDs')
    require(all(isinstance(r['score_fake'], (int,float)) and not isinstance(r['score_fake'],bool)
                and math.isfinite(r['score_fake']) and 0 <= r['score_fake'] <= 1 for r in rows),
            'Nonfinite or invalid probability')
    return {'count':1200,'mode':mode,'split':'val','checkpoint_sha256':checkpoint}


def verify_matched_benchmarks(rows, checkpoint, run_dir=None):
    from avfuse.member3_runtime.artifacts import validate_benchmark_records
    validate_benchmark_records(rows, run_dir=run_dir)
    require(len(rows) == 4 and {r['config'] for r in rows} == set(MODES), 'Four unique benchmark modes required')
    fields = ('device','backend','batch_size','warmup','repeats','workload','environment','measurement')
    base = rows[0]
    require(all(all(r[k] == base[k] for k in fields) for r in rows), 'Mixed benchmark workload/runtime/protocol')
    require(len({r['run_id'] for r in rows}) == len({r['experiment_id'] for r in rows}) == 1, 'Mixed benchmark experiment')
    require(base['device'] == 'cpu' and base['backend'] == 'sdpa' and base['batch_size'] == 1,
            'Unapproved device/backend/batch')
    require(base['warmup'] == 10 and base['repeats'] == 50, 'Benchmark protocol reduced')
    require(base['workload']['checkpoint_sha256'] == checkpoint and base['workload']['input_identity_sha256'],
            'Missing/wrong benchmark input/checkpoint identity')
    require(base['scope'] == 'cached_feature_model' and all(r['scope'] == base['scope'] for r in rows), 'Mixed benchmark scope')
    for r in rows:
        require(r['precision']['execution_dtype'] == 'float32' and r['precision']['real_int8_kernel'] is False, 'Precision semantics differ')
        require(r['peak_memory_bytes'] is None and r['memory_method'] == 'not_measured', 'Unsupported CPU memory claim')
        if r['config'] != 'reference':
            require(r['conversion_overhead_included_in_latency'] is True, 'Missing precision overhead')
    times = {r['config']:r['p50_ms'] for r in rows}
    return {'status':'PASS','p50_ms':times,
            'E_value_ms':times['restore_value_output']-times['low_precision'],
            'E_query_ms':times['restore_query']-times['low_precision'],
            'scope':'descriptive measured differences; no automatic equal-cost conclusion'}
