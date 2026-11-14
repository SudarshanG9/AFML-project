"""Fail closed on cohort drift and heterogeneous restoration-cost evidence."""
import copy
import pytest
from scripts.dayB_finish.run_g8_g9 import CHECKPOINT, MODES, check_cohort, check_matched, finite_tree


def test_full_restored_cohort_contract():
    rows = [{'run_id':'unit','clip_id':str(i),'split':'val','config':'restore_value_output',
             'score_fake':0.5,'checkpoint_sha256':CHECKPOINT} for i in range(1200)]
    ids = {str(i) for i in range(1200)}
    check_cohort(rows,ids,'restore_value_output')
    rows[-1]['clip_id'] = 'outside'
    with pytest.raises(RuntimeError,match='Cohort'):
        check_cohort(rows,ids,'restore_value_output')


def records():
    # The existing schema/raw-statistic validator is tested independently. These
    # minimal unit fixtures isolate the extra cross-mode compatibility guard.
    return [{'config':mode,'workload':{'checkpoint_sha256':CHECKPOINT,'input_identity_sha256':'same',
                'input_shapes':{'video_tokens':[192,768]}},'device':'cpu','batch_size':1,
             'backend':'sdpa','environment':{'torch_version':'unit'},'warmup':10,'repeats':50,
             'scope':'cached_feature_model','experiment_id':'unit','run_id':'unit',
             'timing_samples_ms':[1.0]*50,'peak_memory_bytes':None,'memory_method':'not_measured',
             'precision':{'execution_dtype':'float32','real_int8_kernel':False},
             'conversion_overhead_included_in_latency':mode!='reference'} for mode in MODES]


@pytest.fixture
def schema_stub(monkeypatch):
    monkeypatch.setattr('avfuse.member3_runtime.artifacts.validate_benchmark_records',lambda rows:None)


def test_compatible_three_mode_cost_set(schema_stub):
    check_matched(records())


@pytest.mark.parametrize('change',['shape','identity','device','dtype','duplicate_mode'])
def test_mixed_cost_set_is_rejected(schema_stub,change):
    rows = copy.deepcopy(records())
    if change=='shape':rows[-1]['workload']['input_shapes']['video_tokens']=[493,768]
    elif change=='identity':rows[-1]['workload']['input_identity_sha256']='different'
    elif change=='device':rows[-1]['device']='cuda:0'
    elif change=='dtype':rows[-1]['precision']['execution_dtype']='float16'
    elif change=='duplicate_mode':rows[-1]['config']='reference'
    with pytest.raises(RuntimeError):
        check_matched(rows)


def test_undefined_metrics_cannot_be_marked_ready():
    with pytest.raises(RuntimeError,match='Undefined'):
        finite_tree({'ci_low':float('nan')})
