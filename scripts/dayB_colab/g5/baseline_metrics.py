"""G5 provisional reference-validation calibration and baseline metrics only."""
import argparse
import math
from pathlib import Path
from common import ROOT, WHOLE, CHECKPOINT, context, digest, load, save, run_logged, review_bundle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    args = p.parse_args()
    run, c = context(args.run_dir)
    check = load(run / 'reports/dayB_reference_prediction_check.json')
    assert check['configs']['reference']['val']['orientation'] == 'ok', 'Stop for scientific review of score orientation'
    prediction = run / 'predictions/dayB_reference_val.jsonl'
    calibration = run / 'reports/dayB_validation_calibration.json'
    run_logged(run, 'calibration', [ROOT / 'scripts/m2_evaluate.py', 'calibrate',
        '--manifest', ROOT / 'data/manifests/manifest_whole.jsonl', '--predictions', prediction,
        '--config','reference','--target-fpr','0.05','--out',calibration], [calibration])
    protocol = load(calibration)
    threshold = protocol['threshold']
    assert math.isfinite(threshold) and 0 <= threshold <= 1, 'Invalid provisional threshold: stop, do not change calibration method'
    assert protocol['manifest_sha256'] == WHOLE and protocol['checkpoint_sha256'] == CHECKPOINT
    assert protocol['calibration_split'] == 'val' and protocol['calibration_config'] == 'reference'
    assert protocol['target_fpr'] == 0.05 and protocol['n_real'] == 300 and protocol['n_fake'] == 900
    output = run / 'reports/dayB_reference_val'
    run_logged(run, 'baseline_metrics', [ROOT / 'scripts/m2_evaluate.py','evaluate',
        '--manifest',ROOT / 'data/manifests/manifest_whole.jsonl','--predictions',prediction,
        '--protocol',calibration,'--split','val','--n-boot','1000','--seed','0','--out',output], [output])
    metrics = load(output / 'metrics.json')
    assert metrics['split'] == 'val' and metrics['n_boot'] == 1000 and metrics['seed'] == 0
    assert metrics['manifest_sha256'] == WHOLE and metrics['threshold'] == threshold
    assert set(metrics['configs']) == {'reference'} and metrics['test_accesses_so_far'] == 0
    m = metrics['configs']['reference']
    assert m['n'] == 1200 and m['n_real'] == 300 and m['n_fake'] == 900
    assert set(m['per_type']) == {'audio_fake','video_fake','both_fake'}
    def finite(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ('value','ci_low','ci_high'):
                    assert isinstance(item,(float,int)) and math.isfinite(item), 'Undefined metric/CI: stop for review'
                finite(item)
        elif isinstance(value,list):
            for item in value: finite(item)
        elif isinstance(value,float):
            assert math.isfinite(value), 'Nonfinite metric: stop for review'
    finite(metrics)
    for block in m['per_type'].values():
        assert block['n_real'] == block['n_fake'] == 300
        for metric in ('auroc','auprc','recall'):
            assert all(k in block[metric] for k in ('value','ci_low','ci_high'))
    for filename in ('curves.png','recall_by_type.png','summary.md'):
        assert (output / filename).is_file()
    save(run / 'metrics_stage.json', {'status':'VALIDATED_OUTPUTS_AWAITING_HUMAN_REVIEW',
        'calibration_sha256':digest(calibration), 'threshold':threshold,
        'provisional_validation_evidence_only':True, 'Day9_final_freeze':False,
        'reuse_this_exact_calibration_for_G6_G8':str(calibration), 'test_scoring':False})
    review_bundle(run, 'baseline_metrics')

if __name__ == '__main__':
    main()
