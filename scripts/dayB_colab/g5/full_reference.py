"""G5 full reference and prediction orientation check; no calibration or later mode."""
import argparse
import math
from pathlib import Path
from common import ROOT, CHECKPOINT, context, load, save, run_logged, review_bundle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    args = p.parse_args()
    run, c = context(args.run_dir)
    assert load(run / 'runtime_parity.json')['status'] == 'PASS'
    prediction = run / 'predictions/dayB_reference_val.jsonl'
    run_logged(run, 'full_reference', [ROOT / 'scripts/dayB_inference.py',
        '--manifest', c['ready_manifest'], '--mode', 'reference', '--device', 'cpu',
        '--upstream', c['upstream'], '--run-id', run.name + '-reference-val', '--out', prediction], [prediction])
    from avfuse.member2_data.manifest import read_manifest
    from avfuse.member2_data.predictions import validate_predictions
    from avfuse.shared.jsonl import read_jsonl
    rows = read_jsonl(prediction)
    validate_predictions(rows)
    ids = {r['clip_id'] for r in read_manifest(c['ready_manifest'])}
    assert len(rows) == len(ids) == 1200 and {r['clip_id'] for r in rows} == ids
    assert all(set(r) == {'run_id','clip_id','split','config','score_fake','checkpoint_sha256'}
               and r['split'] == 'val' and r['config'] == 'reference'
               and r['checkpoint_sha256'] == CHECKPOINT and math.isfinite(r['score_fake'])
               and 0 <= r['score_fake'] <= 1 for r in rows)
    check = run / 'reports/dayB_reference_prediction_check.json'
    run_logged(run, 'prediction_check', [ROOT / 'scripts/m2_check_predictions.py',
        '--manifest', ROOT / 'data/manifests/manifest_whole.jsonl',
        '--predictions', prediction, '--out', check], [check])
    orientation = load(check)['configs']['reference']['val']['orientation']
    save(run / 'reference_stage.json', {'status': 'REQUIRES_HUMAN_ORIENTATION_REVIEW',
         'prediction_rows': 1200, 'config':'reference', 'device':'cpu',
         'orientation': orientation, 'test_feature_access': False,
         'next_action':'STOP; return evidence before provisional calibration or later stages'})
    review_bundle(run, 'reference_orientation')

if __name__ == '__main__':
    main()
