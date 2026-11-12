"""User-run G6 only, reusing the immutable, accepted G5 CPU package and runtime."""
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys
import uuid
import zipfile

ADDON = Path(__file__).resolve().parents[2] / 'artifacts/dayB/local_deliverables/g6_colab/AFML_G6_CPU_Addon'
REF = 'e146432023638635b53dc517f3b4b06ecd87a6599efe3709ed14b145c7458ff3'
CAL = '358c037242f6980fed2a8d0ff0ef830f17b4789676b391c8f6dff245842b37dd'
G5_PACKAGE = '6d389dc16e7dc46f658cc81c776ceece5913913ec1bfcf9c43b0bcf0989e9863'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def finite_tree(value):
    if isinstance(value, dict):
        for v in value.values():
            finite_tree(v)
    elif isinstance(value, list):
        for v in value:
            finite_tree(v)
    elif isinstance(value, float):
        require(math.isfinite(value), 'Nonfinite metric/CI; stop without changing protocol')


def validate_rows(rows, ids, mode, checkpoint):
    require(len(rows) == len(ids) == 1200, 'Full 1200-clip cohort is mandatory')
    require(len({r['clip_id'] for r in rows}) == 1200 and {r['clip_id'] for r in rows} == ids,
            'Duplicate/missing/extra clip IDs')
    for r in rows:
        require(set(r) == {'run_id', 'clip_id', 'split', 'config', 'score_fake', 'checkpoint_sha256'},
                'Unexpected prediction schema')
        require(r['split'] == 'val' and r['config'] == mode and r['checkpoint_sha256'] == checkpoint,
                'Split/config/checkpoint mismatch')
        require(math.isfinite(r['score_fake']) and 0 <= r['score_fake'] <= 1, 'Invalid score')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--g5-root', type=Path, required=True)
    p.add_argument('--g5-run', type=Path, required=True)
    p.add_argument('--output-root', type=Path, default=Path('/content/g6_results'))
    args = p.parse_args()
    manifest = json.loads((ADDON / 'ADDON_MANIFEST.json').read_text())
    actual = {x.relative_to(ADDON).as_posix() for x in ADDON.rglob('*')
              if x.is_file() and '__pycache__' not in x.parts and x.name != 'ADDON_MANIFEST.json'}
    require(actual == set(manifest['files']), 'Add-on file set changed')
    for name, sha in manifest['files'].items():
        require(digest(ADDON / name) == sha, 'Add-on checksum mismatch: ' + name)
    root = args.g5_root.resolve(strict=True)
    require(digest(root / 'PACKAGE_MANIFEST.json') == G5_PACKAGE, 'Unapproved G5 code package')
    sys.path[:0] = [str(root / 'g5'), str(root / 'src'), str(root)]
    from common import context, load, save, run_logged, CHECKPOINT, WHOLE
    original_run, c = context(args.g5_run)
    require(load(original_run / 'runtime_parity.json')['status'] == 'PASS', 'G5 parity missing')
    require(c['environment'] == load(ADDON / 'accepted_g5_execution_context.json')['environment'],
            'Accepted G5 CPU environment changed; stop for review')
    original_ref = original_run / 'predictions/dayB_reference_val.jsonl'
    original_cal = original_run / 'reports/dayB_validation_calibration.json'
    require(digest(original_ref) == REF and digest(original_cal) == CAL, 'Accepted reference/calibration changed')
    args.output_root.mkdir(parents=True, exist_ok=True)
    run = args.output_root / ('G6_CPU_' + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8])
    run.mkdir(exist_ok=False)
    print('G6_RUN_DIR=' + str(run), flush=True)
    try:
        ref = run / 'predictions/dayB_reference_val.jsonl'
        cal = run / 'reports/dayB_validation_calibration.json'
        for source, target in [(original_ref, ref), (original_cal, cal)]:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(source.read_bytes())
        save(run / 'execution_context.json', {'gate': 'G6', 'g5_run': str(original_run),
             'g5_context': c, 'addon_manifest_sha256': digest(ADDON / 'ADDON_MANIFEST.json'),
             'reference_sha256': REF, 'calibration_sha256': CAL, 'test_feature_access': False,
             'approval': 'User ACK GATE G5; no G7/G8 execution authorized by this helper'})
        low = run / 'predictions/dayB_low_precision_val.jsonl'
        logits = run / 'reports/low_precision_finiteness.json'
        print('G6: full 1,200-clip low-precision validation inference', flush=True)
        run_logged(run, 'full_low_precision', [root / 'scripts/dayB_inference.py',
            '--manifest', c['ready_manifest'], '--mode', 'low_precision', '--device', 'cpu',
            '--upstream', c['upstream'], '--run-id', run.name + '-low_precision-val', '--out', low], [low, logits])
        from avfuse.shared.jsonl import read_jsonl
        from avfuse.member2_data.manifest import read_manifest
        ids = {r['clip_id'] for r in read_manifest(c['ready_manifest'])}
        reference, quantized = read_jsonl(ref), read_jsonl(low)
        validate_rows(reference, ids, 'reference', CHECKPOINT)
        validate_rows(quantized, ids, 'low_precision', CHECKPOINT)
        # The unchanged adapter raises on nonfinite logits and scores in every
        # forward_with_mode call. A successful full run therefore checks all clips.
        save(logits, {'calls': 1200, 'all_logits_finite': True, 'all_scores_finite': True,
             'evidence': 'Original forward_with_mode finite-logit/score guards; successful unchanged 1200-clip inference',
             'adapter_sha256': digest(root / 'src/avfuse/member1_fusion/dimodif_adapter.py')})
        check = run / 'reports/dayB_reference_lowp_check.json'
        common = ['--manifest', root / 'data/manifests/manifest_whole.jsonl', '--predictions', ref, low]
        print('G6: prediction checks and fixed-threshold paired evaluation', flush=True)
        run_logged(run, 'prediction_check', [root / 'scripts/m2_check_predictions.py', *common, '--out', check], [check])
        orientations = {mode: load(check)['configs'][mode]['val']['orientation']
                        for mode in ('reference', 'low_precision')}
        require(all(x == 'ok' for x in orientations.values()), 'Orientation inverted/unclear; stop for interpretation, never flip scores')
        output = run / 'reports/dayB_reference_lowp_val'
        fixed = ['--protocol', cal, '--n-boot', '1000', '--seed', '0']
        run_logged(run, 'baseline_metrics', [root / 'scripts/m2_evaluate.py', 'evaluate', *common,
                   *fixed, '--split', 'val', '--out', output], [output])
        paired = run / 'reports/dayB_lowp_paired_val'
        run_logged(run, 'paired_metrics', [root / 'scripts/m2_paired.py', *common, *fixed, '--out', paired], [paired])
        metrics = load(output / 'metrics.json')
        finite_tree(metrics)
        finite_tree(load(paired / 'paired.json'))
        require(metrics['manifest_sha256'] == WHOLE and metrics['test_accesses_so_far'] == 0
                and metrics['n_boot'] == 1000 and metrics['seed'] == 0, 'Metric provenance mismatch')
        require(metrics['threshold'] == load(cal)['threshold'], 'Threshold changed')
        # Additional per-type paired CIs using the exact existing bootstrap engine.
        import numpy as np
        from avfuse.member2_data.paired import aligned, paired_delta
        from avfuse.member2_data.evaluate import _auroc, _auprc, bce, recall_at
        from avfuse.member2_data.splits import group_clusters
        whole = read_manifest(root / 'data/manifests/manifest_whole.jsonl')
        clusters = group_clusters(whole)
        y, ordered_ids, types, scores = aligned(whole, reference + quantized, ['reference', 'low_precision'], 'val')
        groups = np.array([clusters[i] for i in ordered_ids])
        threshold = load(cal)['threshold']
        fns = {'auroc': _auroc, 'auprc': _auprc, 'bce': bce,
               'recall': lambda yy, ss: recall_at(yy, ss, threshold),
               'real_fpr': lambda yy, ss: float(np.mean(ss[yy == 0] >= threshold)) if np.any(yy == 0) else float('nan')}
        h2 = {}
        for name in ('overall', 'audio_fake', 'video_fake', 'both_fake'):
            mask = np.ones(len(y), dtype=bool) if name == 'overall' else ((types == 'real') | (types == name))
            h2[name] = {k: paired_delta(f, y[mask], scores['reference'][mask], scores['low_precision'][mask],
                        groups[mask], 1000, seed=0) for k, f in fns.items()}
        finite_tree(h2)
        diffs = scores['low_precision'] - scores['reference']
        save(run / 'reports/H2_paired_per_type.json', {'direction': 'low_precision minus reference',
             'n_boot': 1000, 'seed': 0, 'threshold': threshold, 'results': h2,
             'nonzero_score_differences': int(np.count_nonzero(diffs)),
             'max_abs_score_difference': float(np.max(np.abs(diffs))),
             'scientific_interpretation': 'PENDING HUMAN REVIEW; no positive damage or rescue assumed'})
        from avfuse.member3_runtime.artifacts import validate_benchmark_records
        for item in c['approved_workloads']:
            role = item['role']
            print('G6: matched reference/lowp model benchmark: ' + role, flush=True)
            target = run / ('benchmarks/dayB_model_reference_lowp_' + role + '.jsonl')
            run_logged(run, 'benchmark_' + role, [root / 'scripts/dayB_model_benchmark.py',
                '--features', item['feature_path'], '--upstream', c['upstream'], '--device', 'cpu',
                '--modes', 'reference', 'low_precision', '--warmup', '10', '--repeats', '50',
                '--run-id', run.name + '-' + role, '--experiment-id',
                ('dayB-matched-validation-model' if role == 'typical_anchor' else 'dayB-matched-validation-model-' + role),
                '--out', target], [target])
            records = read_jsonl(target)
            validate_benchmark_records(records)
            require(len(records) == 2 and {r['config'] for r in records} == {'reference', 'low_precision'}, 'Benchmark modes differ')
            for r in records:
                require(r['device'] == 'cpu' and r['warmup'] == 10 and r['repeats'] == 50
                        and len(r['timing_samples_ms']) == 50 and r['peak_memory_bytes'] is None
                        and r['scope'] == 'cached_feature_model' and r['workload']['checkpoint_sha256'] == CHECKPOINT,
                        'Benchmark protocol mismatch')
        operator = run / 'benchmarks/dayB_attention_reference_lowp.json'
        print('G6: isolated attention benchmarks on original Day A Q/K/V', flush=True)
        run_logged(run, 'operator_benchmark', [ADDON / 'g6/operator_g6.py', '--g5-root', root, '--out', operator], [operator])
        require(digest(ref) == REF and digest(cal) == CAL, 'Reference or calibration changed')
        save(run / 'G6_status.json', {'status': 'OUTPUTS_READY_FOR_G6_HUMAN_AUDIT', 'n_val': 1200,
             'test_feature_access': False, 'G7_executed': False, 'G8_executed': False,
             'limitations': ['checkpoint-selection bias', 'source groups are not speaker identities',
                'training overlap not independently verified', 'shortcut controls outstanding'],
             'next_action': 'STOP; return ZIP for G6 interpretation and acknowledgement'})
    except BaseException:
        import traceback
        save(run / 'G6_failure.json', {'traceback': traceback.format_exc(), 'action': 'STOP; preserve files and return failure bundle'})
        raise
    finally:
        files = [x for x in run.rglob('*') if x.is_file() and x.suffix != '.zip']
        save(run / 'output_hashes.json', {x.relative_to(run).as_posix(): digest(x) for x in files})
        target = run / 'G6_review.zip'
        with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED) as z:
            for x in files + [run / 'output_hashes.json']:
                z.write(x, x.relative_to(run).as_posix())
        print('REVIEW ZIP: ' + str(target), flush=True)
        print('STOP AT G6. Do not execute G7 or G8.', flush=True)


if __name__ == '__main__':
    main()
