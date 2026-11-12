"""One manual CPU run for remaining Day B G8/G9; no query/freeze/test stage."""
import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'artifacts/dayB/finish_assets'
MODES = ('reference', 'low_precision', 'restore_value_output')
CHECKPOINT = 'e1e59006d1bb6ad36a3859af5486a6649ed41bba23258b7c41e24f873be7948e'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_tree(value):
    if isinstance(value, dict):
        for child in value.values():
            finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            finite_tree(child)
    elif isinstance(value, float):
        require(math.isfinite(value), 'Undefined metric/CI; preserve evidence and stop')


def check_cohort(rows, ids, mode):
    from avfuse.member2_data.predictions import validate_predictions
    validate_predictions(rows)
    require(len(rows) == len(ids) == len({r['clip_id'] for r in rows}) == 1200
            and {r['clip_id'] for r in rows} == ids, 'Cohort/ID mismatch; never skip clips')
    require(all(set(r) == {'run_id','clip_id','split','config','score_fake','checkpoint_sha256'}
                and r['split'] == 'val' and r['config'] == mode and r['checkpoint_sha256'] == CHECKPOINT
                and math.isfinite(r['score_fake']) and 0 <= r['score_fake'] <= 1 for r in rows),
            'Prediction schema/config/checkpoint/split/score mismatch')


def check_matched(rows):
    from avfuse.member3_runtime.artifacts import validate_benchmark_records
    validate_benchmark_records(rows)
    require(len(rows) == 3 and {r['config'] for r in rows} == set(MODES), 'Exactly one record per Day B mode required')
    first = rows[0]
    for r in rows:
        require(all(r[k] == first[k] for k in ('workload','device','batch_size','backend','environment',
                    'warmup','repeats','scope','experiment_id','run_id')), 'Heterogeneous benchmark workloads cannot define restoration cost')
        require(r['device'] == 'cpu' and r['scope'] == 'cached_feature_model' and r['backend'] == 'sdpa'
                and r['warmup'] == 10 and r['repeats'] == len(r['timing_samples_ms']) == 50
                and r['peak_memory_bytes'] is None and r['memory_method'] == 'not_measured'
                and r['workload']['checkpoint_sha256'] == CHECKPOINT
                and r['precision']['execution_dtype'] == 'float32' and not r['precision']['real_int8_kernel']
                and r['conversion_overhead_included_in_latency'] == (r['config'] != 'reference'),
                'Benchmark device/precision/sampling/memory protocol mismatch')


def copy_exclusive(source, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        stream.write(Path(source).read_bytes())
    require(digest(source) == digest(target), 'Copy checksum mismatch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--g5-root', type=Path, required=True)
    parser.add_argument('--g5-run', type=Path, required=True)
    parser.add_argument('--g6-run', type=Path, required=True)
    parser.add_argument('--g7-run', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, default=Path('/content/dayB_final_results'))
    args = parser.parse_args()
    manifest = json.loads((ASSETS / 'PACKAGE_MANIFEST.json').read_text())
    for name, sha in manifest['files'].items():
        require(not (ROOT / name).is_symlink() and digest(ROOT / name) == sha, 'Final package checksum mismatch: ' + name)
    sys.path[:0] = [str(args.g5_root / 'g5'), str(args.g5_root / 'src'), str(args.g5_root), str(ROOT)]
    from common import context, load, save, run_logged, WHOLE
    g5, c = context(args.g5_run)
    accepted = load(ASSETS / 'accepted_context.json')
    require(c == accepted['g5_context'], 'Approved CPU runtime/context changed')
    g6, g7 = args.g6_run.resolve(strict=True), args.g7_run.resolve(strict=True)
    for previous in (g6, g7):
        for name, sha in load(previous / 'output_hashes.json').items():
            require(digest(previous / name) == sha, 'Prior evidence changed: ' + str(previous / name))
    require(digest(g7 / 'output_hashes.json') == accepted['g7_output_hashes_sha256'], 'Wrong G7 result set')
    require(load(g7 / 'execution_context.json')['g5_context'] == c, 'G7 runtime linkage differs')
    require(load(g6 / 'execution_context.json')['g5_context'] == c, 'G6 runtime linkage differs')
    sources = {'reference': g5 / 'predictions/dayB_reference_val.jsonl',
               'low_precision': g6 / 'predictions/dayB_low_precision_val.jsonl'}
    require(digest(sources['reference']) == accepted['g5_reference_sha256'], 'Reference changed')
    require(digest(sources['low_precision']) == accepted['g6_lowp_sha256'], 'Lowp changed')
    original_cal = g5 / 'reports/dayB_validation_calibration.json'
    require(digest(original_cal) == accepted['calibration_sha256'], 'Threshold file changed')
    args.output_root.mkdir(parents=True, exist_ok=True)
    run = args.output_root / ('DayB_FINAL_CPU_' + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8])
    run.mkdir(exist_ok=False)
    print('FINAL_RUN_DIR=' + str(run), flush=True)
    try:
        save(run / 'execution_context.json', {'gates': ['G8','G9'], 'accepted': accepted, 'environment': c['environment'],
             'package_manifest_sha256': digest(ASSETS / 'PACKAGE_MANIFEST.json'), 'test_feature_access': False,
             'Day8_query_control_executed': False, 'Day9_final_policy_frozen': False,
             'next_review': 'Single final review requested by user'})
        cal = run / 'reports/dayB_validation_calibration.json'
        copy_exclusive(original_cal, cal)
        paths = {mode: run / ('predictions/dayB_' + mode + '_val.jsonl') for mode in MODES}
        for mode, source in sources.items():
            copy_exclusive(source, paths[mode])
        # Include earlier evidence needed for one final review, without weights/cache.
        for source_dir, subdir in [(args.g5_root / 'data/g4_evidence','g4'), (g7,'g7')]:
            for relative in [*load(source_dir/'output_hashes.json'), 'output_hashes.json']:
                source = source_dir/relative
                require(source.suffix in ('.json','.jsonl','.log'), 'Unexpected payload in prior evidence')
                copy_exclusive(source, run / 'provenance' / subdir / relative)
        for source, relative in [(g5/'runtime_parity.json','g5/runtime_parity.json'),
                (g5/'execution_context.json','g5/execution_context.json'),
                (g5/'reports/dayB_reference_val/metrics.json','g5/baseline_metrics.json'),
                (g6/'reports/H2_paired_per_type.json','g6/H2_paired_per_type.json'),
                (g6/'reports/dayB_lowp_paired_val/paired.json','g6/paired.json'),
                (ASSETS/'local_suite.xml','tests/local_suite.xml'),
                (ASSETS/'local_tests.json','tests/local_tests.json'),
                (ASSETS/'G7_AUDIT.json','g7/G7_AUDIT.json')]:
            copy_exclusive(source, run / 'provenance' / relative)
        print('G8: all 1,200 V/output-restored validation predictions', flush=True)
        run_logged(run, 'full_restore_value_output', [args.g5_root/'scripts/dayB_inference.py',
            '--manifest', c['ready_manifest'], '--mode', 'restore_value_output', '--device', 'cpu',
            '--upstream', c['upstream'], '--run-id', run.name+'-restore-value-output-val',
            '--out', paths['restore_value_output']], [paths['restore_value_output']])
        from avfuse.shared.jsonl import read_jsonl
        from avfuse.member2_data.manifest import read_manifest
        whole = read_manifest(args.g5_root/'data/manifests/manifest_whole.jsonl')
        ids = {r['clip_id'] for r in whole if r['split']=='val'}
        require(Counter(r['manipulation_type'] for r in whole if r['split']=='val') ==
                {'real':300,'audio_fake':300,'video_fake':300,'both_fake':300}, 'Category counts changed')
        predictions = []
        for mode, path in paths.items():
            rows = read_jsonl(path)
            check_cohort(rows, ids, mode)
            predictions.extend(rows)
        save(run/'reports/restored_finiteness.json', {'calls':1200,'all_logits_finite':True,
             'all_scores_finite':True,'evidence':'Successful unchanged adapter per-call finite guards',
             'adapter_sha256':digest(args.g5_root/'src/avfuse/member1_fusion/dimodif_adapter.py')})
        common = ['--manifest', args.g5_root/'data/manifests/manifest_whole.jsonl', '--predictions', *paths.values()]
        check = run/'reports/dayB_three_mode_prediction_check.json'
        print('G8: three-mode fixed-threshold evaluation', flush=True)
        run_logged(run, 'prediction_check', [args.g5_root/'scripts/m2_check_predictions.py', *common, '--out', check], [check])
        require(all(load(check)['configs'][mode]['val']['orientation']=='ok' for mode in MODES),
                'Score orientation inverted/unclear; stop without flipping scores')
        metrics_dir = run/'reports/dayB_three_mode_val'
        fixed = ['--protocol', cal, '--n-boot', '1000', '--seed', '0']
        run_logged(run, 'three_mode_metrics', [args.g5_root/'scripts/m2_evaluate.py', 'evaluate',
            *common, *fixed, '--split', 'val', '--out', metrics_dir], [metrics_dir])
        metrics = load(metrics_dir/'metrics.json')
        finite_tree(metrics)
        require(metrics['threshold']==load(cal)['threshold'] and metrics['manifest_sha256']==WHOLE
                and metrics['test_accesses_so_far']==0 and metrics['split']=='val'
                and metrics['n_boot']==1000 and metrics['seed']==0 and set(metrics['configs'])==set(MODES),
                'Evaluation protocol changed')
        for block in metrics['configs'].values():
            require(block['n']==1200 and block['n_real']==300 and block['n_fake']==900
                    and set(block['per_type'])=={'audio_fake','video_fake','both_fake'}, 'Metric counts changed')
            require(all(row['n_real']==row['n_fake']==300 for row in block['per_type'].values()),
                    'Per-type metric counts changed')
        require(all((metrics_dir/name).is_file() for name in ['curves.png','recall_by_type.png','summary.md']),
                'Required metric report/plots missing')
        print('G8: fresh three-mode model benchmarks on approved workloads', flush=True)
        matched = run/'benchmarks/dayB_model_matched.jsonl'
        for item in c['approved_workloads']:
            role = item['role']
            target = matched if role=='typical_anchor' else run/('benchmarks/dayB_model_matched_'+role+'.jsonl')
            print('Benchmark workload: '+role, flush=True)
            run_logged(run, 'benchmark_'+role, [args.g5_root/'scripts/dayB_model_benchmark.py',
                '--features', item['feature_path'], '--upstream', c['upstream'], '--device', 'cpu',
                '--modes', *MODES, '--warmup', '10', '--repeats', '50', '--run-id', run.name+'-'+role,
                '--experiment-id', 'dayB-matched-validation-model'+('' if role=='typical_anchor' else '-'+role),
                '--out', target], [target])
            records = read_jsonl(target)
            check_matched(records)
            # Match the source identity/shapes to the accepted G5 benchmark too.
            oldname = 'dayB_model_reference.jsonl' if role=='typical_anchor' else 'dayB_model_reference_'+role+'.jsonl'
            require(records[0]['workload']==read_jsonl(g5/'benchmarks'/oldname)[0]['workload'], 'Benchmark workload differs from G5')
        paired_dir = run/'reports/dayB_restoration_paired_val'
        print('G8: restoration recovery using ONLY the typical matched cost set', flush=True)
        run_logged(run, 'paired_restoration', [args.g5_root/'scripts/m2_paired.py', *common, *fixed,
            '--benchmarks', matched, '--scope', 'cached_feature_model', '--out', paired_dir], [paired_dir])
        paired = load(paired_dir/'paired.json')
        finite_tree(paired)
        recovery = paired['restoration']['rows'][0]
        require(recovery['config']=='restore_value_output', 'Wrong restoration arm')
        if paired['restoration']['quantization_damage'] <= 0:
            require(recovery['share_of_damage_recovered'] is None, 'Do not normalize by absent/negative damage')
        if recovery['extra_p50_ms'] <= 0:
            require(recovery.get('R_per_ms') is None, 'Do not claim a positive recovery-per-cost ratio for nonpositive added cost')
        operator = run/'benchmarks/dayB_operator_raw_diagnostics.json'
        run_logged(run, 'operator_benchmarks', [ROOT/'scripts/dayB_finish/operator_final.py',
            '--g5-root', args.g5_root, '--out', operator], [operator])
        # Full per-type paired changes and recovery CIs, using the existing engine.
        import numpy as np
        from avfuse.member2_data.paired import aligned, paired_delta
        from avfuse.member2_data.splits import group_clusters
        from avfuse.member2_data.evaluate import _auroc, _auprc, bce, recall_at, fpr_at
        y, ordered, types, scores = aligned(whole, predictions, MODES, 'val')
        cluster_map = group_clusters(whole)
        groups = np.array([cluster_map[i] for i in ordered])
        threshold = load(cal)['threshold']
        functions = {'auroc':_auroc,'auprc':_auprc,'bce':bce,
            'recall':lambda yy,ss:recall_at(yy,ss,threshold),'real_fpr':lambda yy,ss:fpr_at(yy,ss,threshold)}
        per_type = {}
        for name in ('overall','audio_fake','video_fake','both_fake'):
            mask = np.ones(len(y),dtype=bool) if name=='overall' else ((types=='real')|(types==name))
            block = {}
            for first, second in [('reference','low_precision'),('low_precision','restore_value_output'),('reference','restore_value_output')]:
                block[second+'_minus_'+first] = {k:paired_delta(fn,y[mask],scores[first][mask],scores[second][mask],groups[mask],1000,seed=0) for k,fn in functions.items()}
            block['recovery_BCE_lowp_minus_restored'] = paired_delta(bce,y[mask],scores['restore_value_output'][mask],scores['low_precision'][mask],groups[mask],1000,seed=0)
            per_type[name] = block
        finite_tree(per_type)
        save(run/'reports/dayB_paired_per_type.json', {'n_boot':1000,'seed':0,'threshold':threshold,'results':per_type})
        print('G9: end-to-end completeness checks and handoffs', flush=True)
        for mode, path in paths.items():
            check_cohort(read_jsonl(path),ids,mode)
        require(digest(paths['reference'])==accepted['g5_reference_sha256']
                and digest(paths['low_precision'])==accepted['g6_lowp_sha256']
                and digest(cal)==accepted['calibration_sha256'], 'Preserved inputs changed')
        save(run/'reports/G9_completeness.json', {'status':'PASS','counts':{mode:1200 for mode in MODES},
            'category_counts':{'real':300,'audio_fake':300,'video_fake':300,'both_fake':300},
            'prediction_sha256':{mode:digest(path) for mode,path in paths.items()},
            'checkpoint_sha256':CHECKPOINT,'manifest_sha256':WHOLE,
            'calibration_sha256':digest(cal),'threshold':threshold,'test_feature_access':False})
        configs = {'semantics':'symmetric dynamic per-tensor fake INT8; floating tensors, no integer kernels',
            'layers':'all five joint AV self-attention layers; surrounding model weights/projections/head remain FP32',
            'reference':{'Q':'FP32','K':'FP32','V':'FP32','attention_output':'FP32'},
            'low_precision':{'Q':'fake_INT8','K':'fake_INT8','V':'fake_INT8','attention_output':'fake_INT8'},
            'restore_value_output':{'Q':'fake_INT8','K':'fake_INT8','V':'FP32','attention_output':'FP32'},
            'restore_query':{'Q':'FP32','K':'fake_INT8','V':'fake_INT8','attention_output':'fake_INT8','status':'Day8; not evaluated in DayB'}}
        save(run/'handoff/config_definitions.json',configs)
        for mode,path in paths.items():
            copy_exclusive(path,run/('handoff/'+mode+'_val.jsonl'))
        for source in [cal, metrics_dir/'metrics.json', paired_dir/'paired.json', run/'reports/dayB_paired_per_type.json', matched]:
            copy_exclusive(source,run/'handoff'/source.name)
        limits = ['validation checkpoint-selection bias','source_id is not verified speaker identity',
            'training overlap not independently verified','trailing-silence/other shortcut risks outstanding',
            'cumulative whole-model intervention differs from per-target matched-upstream diagnostic',
            'simulated INT8, no integer kernels; CPU peak memory not measured']
        save(run/'handoff/provenance.json', {'checkpoint_sha256':CHECKPOINT,'manifest_sha256':WHOLE,
            'calibration_sha256':digest(cal),'scientific_limitations':limits,
            'Day8_equal_cost_control':'not yet evaluated','test_access':False,'no_Day9_final_freeze':True})
        readme = ('# Member 2 Day B handoff\n\n'
            'Three validation prediction files, exactly 1,200 approved IDs each: reference_val.jsonl, '
            'low_precision_val.jsonl and restore_value_output_val.jsonl. Reference and lowp bytes are preserved. '
            'Join labels only from the approved canonical manifest identified in provenance.json.\n\n'
            'Reuse the exact provisional calibration JSON; do not recalibrate by configuration. '
            'metrics.json contains three-mode baseline/type metrics; paired.json contains restoration recovery '
            'using only the typical matched model benchmark; dayB_paired_per_type.json contains paired type CIs. '
            'All bootstrap calculations use source groups, 1,000 resamples, seed 0.\n\n'
            'Read config_definitions.json and provenance.json for precision boundaries and limitations. '
            'Day 8 equal-cost control comparison not yet evaluated. restore_query_val.jsonl is outstanding '
            'outside Day B; there are no test predictions and no Day 9 final policy freeze. '
            'A nonpositive quantization damage does not justify a rescue narrative. '
            'Share recovered and R per ms are unavailable when their denominators are not positive.\n')
        with (run/'handoff/README.md').open('x') as stream:
            stream.write(readme)
        handoff_files = [path for path in (run/'handoff').iterdir() if path.is_file()]
        save(run/'handoff/SHA256.json',{p.name:digest(p) for p in handoff_files})
        with zipfile.ZipFile(run/'Member2_DayB_handoff.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for path in (run/'handoff').iterdir():
                z.write(path,path.name)
        save(run/'DAYB_FINAL_STATUS.json', {'status':'G8_G9_OUTPUTS_READY_FOR_SINGLE_FINAL_REVIEW',
            'DayB_complete_claim':'PENDING FINAL EVIDENCE AUDIT AND HUMAN SCIENTIFIC REVIEW',
            'local_tests':load(ASSETS/'local_tests.json'),'acknowledgements':accepted['acknowledgements'],
            'quantization_damage':paired['restoration']['quantization_damage'],'recovery':recovery,
            'limitations':limits,'Day8_equal_cost_control':'not yet evaluated',
            'Day9_final_policy_frozen':False,'Day10_test_executed':False,'git_writes':False,
            'next_action':'STOP; return this single final ZIP; no intermediate upload required'})
        summary = ['# Day B (original Days 5 + 6 + 7) — final execution evidence', '',
            'G8/G9 outputs are ready. Final independent evidence audit and human scientific review are pending.', '',
            'Day 8 equal-cost control comparison not yet evaluated.', '',
            '| Workstream | Technical output status | Evidence |', '|---|---|---|',
            '| Day 5 Member 1 | PASS | provenance/g5/runtime_parity.json; unchanged checkpoint and code |',
            '| Day 5 Member 2 | PASS | reference 1200; provenance/g5/baseline_metrics.json; fixed calibration |',
            '| Day 5 Member 3 | PASS | fresh reference records in benchmarks/dayB_model_matched*.jsonl |',
            '| Day 6 Member 1 | PASS | lowp 1200; accepted prediction SHA preserved |',
            '| Day 6 Member 2 | PASS | provenance/g6/H2_paired_per_type.json; current three-mode metrics |',
            '| Day 6 Member 3 | PASS | fresh lowp model/operator records, same approved CPU/input |',
            '| Day 7 Member 1 | PASS | restored 1200; provenance/g7 upstream, trace and matched diagnostics |',
            '| Day 7 Member 2 | PASS | reports/dayB_restoration_paired_val/paired.json; per-type paired CIs |',
            '| Day 7 Member 3 | PASS | matched three-mode model benchmark; separate operator report |',
            '| Research hygiene | PASS with stated limitations | fixed labels/groups/orientation; limits below |',
            '| Reproducibility | PASS | execution_context.json; output_hashes.json; command logs; local tests |',
            '| Ethics and local-only handling | PASS for recorded workflow | no raw media/test features, Git writes or remote account actions |', '',
            '## Completeness and preserved protocol', '',
            f'Counts: reference=1200, lowp=1200, restored=1200. Checkpoint: {CHECKPOINT}.',
            f'Canonical manifest: {WHOLE}. Fixed provisional threshold: {threshold}.',
            'See reports/G9_completeness.json for prediction/calibration hashes and category counts.', '',
            '## Recovery and measured cost', '',
            f'Quantization damage: {paired["restoration"]["quantization_damage"]}.',
            f'Recovery and CI, added p50 milliseconds and valid normalization: {json.dumps(recovery)}',
            'See metrics, paired per-type report, plots and all raw benchmark samples for full estimates/CIs.',
            'CPU peak memory is not measured. Fake quant/dequant is included in timed mode latency. '
            'Operator-only and cached-model scopes are separate; neither is raw-video latency.', '',
            '## Limitations', '', *['- '+item for item in limits], '',
            '## Tests and repository preservation', '',
            json.dumps(load(ASSETS/'local_tests.json'),indent=2), '',
            '## Handoffs and next step', '',
            'Member2_DayB_handoff.zip contains the three files, exact definitions, calibration, metrics, '
            'paired results, typical benchmark, README and hashes. Query and test outputs remain outstanding '
            'outside this Day B scope. No Day 9 final policy has been locked.', '',
            'DAY B STATUS: PARTIALLY COMPLETE — final returned-evidence audit and human scientific review pending.',
            'STOP AT G9. Return DayB_G8_G9_review.zip for one final review.']
        with (run/'DAYB_FINAL_SUMMARY.md').open('x') as stream:
            stream.write('\n'.join(summary)+'\n')
    except BaseException:
        import traceback
        save(run/'FAILURE.json',{'traceback':traceback.format_exc(),'action':'Preserve all evidence and return the final ZIP; do not skip clips or bypass guards'})
        raise
    finally:
        files = [path for path in run.rglob('*') if path.is_file() and (path.suffix!='.zip' or path.name=='Member2_DayB_handoff.zip')]
        save(run/'output_hashes.json',{p.relative_to(run).as_posix():digest(p) for p in files})
        archive = run/'DayB_G8_G9_review.zip'
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for path in files+[run/'output_hashes.json']:
                z.write(path,path.relative_to(run).as_posix())
        print('FINAL REVIEW ZIP: '+str(archive),flush=True)
        print('STOP. G8/G9 outputs await one final review. No query, freeze or held-out test executed.',flush=True)


if __name__ == '__main__':
    main()
