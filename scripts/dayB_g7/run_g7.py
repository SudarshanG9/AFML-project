"""Manual G7 CPU audit only; no full restored predictions or G8 commands."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'artifacts/dayB/g7_assets'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--g5-root', type=Path, required=True)
    p.add_argument('--g5-run', type=Path, required=True)
    p.add_argument('--g6-run', type=Path, required=True)
    p.add_argument('--output-root', type=Path, default=Path('/content/g7_results'))
    p.add_argument('--matched-scope', choices=['one-target-lowp-prefix-and-suffix'],
                   help='Use only after the user approves the separate single-clip diagnostic scope')
    args = p.parse_args()
    manifest = json.loads((ASSETS / 'G7_PACKAGE_MANIFEST.json').read_text())
    for name, sha in manifest['files'].items():
        path = ROOT / name
        if path.is_symlink() or digest(path) != sha:
            raise RuntimeError('G7 package checksum mismatch: ' + name)
    sys.path[:0] = [str(args.g5_root / 'g5'), str(args.g5_root / 'src'), str(args.g5_root), str(ROOT)]
    from common import context, load, save, run_logged
    g5, c = context(args.g5_run)
    accepted = load(ASSETS / 'accepted_context.json')
    if c != accepted['g5_context']:
        raise RuntimeError('Approved G5 runtime context differs; stop')
    g6 = args.g6_run.resolve(strict=True)
    if digest(g6 / 'predictions/dayB_low_precision_val.jsonl') != accepted['g6_lowp_sha256']:
        raise RuntimeError('Accepted G6 predictions differ')
    if digest(g5 / 'predictions/dayB_reference_val.jsonl') != accepted['g5_reference_sha256']:
        raise RuntimeError('Accepted reference predictions differ')
    if digest(g5 / 'reports/dayB_validation_calibration.json') != accepted['calibration_sha256']:
        raise RuntimeError('Provisional threshold changed')
    if load(g6 / 'execution_context.json')['g5_context'] != c:
        raise RuntimeError('G6/G5 runtime linkage differs')
    for name, sha in load(g6 / 'output_hashes.json').items():
        if digest(g6 / name) != sha:
            raise RuntimeError('G6 evidence changed: ' + name)
    anchor = next(item for item in c['approved_workloads'] if item['role'] == 'typical_anchor')
    if anchor['clip_id'] != '7cea6ad75b4ebf41' or digest(anchor['feature_path']) != anchor['feature_sha256']:
        raise RuntimeError('Approved validation anchor differs')
    if args.matched_scope and load(ASSETS / 'scope_decision.json')['status'] != 'APPROVED':
        raise RuntimeError('Separate diagnostic scope approval is pending; stop')
    args.output_root.mkdir(parents=True, exist_ok=True)
    run = args.output_root / ('G7_CPU_' + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8])
    run.mkdir(exist_ok=False)
    print('G7_RUN_DIR=' + str(run), flush=True)
    try:
        save(run / 'execution_context.json', {'gate': 'G7', 'g5_context': c, 'g6_run': str(g6),
             'G7_package_manifest_sha256': digest(ASSETS / 'G7_PACKAGE_MANIFEST.json'),
             'scope_decision': load(ASSETS / 'scope_decision.json'), 'matched_scope': args.matched_scope,
             'workload': anchor, 'test_feature_access': False, 'labels_used': False,
             'G8_executed': False})
        upstream = run / 'upstream_audit.json'
        print('G7: standard cumulative whole-model upstream audit', flush=True)
        run_logged(run, 'upstream_audit', [ROOT / 'scripts/dayB_upstream_audit.py',
            '--features', anchor['feature_path'], '--upstream', c['upstream'], '--device', 'cpu', '--out', upstream], [upstream])
        rows = load(upstream)['layers']
        if len(rows) != 5 or not rows[0]['upstream_bit_exact']:
            raise RuntimeError('Five-layer upstream audit failed')
        trace_out = run / 'restore_value_output_trace.json'
        print('G7: original full-model pilot restoration trace', flush=True)
        run_logged(run, 'restore_pilot_trace', [args.g5_root / 'scripts/dayB_inference.py',
            '--features', anchor['feature_path'], '--mode', 'restore_value_output', '--device', 'cpu',
            '--upstream', c['upstream'], '--trace-out', trace_out], [trace_out])
        trace = load(trace_out)['trace']
        if len(trace) != 20:
            raise RuntimeError('Expected all 20 precision boundaries')
        for row in trace:
            changed = row['fraction_changed']
            if row['tensor'] in ('V', 'attention_output'):
                if changed != 0:
                    raise RuntimeError('V/output unexpectedly quantized')
            elif changed <= 0:
                raise RuntimeError('Q/K fake quantization had no observed effect; stop for interpretation')
        if args.matched_scope:
            target = run / 'matched_upstream_diagnostic.json'
            print('G7 separate subphase: five single-target matched-upstream comparisons', flush=True)
            run_logged(run, 'matched_upstream_diagnostic', [ROOT / 'scripts/dayB_matched_upstream_diagnostic.py',
                '--features', anchor['feature_path'], '--upstream', c['upstream'], '--device', 'cpu', '--out', target], [target])
            diag = load(target)
            if len(diag['layers']) != 5 or not diag['baseline_matches_original_adapter']:
                raise RuntimeError('Separate matched-upstream diagnostic failed')
            from avfuse.shared.jsonl import read_jsonl
            previous = next(row for row in read_jsonl(g6 / 'predictions/dayB_low_precision_val.jsonl')
                            if row['clip_id'] == anchor['clip_id'])
            if previous['score_fake'] != diag['official_low_precision_score']:
                raise RuntimeError('Single-clip lowp score differs from accepted G6')
        status = 'G7_EVIDENCE_READY_FOR_HUMAN_REVIEW' if args.matched_scope else 'STANDARD_AUDIT_ONLY_SEPARATE_DIAGNOSTIC_OUTSTANDING'
        save(run / 'G7_status.json', {'status': status, 'G8_executed': False, 'test_feature_access': False,
             'matched_diagnostic_executed': bool(args.matched_scope),
             'next_action': 'STOP. Return G7_review.zip for research interpretation and ACK G7.'})
    except BaseException:
        import traceback
        save(run / 'G7_failure.json', {'traceback': traceback.format_exc(), 'action': 'STOP and preserve evidence'})
        raise
    finally:
        files = [path for path in run.rglob('*') if path.is_file() and path.suffix != '.zip']
        save(run / 'output_hashes.json', {path.relative_to(run).as_posix(): digest(path) for path in files})
        archive = run / 'G7_review.zip'
        with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
            for path in files + [run / 'output_hashes.json']:
                z.write(path, path.relative_to(run).as_posix())
        print('REVIEW ZIP: ' + str(archive), flush=True)
        print('STOP AT G7; no G8 full restoration inference.', flush=True)


if __name__ == '__main__':
    main()
