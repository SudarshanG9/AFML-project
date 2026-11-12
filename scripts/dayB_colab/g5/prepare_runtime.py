"""User-run CPU runtime preparation only; no model inference or Drive mounting."""
import argparse
import datetime as dt
from pathlib import Path
import shutil
import uuid
from common import ROOT, STEM, CHECKPOINT, digest, verify_code, environment, inputs, save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output-root', type=Path, default=Path('/content/g5_results'))
    args = p.parse_args()
    verify_code()
    env = environment()
    _, workloads = inputs()
    assert args.checkpoint.is_file() and digest(args.checkpoint) == CHECKPOINT
    args.output_root.mkdir(parents=True, exist_ok=True)
    run = args.output_root / ('G5_CPU_' + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8])
    run.mkdir(exist_ok=False)
    upstream = run / 'runtime/dimodif'
    mapping = [(args.checkpoint, upstream / 'ckpt/dfd' / (STEM + '.pth')),
               (ROOT / 'third_party/dimodif/ckpt/dfd' / (STEM + '.json'), upstream / 'ckpt/dfd' / (STEM + '.json')),
               (ROOT / 'third_party/dimodif/src/models.py', upstream / 'src/models.py')]
    for source, target in mapping:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(source.read_bytes())
        assert digest(source) == digest(target)
    save(run / 'execution_context.json', {'gate': 'G5', 'device': 'cpu', 'environment': env,
         'package_manifest_sha256': digest(ROOT / 'PACKAGE_MANIFEST.json'), 'upstream': str(upstream.resolve()),
         'ready_manifest': str(ROOT / 'data/g4_evidence/dayB_val_ready.jsonl'),
         'runtime_assets': [{'path': str(dst.resolve()), 'sha256': digest(dst)} for _, dst in mapping],
         'approved_workloads': workloads, 'matched_cost_anchor': '7cea6ad75b4ebf41',
         'test_feature_access': False, 'inference_performed_during_preparation': False})
    print('G5_RUN_DIR=' + str(run.resolve()))
    print('Prepared CPU environment and runtime assets only; parity and full inference NOT executed.')

if __name__ == '__main__':
    main()
