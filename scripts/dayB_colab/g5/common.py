"""Shared G5 manual-execution guards; no account or Drive operations."""
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
CHECKPOINT = 'e1e59006d1bb6ad36a3859af5486a6649ed41bba23258b7c41e24f873be7948e'
VAL = '22d1d9e02403cfdbb05f2116da16bc4015f540547a4ac00bf6b7127c119e359c'
WHOLE = '8339b2476a71e62c6c6fca7d937ba4cc7a704b09d26f08506c20d1f10de8f3dd'
READY = 'c79ad908517878ebf0f3aa0e0dac053a01d9d5c4eb264583db26096985890481'
STEM = 'dfd_lavdf_reduceonplateau_64_4_1_5_True'
COUNTS = {'real': 300, 'audio_fake': 300, 'video_fake': 300, 'both_fake': 300}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def load(path):
    return json.loads(Path(path).read_text())


def verify_code():
    manifest = load(ROOT / 'PACKAGE_MANIFEST.json')
    expected = {row['path'] for row in manifest['files']}
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts and p.name != 'PACKAGE_MANIFEST.json'}
    assert actual == expected, 'Package file set changed; use the verified archive'
    for row in manifest['files']:
        path = ROOT / row['path']
        if path.is_symlink() or digest(path) != row['sha256']:
            raise RuntimeError('Package checksum mismatch: ' + row['path'])
    return manifest


def environment():
    import platform
    import torch
    expected = load(ROOT / 'data/g4_evidence/execution_environment.json')
    versions = {name: importlib.metadata.version(name) for name in
                ('numpy', 'jsonschema', 'torch', 'scikit-learn', 'matplotlib', 'pytest')}
    if versions['torch'] != expected['package_versions']['torch'] or versions['torch'] != '2.14.1+cpu':
        raise RuntimeError('Approved existing CPU Torch environment changed; stop for review, do not reinstall Torch silently')
    assert versions['numpy'] == '2.1.3' and versions['jsonschema'] == '4.23.0'
    return {'python': sys.version, 'platform': platform.platform(), 'versions': versions,
            'device': 'cpu', 'torch_threads': torch.get_num_threads(),
            'torch_interop_threads': torch.get_num_interop_threads(),
            'cuda_available': torch.cuda.is_available()}


def inputs():
    from collections import Counter
    from avfuse.member2_data.manifest import read_manifest
    from avfuse.member2_data.splits import split_audit
    g4 = ROOT / 'data/g4_evidence'
    for name, expected in load(g4 / 'output_hashes.json').items():
        assert digest(g4 / name) == expected, name
    report = load(g4 / 'preflight_report.json')
    assert report['status'] == 'PASS' and report['n_verified'] == report['n_expected'] == 1200
    assert report['errors_count'] == 0 and report['errors'] == []
    assert digest(ROOT / 'data/manifests/manifest_whole.jsonl') == WHOLE
    assert digest(ROOT / 'data/manifests/manifest_whole_val.jsonl') == VAL
    assert digest(g4 / 'dayB_val_ready.jsonl') == READY
    val = read_manifest(ROOT / 'data/manifests/manifest_whole_val.jsonl')
    ready = read_manifest(g4 / 'dayB_val_ready.jsonl')
    whole = read_manifest(ROOT / 'data/manifests/manifest_whole.jsonl')
    assert len(whole) == 3200 and split_audit(whole)['passed']
    assert len(ready) == len(val) == 1200 and Counter(r['manipulation_type'] for r in ready) == COUNTS
    originals = {r['clip_id']: r for r in val}
    assert {r['clip_id'] for r in ready} == set(originals)
    val_root = Path(report['cache_root']) / 'val'
    if not val_root.is_dir() or val_root.is_symlink():
        raise RuntimeError('Original G4-mounted val/ path unavailable; do not rewrite the ready manifest')
    for row in ready:
        path = Path(row['feature_path'])
        assert row['split'] == 'val' and path == val_root / row['clip_id'] / 'features.npz'
        assert path.is_file() and path.resolve().is_relative_to(val_root)
        assert {k: v for k, v in row.items() if k != 'feature_path'} == {
            k: v for k, v in originals[row['clip_id']].items() if k != 'feature_path'}
    workloads = load(ROOT / 'g5/approved_workloads.json')['candidates']
    for item in workloads:
        path = Path(item['feature_path'])
        assert item['clip_id'] in originals and path == val_root / item['clip_id'] / 'features.npz'
        assert digest(path) == item['feature_sha256']
    return ready, workloads


def context(run):
    verify_code()
    current = environment()
    run = Path(run).resolve(strict=True)
    c = load(run / 'execution_context.json')
    assert c['device'] == 'cpu' and c['package_manifest_sha256'] == digest(ROOT / 'PACKAGE_MANIFEST.json')
    assert c['environment'] == current, 'Runtime changed between G5 stages; stop for review'
    for row in c['runtime_assets']:
        assert digest(row['path']) == row['sha256']
    assert digest(Path(c['upstream']) / 'ckpt/dfd' / (STEM + '.pth')) == CHECKPOINT
    inputs()
    return run, c


def run_logged(run, label, arguments, protected=()):
    record = Path(run) / (label + '_command.json')
    assert not record.exists(), 'Preserve earlier command logs; use a fresh run for a rerun'
    for path in protected:
        assert not Path(path).exists(), 'Refusing to overwrite: ' + str(path)
    argv = [sys.executable, '-B', *map(str, arguments)]
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True,
                            env=dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / 'src'), str(ROOT)]),
                                     PYTHONDONTWRITEBYTECODE='1', GIT_OPTIONAL_LOCKS='0', MPLBACKEND='Agg'))
    save(record, {'argv': argv, 'cwd': str(ROOT), 'exit_status': result.returncode,
                  'stdout': result.stdout, 'stderr': result.stderr})
    print(result.stdout, end='')
    print(result.stderr, end='', file=sys.stderr)
    if result.returncode:
        raise RuntimeError(f'{label} failed with exit {result.returncode}; preserve logs and stop')


def review_bundle(run, stage):
    run = Path(run)
    files = [p for p in run.rglob('*') if p.is_file() and 'runtime' not in p.relative_to(run).parts
             and p.suffix != '.zip' and '__pycache__' not in p.parts]
    hashes = {p.relative_to(run).as_posix(): digest(p) for p in files}
    save(run / ('review_hashes_' + stage + '.json'), hashes)
    target = run / ('G5_review_' + stage + '.zip')
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for p in files + [run / ('review_hashes_' + stage + '.json')]:
            z.write(p, arcname=p.relative_to(run).as_posix())
    print('REVIEW ZIP:', target)
    print('STOP. Return the evidence for human review; no next stage was executed.')
