# Restore exact original inputs — user-run Colab cell

This implements the user's request for a restore script using the base ZIPs already in `/content`. It restores accepted original bytes; it does not fabricate/regenerate execution context, manifests, weights or features. The agent has prepared this locally and has not executed it on Colab.

## Existing base ZIPs: paste ONE cell

Keep the existing v2 pilot extraction; the script reads its unchanged approved release. Paste the complete cell below. It searches ZIPs already in `/content` for nine required originals and uses the packaged accepted G5 context copy only after verifying its exact original SHA. It verifies all sources and any existing destinations before writing. Existing matching files are preserved; any conflicting file stops without overwrite. Missing originals stop before writing any destination. Only the recorded original `/content` model/metadata paths are restored. No archive is extracted wholesale.

```python
"""User-run exact-byte restoration from backups; never fabricate or run inference."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import uuid
import zipfile


def sha_bytes(data):return hashlib.sha256(data).hexdigest()


def sha_file(path):
    d=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):d.update(b)
    return d.hexdigest()


def validate_destination(root, path, expected):
    root=Path(root).resolve(strict=True);path=Path(path)
    if not path.is_absolute():path=root/path
    if not path.resolve().is_relative_to(root) or path.is_symlink():
        raise PermissionError('Unsafe destination: '+str(path))
    if os.path.lexists(path):
        if not path.is_file() or sha_file(path)!=expected:
            raise FileExistsError('Existing destination differs; preserving it: '+str(path))
        return path,True
    return path,False


def write_original(root,path,data,expected):
    path,existing=validate_destination(root,path,expected)
    if existing:return 'PRESERVED_MATCHING_FILE'
    if sha_bytes(data)!=expected:raise ValueError('Original source hash mismatch')
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.parent/('.restore_'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('xb') as f:
            f.write(data);f.flush();os.fsync(f.fileno())
        os.link(temporary,path)  # Atomic, exclusive; never replace a destination.
        if sha_file(path)!=expected:raise RuntimeError('Restored output hash mismatch')
    finally:
        temporary.unlink(missing_ok=True)
    return 'RESTORED_EXACT_ORIGINAL_BYTES'


def restore_from_backups(root, targets, archive_paths, loose_paths=()):
    """Verify all required sources/destinations first, then restore missing files.

    targets: label -> {path, sha256, names}. No code from ZIPs is executed and
    no archive is extracted wholesale. Canonical manifest bytes are not parsed.
    """
    root=Path(root).resolve(strict=True);plan={};sources={};report=[]
    for label,item in targets.items():
        path,exists=validate_destination(root,Path(item['path']),item['sha256'])
        plan[label]=(path,exists)
    missing={name for name,(_,exists) in plan.items() if not exists}
    for path in map(Path,loose_paths):
        if not path.is_file() or path.is_symlink():continue
        labels=[name for name in missing if path.name in targets[name]['names']]
        if labels and path.stat().st_size<=32*1024*1024:
            data=path.read_bytes();digest=sha_bytes(data)
            for name in labels:
                if digest==targets[name]['sha256']:
                    sources[name]=(data,str(path));missing.remove(name)
    for archive in map(Path,archive_paths):
        if not missing:break
        with zipfile.ZipFile(archive) as z:
            if len(z.infolist())>20000:continue
            for info in z.infolist():
                p=PurePosixPath(info.filename)
                if (info.is_dir() or p.is_absolute() or '..' in p.parts
                        or any(part.lower() in {'test','train'} for part in p.parts)
                        or info.file_size>32*1024*1024):continue
                labels=[name for name in missing if p.name in targets[name]['names']]
                if not labels:continue
                data=z.read(info);digest=sha_bytes(data)
                for name in labels:
                    if digest==targets[name]['sha256']:
                        sources[name]=(data,str(archive)+'!'+info.filename);missing.remove(name)
    if missing:
        raise FileNotFoundError('No verified original found for: '+', '.join(sorted(missing))+
            '. No destination was written. Upload the provided Verified Original Inputs ZIP; do not regenerate these files.')
    for label,item in targets.items():
        path,exists=plan[label]
        if exists:state='PRESERVED_MATCHING_FILE';origin='existing destination'
        else:
            data,origin=sources[label];state=write_original(root,path,data,item['sha256'])
        report.append({'asset':label,'path':str(path),'sha256':sha_file(path),'status':state,'source':origin})
    return report


def targets_from_release(c):
    targets={key:{'path':c[key],'sha256':c[key+'_sha256'],'names':[Path(c[key]).name]}
        for key in ('g5_context','canonical_manifest','validation_manifest','ready_manifest','g4_report')}
    targets['g5_context']['names']+=['accepted_g5_context.json','accepted_g5_execution_context.json']
    for asset in c['runtime_assets']:
        name=Path(asset['path']).name
        label='checkpoint' if name.endswith('.pth') else 'model_source' if name=='models.py' else 'model_config'
        targets[label]={'path':asset['path'],'sha256':asset['sha256'],'names':[name]}
    targets['reference_predictions']={'path':c['g5_run']+'/predictions/dayB_reference_val.jsonl',
        'sha256':c['dayb_reference_anchor']['prediction_file_sha256'],
        'names':['dayB_reference_val.jsonl','reference_val.jsonl']}
    return targets


def run_colab_restoration():
    content=Path('/content');addon=content/'AFML_DayC_C8_2_CPU_Pilot_Drift_v2'
    release_path=addon/'configs/dayC/member1/colab_pilot_release.json'
    if sha_file(release_path)!='ea6edfdab55ac88c7f6fa08ce1f472819b6a335b4dbd66aeb758d133b7a93bd6':
        raise ValueError('V2 release does not match approved configuration')
    c=json.loads(release_path.read_text())
    report=restore_from_backups(content,targets_from_release(c),sorted(content.glob('*.zip')),
        [addon/'assets/accepted_g5_context.json',content/'dfd_lavdf_reduceonplateau_64_4_1_5_True.pth'])
    print('ORIGINAL_FILES_RESTORED\n'+json.dumps(report,indent=2))
    ready=Path(c['ready_manifest']);rows=[json.loads(line) for line in ready.read_text().splitlines() if line.strip()]
    if len(rows)!=1200 or any(row['split']!='val' for row in rows):raise ValueError('Not the approved full validation cohort')
    cache=Path(c['cache_root']);present=0
    for row in rows:
        feature=Path(row['feature_path'])
        if feature!=cache/'val'/row['clip_id']/'features.npz':raise ValueError('Original manifest path differs')
        present+=feature.is_file()
    print('VALIDATION_FEATURE_FILES_PRESENT',present,'OF',1200)
    if present!=1200:
        print('CACHE STILL MISSING. The metadata/code ZIPs cannot generate the 1,200 cached feature tensors. No Drive paths were created or manifests rewritten.')
    else:print('All feature paths exist. Return this output for exact-cache/preflight verification before a pilot run.')
    print('STOP: restoration only. No inference, training, extraction, installation or full-validation run executed.')


# Optional fallback ZIP: all original bytes, no fabricated replacements.
fallback = Path('/content/AFML_DayC_Verified_Original_Inputs.zip')
if fallback.exists():
    assert sha_file(fallback) == '686a3678708d30c283677c5c8c0af715d96c464f18ec461811eafbe5b1555463', 'Fallback ZIP changed'
run_colab_restoration()
```

## If the base ZIPs lack weights or evidence

The original G4/G5 code ZIPs do not include the checkpoint or full validation cache. To avoid requesting separate metadata/model files, a single fallback archive is ready locally:

`/Users/sudarshan/Documents/GitHub/AFML-project/artifacts/dayC/member1/colab/AFML_DayC_Verified_Original_Inputs.zip`

SHA-256: `686a3678708d30c283677c5c8c0af715d96c464f18ec461811eafbe5b1555463`; 4480237 bytes; 11 entries, including **all nine verified original files**, the tested restoration script and a new transfer index. Every asset's bytes match the accepted hash. Original execution context/manifests/model files are copied byte-for-byte into this new transfer ZIP, not rebuilt from parsed JSON. The canonical manifest is opaque bytes; no test rows are parsed or used. This private transfer archive stays under ignored artifacts, separate from versionable source.

If required originals are missing from the existing base ZIPs, manually upload this one fallback ZIP into `/content`. Then run the shorter cell below instead:

```python
from pathlib import Path
import hashlib, zipfile

archive = Path('/content/AFML_DayC_Verified_Original_Inputs.zip')
assert hashlib.sha256(archive.read_bytes()).hexdigest() == '686a3678708d30c283677c5c8c0af715d96c464f18ec461811eafbe5b1555463'
with zipfile.ZipFile(archive) as z:
    script = z.read('restore_verified_inputs.py')
assert hashlib.sha256(script).hexdigest() == '181c3abf819a95c728fb205ffe88af4b2c8c26c75f0e25ca6a8f74fcb129bf2a'
namespace = {}
exec(compile(script, 'verified_restore_script', 'exec'), namespace)
namespace['run_colab_restoration']()
```

## What is restored

- g5_context: `/content/g5_results/G5_CPU_20261111T181935Z_422cf72f/execution_context.json`; SHA-256 `c03488cb787b1230bd86274dc7a36d39983574c7e2bc996c931dcfc8a835bd7c`.
- canonical_manifest: `/content/AFML_G5_CPU_Colab_20261111T180139Z/data/manifests/manifest_whole.jsonl`; SHA-256 `8339b2476a71e62c6c6fca7d937ba4cc7a704b09d26f08506c20d1f10de8f3dd`.
- validation_manifest: `/content/AFML_G5_CPU_Colab_20261111T180139Z/data/manifests/manifest_whole_val.jsonl`; SHA-256 `22d1d9e02403cfdbb05f2116da16bc4015f540547a4ac00bf6b7127c119e359c`.
- ready_manifest: `/content/AFML_G5_CPU_Colab_20261111T180139Z/data/g4_evidence/dayB_val_ready.jsonl`; SHA-256 `c79ad908517878ebf0f3aa0e0dac053a01d9d5c4eb264583db26096985890481`.
- g4_report: `/content/AFML_G5_CPU_Colab_20261111T180139Z/data/g4_evidence/preflight_report.json`; SHA-256 `559445575ff51495dbb79ae175737ad1596fcc41afab058734d138bd01fb9c01`.
- checkpoint: `/content/g5_results/G5_CPU_20261111T181935Z_422cf72f/runtime/dimodif/ckpt/dfd/dfd_lavdf_reduceonplateau_64_4_1_5_True.pth`; SHA-256 `e1e59006d1bb6ad36a3859af5486a6649ed41bba23258b7c41e24f873be7948e`.
- model_config: `/content/g5_results/G5_CPU_20261111T181935Z_422cf72f/runtime/dimodif/ckpt/dfd/dfd_lavdf_reduceonplateau_64_4_1_5_True.json`; SHA-256 `bea128dc7dad6b72bc0487dcc45d03ae3322e48fd597de1a5cd83ad036f30651`.
- model_source: `/content/g5_results/G5_CPU_20261111T181935Z_422cf72f/runtime/dimodif/src/models.py`; SHA-256 `e74b5e3b5fcee24353f9940dddd48d9ac1982ced725664db2518f63a6d853778`.
- reference_predictions: `/content/g5_results/G5_CPU_20261111T181935Z_422cf72f/predictions/dayB_reference_val.jsonl`; SHA-256 `e146432023638635b53dc517f3b4b06ecd87a6599efe3709ed14b145c7458ff3`.

## What it does not do

It never installs packages, mounts/connects Drive, modifies model/precision mathematics, rewrites `feature_path`, replaces existing conflicting files, runs inference or creates a test attempt. It does not touch existing pilot evidence or any Git repository. It does not claim that a reconstructed G5 directory constitutes a new/repeated G5 experiment; the restored context remains the exact historical G5 record, with the separately approved runtime amendment still required by the pilot runner.

**The full 1,200-clip feature cache is not in this fallback ZIP and cannot be generated from hashes or the metadata/code ZIPs.** After restoring the nine files, the cell prints how many original validation feature paths exist. It does not manufacture Drive's `.shortcut-targets-by-id` directory or create an alternative cache location. If the cache is absent, it remains an external dependency. If all feature paths exist, exact-cache/preflight verification is still required before the pilot.

Return the printed restoration report and cache count. Do not clear the old pilot directories or automatically rerun a model attempt. C8A/full validation and Day 9/10 remain gated.

## Local verification

223 Member 1 tests passed; 0 failed, 0 skipped. Tests cover matching-file preservation, no writes when any source is missing, refusal to overwrite conflicting files, wrong hashes, symlink/path escapes, unsafe/test archive members and exclusive write races. Both cells parse, all 11 transfer entries were independently checked against source bytes, and user Git index/current originals remain unchanged. Only preparation and clearly labelled synthetic fixture restoration were executed locally.
