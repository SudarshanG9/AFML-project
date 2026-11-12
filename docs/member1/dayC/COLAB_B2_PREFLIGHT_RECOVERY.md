# B2 exact-input recovery and continuation — one manual cell

The returned `DAYC_B2_review.zip` (SHA-256 `6eb1bf4e6abb5870371ce69f53e607040f91368c63b00d50f986bb25be9b80b3`) contains only a hash-valid failure record and output hash list. It stopped during `full_validation_preflight` because `/content/AFML_G5_CPU_Colab_20261111T180139Z/data/manifests/manifest_whole.jsonl` was missing. No query predictions or benchmarks were produced. The verified B2 runner checks this before launching a model subprocess. This is an external asset-readiness blocker within B2, not its block review.

C8A is still accepted. No new research ACK is requested. The cell below separately confirms no full-validation attempt marker, execution lock or query output exists before it permits continuation. If any exists, it stops and preserves the evidence. It never automatically repeats a model attempt.

## Files to have in /content

1. Existing unchanged [B2 release ZIP](/Users/sudarshan/Documents/GitHub/AFML-project/artifacts/dayC/member1/colab/AFML_DayC_B2_C8A_Release.zip) — SHA-256 `dbd44345b383645b12f6a03dae7778165ec0f82df6018dfec4cd27e7706b0882`. Keep it if already uploaded; otherwise upload it manually.
2. [Verified Original Inputs ZIP](/Users/sudarshan/Documents/GitHub/AFML-project/artifacts/dayC/member1/colab/AFML_DayC_Verified_Original_Inputs.zip) — SHA-256 `686a3678708d30c283677c5c8c0af715d96c464f18ec461811eafbe5b1555463`. Upload this exact backup if absent. It contains the original canonical/validation/ready manifests, accepted G4/G5 evidence and exact approved model assets. These are preserved original bytes, not rebuilt data.

Leave the approved Drive mount connected with `/content/afml_drive/MyDrive/AFML_Project/dimodif_emb/val` accessible. The cell does not connect to services, mount Drive, install packages, download data or touch the test directory. Original backup includes canonical manifest bytes, which are copied as opaque bytes during restoration; no canonical rows are parsed by the restoration helper. The later unchanged B2 preflight performs its already-approved metadata split audit.

The existing B2 extraction and failed evidence are reused only after matching every original package member and the exact Python source set against the pinned release ZIP. Existing matching inputs are preserved; conflicting originals stop without overwrite. All eight required original sources and destinations are verified before any missing input is restored to its historically approved path. The original ready manifest retains its obsolete root and the existing read-only mapping remains unchanged. Feature files are never created or rebuilt.

## Copy this entire cell once

```python
"""Manual Colab preflight recovery; exact originals only, no model-attempt retry."""
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from datetime import datetime,timezone
import uuid
import zipfile

B2_SHA='dbd44345b383645b12f6a03dae7778165ec0f82df6018dfec4cd27e7706b0882'
BACKUP_SHA='686a3678708d30c283677c5c8c0af715d96c464f18ec461811eafbe5b1555463'
FAILED_RETURN_SHA='6eb1bf4e6abb5870371ce69f53e607040f91368c63b00d50f986bb25be9b80b3'


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):value.update(chunk)
    return value.hexdigest()


def verify_or_extract_package(archive,addon,expected_sha):
    archive=Path(archive);addon=Path(addon)
    if digest(archive)!=expected_sha:raise ValueError('B2 release ZIP checksum differs')
    with zipfile.ZipFile(archive) as z:
        infos=z.infolist();names=z.namelist()
        if z.testzip() is not None or len(names)!=len(set(names)):raise ValueError('Invalid archive')
        for info in infos:
            name=Path(info.filename)
            if (name.is_absolute() or '..' in name.parts or name.parts[0]!=addon.name
                    or stat.S_ISLNK(info.external_attr>>16)):
                raise ValueError('Unsafe archive member')
        if not os.path.lexists(addon):
            z.extractall(addon.parent)
        if addon.is_symlink() or not addon.is_dir():raise ValueError('Unsafe package directory')
        for info in infos:
            path=addon.parent/info.filename
            if (path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(addon.resolve())
                    or digest(path)!=hashlib.sha256(z.read(info)).hexdigest()):
                raise ValueError('Existing B2 file differs; preserve and stop: '+str(path))
        actual={str(p.relative_to(addon)) for base in ('src','scripts') for p in (addon/base).rglob('*.py')}
        expected={str(Path(n).relative_to(addon.name)) for n in names if n.endswith('.py')}
        if actual!=expected:raise ValueError('B2 Python source set changed')


def assert_no_model_attempt(addon):
    addon=Path(addon);day8=addon/'artifacts/dayC/member1/day8'
    forbidden=(addon/'artifacts/predictions/restore_query_val.jsonl',day8/'full_validation_execution.lock')
    if any(os.path.lexists(p) for p in forbidden) or any(day8.rglob('*attempt_started.json')):
        raise RuntimeError('A model attempt/output already exists. Preserve it; do not resume automatically.')


def verify_bundle(path):
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        if z.testzip() is not None or len(names)!=len(set(names)):raise ValueError('Invalid return ZIP')
        hashes=json.loads(z.read('output_hashes.json'))
        if set(names)!=set(hashes)|{'output_hashes.json'}:raise ValueError('Return member set differs')
        for name,expected in hashes.items():
            if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Unsafe return path')
            if hashlib.sha256(z.read(name)).hexdigest()!=expected:raise ValueError('Return hash mismatch')
        return {n:z.read(n) for n in names}


def bundle_recovery(original,receipt,destination):
    payload=verify_bundle(original)
    payload['recovery/original_output_hashes.json']=payload.pop('output_hashes.json')
    payload['recovery/exact_input_restoration.json']=(json.dumps(receipt,sort_keys=True,indent=2)+'\n').encode()
    hashes={n:hashlib.sha256(b).hexdigest() for n,b in payload.items()}
    payload['output_hashes.json']=(json.dumps(hashes,sort_keys=True,indent=2)+'\n').encode()
    with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name,value in payload.items():z.writestr(name,value)
    verify_bundle(destination)


def run_manual():
    from google.colab import files
    content=Path('/content');addon=content/'AFML_DayC_B2_Full_Validation'
    archive=content/'AFML_DayC_B2_C8A_Release.zip'
    backup=content/'AFML_DayC_Verified_Original_Inputs.zip'
    assert archive.is_file(), 'Upload the original B2 C8A Release ZIP into /content'
    assert backup.is_file(), 'Upload AFML_DayC_Verified_Original_Inputs.zip into /content'
    assert digest(backup)==BACKUP_SHA, 'Original-input backup ZIP checksum differs'
    verify_or_extract_package(archive,addon,B2_SHA)
    assert_no_model_attempt(addon)
    context=json.loads((addon/'configs/dayC/member1/C8A_context.json').read_text())
    release=context['pilot_release']
    assert (Path(release['validation_cache_mapping']['mapped_root'])/'val').is_dir(), (
        'Reconnect the existing approved /content/afml_drive mount first; this cell does not mount/access accounts')
    with zipfile.ZipFile(backup) as z:
        helper=z.read('restore_verified_inputs.py')
    assert hashlib.sha256(helper).hexdigest()=='181c3abf819a95c728fb205ffe88af4b2c8c26c75f0e25ca6a8f74fcb129bf2a'
    namespace={'__name__':'verified_original_restore_helper'}
    exec(compile(helper,'verified_original_restore_helper','exec'),namespace)
    targets=namespace['targets_from_release'](release)
    targets.pop('reference_predictions')  # B2 preserves its three exact packaged baseline copies.
    # All eight sources/destinations are verified before writing; conflicts stop without replacement.
    restoration=namespace['restore_from_backups'](content,targets,[backup])
    assert_no_model_attempt(addon)
    recovery=addon/'artifacts/dayC/member1/day8'/('EXACT_INPUT_RECOVERY_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    recovery.mkdir(parents=True,exist_ok=False)
    receipt={'status':'EXACT_ORIGINALS_VERIFIED_OR_RESTORED','restored_inputs':restoration,
             'backup_zip_sha256':BACKUP_SHA,'B2_zip_sha256':B2_SHA,'prior_failed_return_sha256':FAILED_RETURN_SHA,
             'no_prior_model_attempt_confirmed':True,'original_manifest_paths_and_bytes_preserved':True,
             'feature_cache_not_created_or_modified':True,'test_access':False}
    with (recovery/'restoration_receipt.json').open('x') as f:json.dump(receipt,f,indent=2)
    print('EXACT_REQUIRED_ORIGINALS_READY=8/8',flush=True)
    print('Failed preflight evidence is preserved. Resuming the existing approved B2 arrangement.',flush=True)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        PYTHONPATH=str(addon/'src')+os.pathsep+str(addon),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',MPLBACKEND='Agg',GIT_OPTIONAL_LOCKS='0')
    lines=[]
    with subprocess.Popen([sys.executable,'-B',str(addon/'scripts/dayC_member1/colab_day8.py')],
            cwd=addon,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1) as process:
        for line in process.stdout:
            lines.append(line);print(line,end='',flush=True)
        returncode=process.wait()
    print('Exit code:',returncode)
    matches=re.findall(r'^RETURN REVIEW ZIP: (.+)$',''.join(lines),flags=re.MULTILINE)
    if not matches:
        print('No run ZIP. Return this output; preserve all folders and do not rerun.')
        return
    original=Path(matches[-1].strip()).resolve()
    assert original.is_file() and original.is_relative_to(addon.resolve())
    destination=recovery/'DAYC_B2_review.zip'
    bundle_recovery(original,receipt,destination)
    print('RETURN ZIP:',destination)
    print('RETURN_ZIP_SHA256:',digest(destination))
    print('Return this ZIP even on failure. B2 continues; no C8B or later research gate is implied.')
    files.download(str(destination))


if __name__=='__main__':
    run_manual()
```

The cell executes the unchanged B2 entry point in a fresh subprocess after restoration. It still verifies all 1200 approved validation IDs and all 2400 feature/metadata hashes, retains the original checkpoint/precision/runtime, performs one full query-validation run and all three four-mode 10-warmup/50-repeat benchmarks. Existing old predictions/calibration are already included byte-for-byte in the B2 package and are not overwritten.

Return the newly downloaded **DAYC_B2_review.zip**, even on a new failure. The recovery cell creates a new combined evidence ZIP containing the original B2 outputs plus a hash-bound exact-restoration receipt and the original output hash list. It preserves the original generated ZIP and all prior failed directories unchanged. Any new failure remains evidence for audit, not permission for automatic repetition. If an upfront package/mount/attempt guard stops the cell before a return ZIP exists, return its output and do not delete any directory/lock.

## Local verification and remaining block work

19 focused recovery/restoration tests passed, 0 failed/skipped. These verify safe reuse, refusal on prior model-attempt/output markers, changed/extra/symlink package sources, wrong ZIP hash, conflict preservation, and new-bundle integrity without changing the original ZIP. Existing B2 release sources and all accepted pilot source hashes remain unchanged; source syntax compiled without Colab execution. Earlier full Member 1 suite remains 291 passed for the B2 implementation; no full-suite count is inferred from this focused run.

Audit/evidence: `/Users/sudarshan/Documents/GitHub/AFML-project/artifacts/dayC/member1/day8/B2_PREFLIGHT_RECOVERY_20261112T152728386160Z`. No real model, Colab/Drive access, Git write or test-feature access was performed by the agent. All local additions are confined to the AFML clone. After returned outputs are audited, continue B2’s owner-dependent evaluation/research work. Ask for its one C8B review only when the full block is complete. Member 2/3 requirements remain in `B2_MEMBER2_MEMBER3_HANDOFF_REQUIREMENTS.md`; no new research protocol or ownership change is introduced by recovery.
