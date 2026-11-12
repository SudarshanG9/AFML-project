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
