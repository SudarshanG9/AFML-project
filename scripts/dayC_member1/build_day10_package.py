"""Build the C9-frozen manual Colab handoff. No test rows are parsed."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile
from avfuse.member1_fusion.dayc_guards import (ensure_inside_repo,read_strict_json,require_new_file,require_sha256,sha256_file)
from avfuse.member1_fusion.dayc_freeze import (SCORING_FILES,verify_release,verify_registry,validate_complete_proposal)

NAME='AFML_DayC_Day10_Frozen'
BASE='artifacts/dayC/member1/colab/AFML_DayC_C8_2_CPU_Pilot_Mapped_v3_Verified.zip'
BASE_SHA='8cb453863ffad1f688318729ca158497b04e2b15161f752faf1521f6a09bc33a'
ORIGINAL_CHECKPOINT='/content/g5_results/G5_CPU_20261111T181935Z_422cf72f/runtime/dimodif/ckpt/dfd/dfd_lavdf_reduceonplateau_64_4_1_5_True.pth'


def safe_name(name):
    path=PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or str(path)!=name or not path.parts:
        raise ValueError('Unsafe package path')
    return name


def collect_payload(root,frozen,policy,run,anchor):
    """Reuse the hash-pinned pilot support files; overlay exact frozen inputs."""
    root=Path(root);require_sha256(root/BASE,BASE_SHA)
    with zipfile.ZipFile(root/BASE) as z:
        prefix='AFML_DayC_C8_2_CPU_Pilot_Mapped_v3/'
        payload={safe_name(n[len(prefix):]):z.read(n) for n in z.namelist()
            if n.startswith(prefix) and not n.endswith('/') and
            not n.endswith(('PACKAGE_MANIFEST.json','PACKAGE_MANIFEST.sha256','README.md'))}
    registry=frozen['inputs']
    for name,item in registry.items():
        if name=='checkpoint':continue
        path=ensure_inside_repo(root,item['path'],must_exist=True);require_sha256(path,item['sha256'])
        relative=safe_name(str(path.relative_to(root)))
        if path.suffix in {'.pth','.pt','.npz','.mp4','.wav'}:raise PermissionError('Binary/data asset cannot enter execution ZIP')
        payload[relative]=path.read_bytes()
    for path in (policy,run,anchor,root/frozen['c9_approval']['path'],root/frozen['source_proposal']['path'],root/'configs/dayC/member1/C8A_context.json'):
        path=ensure_inside_repo(root,path,must_exist=True);payload[safe_name(str(path.relative_to(root)))]=path.read_bytes()
    # Parent initializers are part of the pinned support package, never invented.
    for name in SCORING_FILES:
        if name not in payload:raise ValueError('Frozen execution source absent: '+name)
    checkpoint=registry['checkpoint']
    launch={'schema_id':'avfuse-day10-colab-launch/1.0',
        'policy':str(Path(policy).relative_to(root)),'run':str(Path(run).relative_to(root)),
        'anchor':str(Path(anchor).relative_to(root)),
        'external_inputs':{'checkpoint':dict(checkpoint,suggested_existing_source=ORIGINAL_CHECKPOINT)},
        'commands':['prepare (metadata/model hash/runtime checks only)','obtain actual C10 bound to frozen context and package hashes','run (existing one-attempt runner, audit, local handoff, review ZIP)'],
        'no_mount_install_or_download':True,'no_automatic_retry':True}
    payload['DAY10_LAUNCH.json']=(json.dumps(launch,sort_keys=True,indent=2)+'\n').encode()
    return payload


def write_archive(destination,payload):
    require_new_file(destination)
    for name in payload:safe_name(name)
    manifest={'schema_id':'avfuse-day10-frozen-package/1.0',
        'files':{n:hashlib.sha256(b).hexdigest() for n,b in sorted(payload.items())}}
    manifest_bytes=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
    all_files=dict(payload,**{'PACKAGE_MANIFEST.json':manifest_bytes,
        'PACKAGE_MANIFEST.sha256':(hashlib.sha256(manifest_bytes).hexdigest()+'\n').encode()})
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('xb') as stream:
        with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for name,data in sorted(all_files.items()):z.writestr(NAME+'/'+name,data)
    with zipfile.ZipFile(destination) as z:
        if z.testzip() is not None:raise ValueError('Corrupt execution ZIP')
        for name,data in all_files.items():
            if z.read(NAME+'/'+name)!=data:raise ValueError('Execution ZIP changed')
    return hashlib.sha256(manifest_bytes).hexdigest()


def colab_cell(archive,archive_sha,action):
    # This file is generated beside the ZIP, not inside it (avoids self-hashing).
    return f"""# Run manually in the approved Colab CPU session. No mounts or installs.
from pathlib import Path
import hashlib, os, re, stat, subprocess, sys, zipfile
from google.colab import files
archive=Path('/content')/{archive.name!r}
root=Path('/content')/{NAME!r}
assert hashlib.sha256(archive.read_bytes()).hexdigest()=={archive_sha!r}, 'Wrong execution ZIP'
if {action!r}=='prepare':
    assert not root.exists(), 'Preserve existing package; do not re-extract or overwrite'
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        assert len(names)==len(set(names)) and z.testzip() is None
        assert all(not Path(n).is_absolute() and '..' not in Path(n).parts and Path(n).parts[0]==root.name for n in names)
        assert all(not stat.S_ISLNK(i.external_attr>>16) for i in z.infolist())
        z.extractall('/content')
else:
    assert root.is_dir(), 'Run the prepare cell first'
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
    PYTHONPATH=str(root/'src')+os.pathsep+str(root),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
cmd=[sys.executable,'-B',str(root/'scripts/dayC_member1/colab_day10.py'),
    '--repo-root',str(root),'--action',{action!r}]
if {action!r}=='run':
    approval=Path('/content/C10_approval.json')
    assert approval.is_file(), 'Upload the actual C10 acknowledgement first'
    cmd+=['--approval-file',str(approval),'--package-archive',str(archive)]
lines=[]
with subprocess.Popen(cmd,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1) as process:
    for line in process.stdout:
        print(line,end='',flush=True);lines.append(line)
    code=process.wait()
print('Exit code:',code)
for match in re.findall(r'^RETURN REVIEW ZIP: (.+)$',''.join(lines),flags=re.MULTILINE):
    review=Path(match).resolve()
    assert review.is_relative_to(root.resolve()) and review.is_file()
    print('REVIEW_SHA256:',hashlib.sha256(review.read_bytes()).hexdigest())
    files.download(str(review))
if {action!r}=='prepare':print('STOP: review the frozen context/package hashes and obtain actual C10 approval. No test access occurred.')
else:print('Return the review ZIP even on failure. No automatic retry or test tuning. STOP for review.')
"""


def build(root,destination,policy,run,anchor):
    root=Path(root).resolve(strict=True)
    policy,run,anchor=[ensure_inside_repo(root,p,must_exist=True) for p in (policy,run,anchor)]
    destination=ensure_inside_repo(root,destination);destination.relative_to(root/'artifacts/dayC/member1/colab')
    # Real C9/anchor verification precedes all payload/data reads and writes.
    metadata=verify_release(root,policy,run,anchor,dry_run=True)
    frozen=read_strict_json(run);registry=verify_registry(root,frozen['inputs'])
    proposal=read_strict_json(root/frozen['source_proposal']['path'])
    validate_complete_proposal(root,proposal,read_strict_json(root/'configs/dayC/member1/C8A_context.json'))
    for action in ('prepare','run'):require_new_file(destination.with_name(destination.stem+'_'+action+'.py'))
    payload=collect_payload(root,frozen,policy,run,anchor)
    manifest_sha=write_archive(destination,payload);archive_sha=sha256_file(destination)
    for action in ('prepare','run'):
        with destination.with_name(destination.stem+'_'+action+'.py').open('x') as stream:
            stream.write(colab_cell(destination,archive_sha,action))
    return {'archive':str(destination),'archive_sha256':archive_sha,'package_manifest_sha256':manifest_sha,
        'C10_context_sha256':metadata['C10_context_sha256'],
        'C10_requires_additional_fields':{'execution_package_sha256':archive_sha,'execution_package_manifest_sha256':manifest_sha},
        'prepare_cell':str(destination.with_name(destination.stem+'_prepare.py')),
        'run_cell':str(destination.with_name(destination.stem+'_run.py')),
        'scope':'C9-frozen; prepare does not access test contents; run requires actual C10'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('repo-root','out','frozen-policy','frozen-run','freeze-anchor'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();root=a.repo_root.resolve(strict=True)
    if root!=Path(__file__).resolve().parents[2]:raise PermissionError('Wrong project root')
    print(json.dumps(build(root,a.out,a.frozen_policy,a.frozen_run,a.freeze_anchor),indent=2))


if __name__=='__main__':main()
