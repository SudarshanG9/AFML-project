"""Manual C10 package entry point. Delegates inference/audit/packaging unchanged."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import zipfile
from avfuse.member1_fusion.dayc_guards import (ensure_inside_repo,read_strict_json,require_new_file,require_sha256,sha256_file,write_new_json,require_human_approval)
from avfuse.member1_fusion.dayc_freeze import verify_release,verify_registry
from avfuse.member1_fusion.dayc_heldout import execute_heldout,audit_saved_outputs
from scripts.dayC_member1.package_day10_handoff import package
from scripts.dayC_member1.build_day10_package import safe_name


def verify_package(root):
    manifest=root/'PACKAGE_MANIFEST.json'
    if sha256_file(manifest)!=(root/'PACKAGE_MANIFEST.sha256').read_text().strip():raise ValueError('Package manifest changed')
    record=read_strict_json(manifest)
    if record.get('schema_id')!='avfuse-day10-frozen-package/1.0':raise ValueError('Wrong package schema')
    for name,digest in record['files'].items():require_sha256(ensure_inside_repo(root,safe_name(name),must_exist=True),digest)
    return sha256_file(manifest)


def stage_checkpoint(root,launch):
    item=launch['external_inputs']['checkpoint'];target=ensure_inside_repo(root,item['path'])
    if target.exists():require_sha256(target,item['sha256']);return
    source=Path(item['suggested_existing_source'])
    require_sha256(source,item['sha256']);target.parent.mkdir(parents=True,exist_ok=True)
    # User-executed exact-copy staging of model weights only; original is read-only.
    with source.open('rb') as src,target.open('xb') as dest:shutil.copyfileobj(src,dest,1024*1024)
    require_sha256(source,item['sha256']);require_sha256(target,item['sha256'])


def metadata_preflight(root):
    verify_package(root);launch=read_strict_json(root/'DAY10_LAUNCH.json')
    policy,run,anchor=[ensure_inside_repo(root,launch[k],must_exist=True) for k in ('policy','run','anchor')]
    result=verify_release(root,policy,run,anchor,dry_run=True)
    stage_checkpoint(root,launch)
    frozen=read_strict_json(run);verify_registry(root,frozen['inputs'])
    from scripts.dayC_member1.run_day8 import observed_runtime
    from scripts.dayC_member1.colab_query_pilot import check_environment
    check_environment(observed_runtime(),frozen['runtime'])
    return launch,frozen,result


def write_review(root,run_dir,launch_dir):
    """Return all execution evidence, including failures; never includes caches/labels/weights."""
    destination=launch_dir/'DAYC_C10_review.zip';require_new_file(destination)
    payload={}
    if run_dir.exists():
        for path in sorted(run_dir.rglob('*')):
            if path.is_file():
                if path.is_symlink() or path.suffix in {'.npz','.pt','.pth','.mp4','.wav'}:raise ValueError('Unexpected run payload')
                payload['run/'+str(path.relative_to(run_dir))]=path.read_bytes()
    day10=root/'artifacts/dayC/member1/day10'
    for name in ('test_attempt.lock','test_attempt_ledger.jsonl'):
        path=day10/name
        if path.exists():payload['ledger/'+name]=path.read_bytes()
    receipt=root/'artifacts/dayC/member1/approvals/C10_ack.json'
    if receipt.is_file():payload['approval/C10_ack.json']=receipt.read_bytes()
    for path in launch_dir.iterdir():
        if path.is_file() and path.suffix=='.json':payload['launch/'+path.name]=path.read_bytes()
    manifest={name:hashlib.sha256(data).hexdigest() for name,data in payload.items()}
    payload['output_hashes.json']=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
    with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name,data in payload.items():z.writestr(name,data)
    with zipfile.ZipFile(destination) as z:
        if z.testzip() is not None:raise ValueError('Review ZIP damaged')
        for name,data in payload.items():
            if z.read(name)!=data:raise ValueError('Review ZIP differs')
    return destination


def run_authorized(root,approval,archive):
    # Validate package + actual C10 before any attempt, checkpoint staging or test rows.
    manifest_sha=verify_package(root);launch=read_strict_json(root/'DAY10_LAUNCH.json')
    policy,run,anchor=[ensure_inside_repo(root,launch[k],must_exist=True) for k in ('policy','run','anchor')]
    context=verify_release(root,policy,run,anchor,dry_run=True)['C10_context_sha256']
    require_human_approval(approval,'C10',context_sha256=context)
    record=read_strict_json(approval)
    if record.get('execution_package_manifest_sha256')!=manifest_sha or record.get('execution_package_sha256')!=sha256_file(archive):
        raise PermissionError('C10 must acknowledge the exact execution package and manifest hashes')
    # Copy only the actual approved receipt into the project, preserving its exact bytes.
    internal=root/'artifacts/dayC/member1/approvals/C10_ack.json';internal.parent.mkdir(parents=True,exist_ok=True)
    require_new_file(internal)
    with internal.open('xb') as stream:stream.write(Path(approval).read_bytes())
    run_id='C10_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    launch_dir=root/'artifacts/dayC/member1/colab'/run_id;launch_dir.mkdir(parents=True,exist_ok=False)
    run_dir=root/'artifacts/dayC/member1/day10'/run_id
    try:
        _,frozen,_=metadata_preflight(root);registry=frozen['inputs']
        args=SimpleNamespace(repo_root=root,run_id=run_id,out_dir=run_dir,device='cpu',
            canonical_manifest=root/registry['canonical_manifest']['path'],ready_test_manifest=root/registry['ready_test_manifest']['path'],
            upstream=(root/registry['model_source']['path']).parent.parent,frozen_policy=policy,frozen_run=run,
            freeze_anchor=anchor,approval_file=internal,member2_release=root/registry['member2_release']['path'],member3_release=root/registry['member3_release']['path'])
        execute_heldout(args)
        audit=audit_saved_outputs(root,run_dir,policy,run,anchor,internal)
        write_new_json(run_dir/'reports/saved_output_audit.json',audit)
        handoff=package(root,run_dir,policy,run,anchor,internal)
        write_new_json(launch_dir/'completion.json',{'status':'COMPLETE','run_id':run_id,'handoff_sha256':sha256_file(handoff)})
        return 0
    except BaseException as exc:
        write_new_json(launch_dir/'failure.json',{'status':'FAILED','run_id':run_id,'error':f'{type(exc).__name__}: {exc}','automatic_retry_permitted':False})
        print('STOP:',type(exc).__name__,str(exc),flush=True)
        return 1
    finally:
        review=write_review(root,run_dir,launch_dir)
        print('RETURN REVIEW ZIP: '+str(review),flush=True)
        print('STOP FOR C10R REVIEW. No automatic retry, tuning or second run.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root',type=Path,required=True);p.add_argument('--action',choices=('prepare','run'),required=True)
    p.add_argument('--approval-file',type=Path);p.add_argument('--package-archive',type=Path)
    a=p.parse_args();root=a.repo_root.resolve(strict=True)
    if root!=Path(__file__).resolve().parents[2] or not a.repo_root.is_absolute():raise PermissionError('Wrong execution project')
    if a.action=='prepare':
        _,_,result=metadata_preflight(root)
        print(json.dumps({'status':'METADATA_PREFLIGHT_PASS','C10_context_sha256':result['C10_context_sha256'],
            'package_manifest_sha256':sha256_file(root/'PACKAGE_MANIFEST.json'),'test_contents_inspected':False,'inference_executed':False},indent=2))
    else:
        if not a.approval_file or not a.package_archive:raise PermissionError('Actual C10 receipt and original package ZIP required')
        raise SystemExit(run_authorized(root,a.approval_file,a.package_archive))


if __name__=='__main__':main()
