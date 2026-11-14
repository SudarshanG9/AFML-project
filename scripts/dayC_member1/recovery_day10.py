"""Explicit recovery engineering/freeze/package/manual execution interfaces. No automatic approval."""
import argparse,json
from pathlib import Path
from avfuse.member1_fusion import dayc_recovery as r
from avfuse.member1_fusion.dayc_guards import ensure_inside_repo,read_strict_json,write_new_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root',type=Path,required=True)
    p.add_argument('--action',required=True,choices=('propose','finalize','build','prepare','run'))
    for name in ('proof','out','proposal','approval-file','context','anchor','package-archive'):p.add_argument('--'+name,type=Path)
    a=p.parse_args();root=a.repo_root.resolve(strict=True)
    if root!=Path(__file__).resolve().parents[2] or not a.repo_root.is_absolute():raise PermissionError('Wrong recovery project')
    def local(value,existing=True):
        if value is None:raise PermissionError('Missing required argument')
        return ensure_inside_repo(root,value,must_exist=existing)
    if a.action=='propose':result=r.propose(root,local(a.proof),local(a.out,False))
    elif a.action=='finalize':result=r.finalize(root,local(a.proposal),local(a.approval_file),local(a.out,False));result=[str(p) for p in result]
    elif a.action=='build':result=r.build_package(root,local(a.context),local(a.anchor),local(a.out,False))
    else:
        launch=read_strict_json(root/'RECOVERY_LAUNCH.json');context=local(Path(launch['context']));anchor=local(Path(launch['anchor']))
        if a.action=='prepare':result=r.prepare_remote(root,context,anchor)
        else:
            if a.approval_file is None or a.package_archive is None:raise PermissionError('Actual fresh recovery C10 receipt and ZIP required')
            error=None
            try:
                out=r.execute_recovery(root,context,anchor,a.approval_file,a.package_archive)
                audit=r.audit_recovery(root,context,anchor,a.approval_file,a.package_archive)
                write_new_json(out/'saved_output_audit.json',audit)
                frozen,_=r.verify_amendment(root,context,anchor);r.package_handoff(root,out,audit,frozen)
            except BaseException as exc:error=f'{type(exc).__name__}: {exc}'
            review=r.return_review(root,context,anchor,a.approval_file,error)
            print('RETURN REVIEW ZIP: '+str(review));print('STOP. No automatic retry or third attempt.')
            if error:raise SystemExit(error)
            return
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
