"""Verify frozen release metadata; pre-C10 dry run never opens test rows/features."""
import argparse
from pathlib import Path
from avfuse.member1_fusion.dayc_freeze import verify_release
from avfuse.member1_fusion.dayc_guards import ensure_inside_repo,write_new_json,require_new_file


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('repo-root','frozen-policy','frozen-run','freeze-anchor','out'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('approval-file','member2-release','member3-release'):p.add_argument('--'+name,type=Path)
    p.add_argument('--dry-run',action='store_true');a=p.parse_args();root=a.repo_root.resolve(strict=True)
    if root!=Path(__file__).resolve().parents[2] or not a.repo_root.is_absolute():raise PermissionError('Wrong project root')
    output=ensure_inside_repo(root,a.out);output.relative_to(root/'artifacts/dayC/member1');require_new_file(output)
    try:
        report=verify_release(root,a.frozen_policy,a.frozen_run,a.freeze_anchor,a.approval_file,a.member2_release,a.member3_release,dry_run=a.dry_run)
        report={k:v for k,v in report.items() if k not in ('policy','run')}
    except (OSError,ValueError) as exc:
        report={'status':'BLOCKED','error':f'{type(exc).__name__}: {exc}','test_rows_read':False,'test_features_read':False}
    output.parent.mkdir(parents=True,exist_ok=True);write_new_json(output,report);print(report['status'])
    if report['status']!='READY':raise SystemExit(2)


if __name__=='__main__':main()
