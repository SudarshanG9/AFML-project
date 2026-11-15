"""Read saved authorized test outputs without detector invocation or metrics."""
import argparse
from pathlib import Path
from avfuse.member1_fusion.dayc_heldout import audit_saved_outputs
from avfuse.member1_fusion.dayc_guards import ensure_inside_repo,write_new_json,require_new_file


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('repo-root','run-dir','frozen-policy','frozen-run','freeze-anchor','approval-file','out'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();root=a.repo_root.resolve(strict=True)
    if root!=Path(__file__).resolve().parents[2]:raise PermissionError('Wrong project root')
    out=ensure_inside_repo(root,a.out);out.relative_to(root/'artifacts/dayC/member1/day10');require_new_file(out)
    result=audit_saved_outputs(root,a.run_dir,a.frozen_policy,a.frozen_run,a.freeze_anchor,a.approval_file)
    out.parent.mkdir(parents=True,exist_ok=True);write_new_json(out,result)


if __name__=='__main__':main()
