"""One frozen full four-mode test invocation; source preparation only until actual C10."""
import argparse
from pathlib import Path
from avfuse.member1_fusion.dayc_heldout import execute_heldout


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('repo-root','upstream','canonical-manifest','ready-test-manifest','frozen-policy','frozen-run',
                 'freeze-anchor','approval-file','member2-release','member3-release','out-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--run-id',required=True);p.add_argument('--device',required=True)
    a=p.parse_args()
    if not a.repo_root.is_absolute() or a.repo_root.resolve()!=Path(__file__).resolve().parents[2]:
        raise PermissionError('Wrong project root')
    execute_heldout(a)


if __name__=='__main__':main()
