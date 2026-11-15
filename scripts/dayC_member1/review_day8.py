"""Read owner-supplied B2 evidence and write a non-overwriting research packet."""
import argparse
from pathlib import Path
from avfuse.member1_fusion.dayc_guards import ensure_inside_repo,require_new_file,write_new_json
from avfuse.member1_fusion.dayc_research import (
    read_four_mode_validation_evidence,build_h2_h3_decision_table,describe_limitations)
from avfuse.member1_fusion.dayc_validation import MODES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('repo-root','metric-json','paired-json','benchmark-jsonl','canonical-manifest','evidence-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--prediction-paths',type=Path,nargs=4,required=True,help='reference low_precision restore_value_output restore_query')
    a=p.parse_args();root=a.repo_root.resolve(strict=True)
    from scripts.dayC_member1.run_day8 import load_release
    approval=root/'assets/c8a_approval.json'
    if not approval.exists():
        candidates=sorted((root/'artifacts/dayC/member1/approvals').glob('C8A_ack_*.json'))
        if not candidates:raise PermissionError('Missing C8A')
        approval=candidates[-1]
    load_release(root,approval)
    paths=[a.metric_json,a.paired_json,a.benchmark_jsonl,a.canonical_manifest,*a.prediction_paths]
    for path in paths:ensure_inside_repo(root,path,must_exist=True)
    destination=ensure_inside_repo(root,a.evidence_dir)
    destination.relative_to(root/'artifacts/dayC/member1/day8');require_new_file(destination)
    evidence=read_four_mode_validation_evidence(metric_json=a.metric_json,paired_json=a.paired_json,
        prediction_paths=dict(zip(MODES,a.prediction_paths)),benchmark_jsonl=a.benchmark_jsonl,canonical_manifest=a.canonical_manifest)
    table=build_h2_h3_decision_table(evidence);limits=describe_limitations(evidence)
    destination.mkdir(parents=True,exist_ok=False)
    write_new_json(destination/'day8_review_packet.json',{'evidence':evidence,'decision_table':table,'limitations':limits,
        'C8B_approval_received':False,'policy_frozen':False})
    lines=['# Day 8 research review',str(evidence['status']),'','No policy selection or freeze.',
           '',str(table),'','Limitations:']+['- '+s for s in limits]
    with (destination/'day8_review_packet.md').open('x') as stream:stream.write('\n'.join(lines)+'\n')


if __name__=='__main__':main()
