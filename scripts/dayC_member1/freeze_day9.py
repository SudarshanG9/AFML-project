"""Draft-only preparation or explicit C9 finalization; never auto-select a policy."""
import argparse
from pathlib import Path
from avfuse.member1_fusion.dayc_guards import (
    ensure_inside_repo,read_strict_json,require_human_approval,require_new_file,sha256_file,write_new_json)
from avfuse.member1_fusion.dayc_freeze import (
    check_preparation_authorization,build_draft,expected_owner_context,verify_registry,finalize_proposal)


def prepare(root,output,*,registry_path=None,evidence_path=None,member2=None,member3=None,
            c8b_approval=None,preparation_authorization=None,interpretation=None):
    root=Path(root).resolve(strict=True)
    context_path=root/'configs/dayC/member1/C8A_context.json';context=read_strict_json(context_path)
    evidence_file=ensure_inside_repo(root,evidence_path,must_exist=True) if evidence_path else None
    accepted=False
    if c8b_approval:
        if evidence_file is None:raise PermissionError('C8B must bind the actual B2 research packet')
        require_human_approval(ensure_inside_repo(root,c8b_approval,must_exist=True),'C8B',context_sha256=sha256_file(evidence_file));accepted=True
    else:
        if preparation_authorization is None:raise PermissionError('Missing C8B or explicit B3 source-preparation instruction')
        record=read_strict_json(ensure_inside_repo(root,preparation_authorization,must_exist=True))
        check_preparation_authorization(record,sha256_file(context_path))
    registry=read_strict_json(ensure_inside_repo(root,registry_path,must_exist=True)) if registry_path else {}
    verify_registry(root,registry,complete=False)
    packet=read_strict_json(evidence_file) if evidence_file else None
    m2=read_strict_json(ensure_inside_repo(root,member2,must_exist=True)) if member2 else None
    m3=read_strict_json(ensure_inside_repo(root,member3,must_exist=True)) if member3 else None
    expected=expected_owner_context(root,context,registry)
    proposal=build_draft(context=context,registry=registry,evidence_packet=packet,member2_release=m2,
        member3_release=m3,expected=expected,c8b_accepted=accepted,interpretation=interpretation)
    output=ensure_inside_repo(root,output);output.relative_to(root/'artifacts/dayC/member1/day9')
    require_new_file(output);output.mkdir(parents=True,exist_ok=False)
    digest=write_new_json(output/'policy_proposal.json',proposal)
    write_new_json(output/'freeze_dependency_check.json',{'status':proposal['preparation_state'],
        'blockers':proposal['blockers'],'test_rows_read':False,'test_features_read':False,'freeze_written':False})
    write_new_json(output/'proposed_hash_registry.json',registry)
    text=['# Unapproved Day 9 policy proposal','',proposal['preparation_state'],
        'No final freeze, policy selection or held-out execution is authorized.','',
        'Proposal SHA-256: '+digest,'','Dependencies:']+['- '+b for b in proposal['blockers']]
    with (output/'policy_proposal.md').open('x') as stream:stream.write('\n'.join(text)+'\n')
    return proposal


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root',type=Path,required=True);p.add_argument('--action',choices=('propose','finalize'),required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    for name in ('registry','evidence-packet','member2-release','member3-release','c8b-approval','c9-approval','preparation-authorization','proposal'):
        p.add_argument('--'+name,type=Path)
    p.add_argument('--interpretation',choices=('reference_preferred','no_restoration_justified','restoration_conditionally_supported','inconclusive'))
    a=p.parse_args();root=a.repo_root.resolve(strict=True)
    if not a.repo_root.is_absolute() or root!=Path(__file__).resolve().parents[2]:raise PermissionError('Wrong project root')
    if a.action=='propose':
        result=prepare(root,a.out_dir,registry_path=a.registry,evidence_path=a.evidence_packet,
            member2=a.member2_release,member3=a.member3_release,c8b_approval=a.c8b_approval,
            preparation_authorization=a.preparation_authorization,interpretation=a.interpretation)
        print(result['preparation_state'])
    else:
        if not all((a.proposal,a.c9_approval,a.c8b_approval)):raise PermissionError('Finalization requires actual C8B and proposal-bound C9 records')
        result=finalize_proposal(root,a.proposal,a.c9_approval,a.c8b_approval,a.out_dir,root/'configs/dayC/member1/C8A_context.json')
        print(result['status'])


if __name__=='__main__':main()
