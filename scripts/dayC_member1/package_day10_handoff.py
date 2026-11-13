"""Local whitelisted score handoff; no weights/caches/labels/private path payload."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from avfuse.member1_fusion.dayc_heldout import audit_saved_outputs
from avfuse.member1_fusion.dayc_guards import ensure_inside_repo,require_new_file,read_strict_json,sha256_file
from avfuse.member1_fusion.dayc_validation import MODES


def build_public_payload(run_dir,audit,frozen):
    if audit.get('status')!='READY_FOR_MEMBER2_SCORING':raise PermissionError('Saved-output audit must pass')
    payload={};reference_ids=None
    for mode in MODES:
        path=Path(run_dir)/'predictions'/(mode+'_test.jsonl')
        if sha256_file(path)!=audit['prediction_sha256'][mode]:raise ValueError('Prediction changed since audit')
        from avfuse.member2_data.predictions import read_predictions
        from avfuse.member1_fusion.dayc_heldout import validate_saved_rows
        rows=read_predictions(path);ids={r['clip_id'] for r in rows}
        validate_saved_rows(rows,mode=mode,ids=ids,checkpoint=frozen['checkpoint_sha256'],run_id=audit['run_id'])
        if reference_ids is None:reference_ids=ids
        elif ids!=reference_ids:raise ValueError('Handoff mode ID sets differ')
        if len(ids)!=audit['count_per_mode']:raise ValueError('Handoff cohort differs')
        payload['predictions/'+path.name]=path.read_bytes()
    from avfuse.member1_fusion.dayc_precision_audit import expected_mode_boundaries
    public={'run_id':audit['run_id'],'count_per_mode':audit['count_per_mode'],'four_modes':list(MODES),
        'checkpoint_sha256':frozen['checkpoint_sha256'],'threshold':frozen['fixed_threshold'],
        'canonical_manifest_sha256':frozen['inputs']['canonical_manifest']['sha256'],
        'protocol_sha256':frozen['inputs']['calibration']['sha256'],
        'member2_release_sha256':frozen['inputs']['member2_release']['sha256'],
        'member3_release_sha256':frozen['inputs']['member3_release']['sha256'],
        'mode_definitions':{m:expected_mode_boundaries(m) for m in MODES},'layers':5,'backend':'sdpa',
        'precision':'floating-point fake INT8; no custom/integer kernel',
        'source_code_sha256':{k.removeprefix('source:'):v['sha256'] for k,v in frozen['inputs'].items() if k.startswith('source:')},
        'known_test_label_summary_access':True,'test_metrics_computed_by_member1':False}
    payload['frozen_public_references.json']=(json.dumps(public,sort_keys=True,indent=2)+'\n').encode()
    payload['MEMBER1_TO_MEMBER2_DAY10.md']=b'Four complete frozen test prediction files. Member 2 evaluates once with its signed seal, original fixed threshold, source-group bootstrap 1000 seed0. No recalibration or post-test policy tuning. No Member 2 evaluation has been executed by Member 1. Source grouping is not guaranteed speaker separation; historical test-label summaries were inspected.\n'
    payload['MEMBER1_TO_MEMBER3_DAY10.md']=b'Frozen four-mode input/score identity only. Member 3 owns final operator and cached-model p50/p95/memory, conversion overhead and raw samples under its signed plan; preprocessing separately. Fake INT8 is floating-point emulation, not integer acceleration. No benchmark or memory result is invented in this handoff.\n'
    return payload


def add_accepted_query_validation(root,payload,frozen):
    """Reuse the C8B-bound bytes; never regenerate validation predictions."""
    item=frozen['inputs']['predictions_restore_query']
    path=ensure_inside_repo(root,item['path'],must_exist=True)
    if sha256_file(path)!=item['sha256']:raise ValueError('Accepted query validation changed')
    from avfuse.member2_data.predictions import read_predictions
    rows=read_predictions(path)
    if len(rows)!=1200 or len({r['clip_id'] for r in rows})!=1200:raise ValueError('Expected accepted 1200 validation scores')
    allowed={'run_id','clip_id','split','config','score_fake','checkpoint_sha256'}
    if any(set(r)!=allowed or r['split']!='val' or r['config']!='restore_query' or
        r['checkpoint_sha256']!=frozen['checkpoint_sha256'] for r in rows):raise ValueError('Invalid validation handoff identity/schema')
    payload['predictions/restore_query_val.jsonl']=path.read_bytes()


def package(root,run_dir,policy,run,anchor,approval):
    root=Path(root).resolve(strict=True);run_dir=ensure_inside_repo(root,run_dir,must_exist=True)
    audit=audit_saved_outputs(root,run_dir,policy,run,anchor,approval)
    frozen=read_strict_json(ensure_inside_repo(root,run,must_exist=True))
    payload=build_public_payload(run_dir,audit,frozen)
    add_accepted_query_validation(root,payload,frozen)
    payload['saved_output_audit.json']=(json.dumps(audit,sort_keys=True,indent=2)+'\n').encode()
    seal_hashes=read_strict_json(run_dir/'provenance/freeze_and_approval_hashes.json')
    required={'frozen_policy_sha256','frozen_run_sha256','freeze_anchor_sha256','approved_c10_ack_sha256'}
    if set(seal_hashes)!=required:raise ValueError('Unexpected seal reference fields')
    seal_evidence={'release_hashes':seal_hashes,'attempts_opened':audit['attempts_opened'],
        'attempts_completed':audit['attempts_completed'],'run_id':audit['run_id'],
        'known_test_label_summary_access':True,'test_metrics_computed_by_member1':False}
    payload['sealed_test_execution_evidence.json']=(json.dumps(seal_evidence,sort_keys=True,indent=2)+'\n').encode()
    destination=run_dir/'handoff';require_new_file(destination);destination.mkdir()
    inventory={n:hashlib.sha256(b).hexdigest() for n,b in payload.items()}
    payload['handoff_manifest.json']=(json.dumps({'status':'MEMBER1_DONE_TEAM_SCORING_PENDING','files':inventory},sort_keys=True,indent=2)+'\n').encode()
    for name,data in payload.items():
        path=destination/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:stream.write(data)
    archive=destination/'member1_day10_handoff.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name,data in payload.items():z.writestr(name,data)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise RuntimeError('Handoff archive damaged')
        for name,data in payload.items():
            if z.read(name)!=data:raise RuntimeError('Handoff bytes differ')
    return archive


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('repo-root','run-dir','frozen-policy','frozen-run','freeze-anchor','approval-file'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();root=a.repo_root.resolve(strict=True)
    if root!=Path(__file__).resolve().parents[2]:raise PermissionError('Wrong project root')
    print(package(root,a.run_dir,a.frozen_policy,a.frozen_run,a.freeze_anchor,a.approval_file))


if __name__=='__main__':main()
