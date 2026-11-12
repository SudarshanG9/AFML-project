"""Build B2 manual-execution package without copying checkpoint/cache/test inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from avfuse.member1_fusion.dayc_guards import (
    ensure_inside_repo,read_strict_json,require_human_approval,require_new_file,require_sha256,sha256_file)

NAME='AFML_DayC_B2_Full_Validation'
NEW_FILES=(
    'src/avfuse/member1_fusion/dayc_validation.py','src/avfuse/member1_fusion/dayc_research.py',
    'scripts/dayC_member1/run_day8.py','scripts/dayC_member1/day8_child.py',
    'scripts/dayC_member1/colab_day8.py','scripts/dayC_member1/review_day8.py')


def build(root,destination,approval):
    root=Path(root).resolve(strict=True);destination=ensure_inside_repo(root,destination)
    destination.relative_to(root/'artifacts/dayC/member1/colab');require_new_file(destination)
    context_path=root/'configs/dayC/member1/C8A_context.json';context=read_strict_json(context_path)
    approval=ensure_inside_repo(root,approval,must_exist=True)
    require_human_approval(approval,'C8A',context_sha256=sha256_file(context_path))
    require_sha256(root/context['review_report'],context['review_report_sha256'])
    source_zip=root/'artifacts/dayC/member1/colab/AFML_DayC_C8_2_CPU_Pilot_Mapped_v3_Verified.zip'
    require_sha256(source_zip,'8cb453863ffad1f688318729ca158497b04e2b15161f752faf1521f6a09bc33a')
    with zipfile.ZipFile(source_zip) as z:
        prefix=context['pilot_release']['package_name']+'/'
        payload={n[len(prefix):]:z.read(n) for n in z.namelist() if not n.endswith(('PACKAGE_MANIFEST.json','PACKAGE_MANIFEST.sha256'))}
    for name,expected in context['pilot_source_sha256'].items():
        require_sha256(root/name,expected)
        if hashlib.sha256(payload[name]).hexdigest()!=expected:raise ValueError('Pilot source differs')
    spec=context['experiment_spec']
    for name in NEW_FILES:
        payload[name]=ensure_inside_repo(root,name,must_exist=True).read_bytes()
    frozen=read_strict_json(root/'configs/dayB/frozen_g5_package_manifest.json')
    expected={r['path']:r['sha256'] for r in frozen['files']}
    extra=['scripts/dayB_model_benchmark.py','contracts/schemas/benchmark.schema.json','contracts/schemas/benchmark_config.schema.json',
           'scripts/m2_check_predictions.py','scripts/m2_evaluate.py','scripts/m2_paired.py']
    for name in extra:
        require_sha256(root/name,expected[name]);payload[name]=(root/name).read_bytes()
    for key in ('predictions_reference','predictions_low_precision','predictions_restore_value_output','calibration'):
        item=spec['local_inputs'][key];path=root/item['path'];require_sha256(path,item['sha256'])
        payload[item['path']]=path.read_bytes()
    report=root/context['review_report']
    audit=report.parent
    payload['assets/accepted_pilot.json']=(audit/'received/pilot.json').read_bytes()
    payload['assets/accepted_pilot_audit.json']=(audit/'audit.json').read_bytes()
    if hashlib.sha256(payload['assets/accepted_pilot_audit.json']).hexdigest()!=context['pilot_audit_sha256']:
        raise ValueError('Accepted pilot audit changed')
    payload['assets/c8a_approval.json']=approval.read_bytes()
    payload['configs/dayC/member1/C8A_context.json']=context_path.read_bytes()
    payload['README.md']=b'B2 C8A-authorized full 1200 validation query run and original short/typical/long four-mode benchmarks (10/50). No installs/mounts/test features/Day9/10. Return DAYC_B2_review.zip even on failure. No retries. Member2/3 evidence review remains required.\n'
    forbidden={'.pth','.pt','.npz','.mp4','.wav'}
    if any(Path(n).suffix in forbidden for n in payload):raise PermissionError('Forbidden payload')
    manifest={'schema_id':'avfuse-member1-day8-b2-package/1.0','files':{n:hashlib.sha256(b).hexdigest() for n,b in sorted(payload.items())},
              'C8A_context_sha256':sha256_file(context_path),'scope':'B2 validation only; no test assets'}
    manifest_bytes=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
    payload['PACKAGE_MANIFEST.json']=manifest_bytes
    payload['PACKAGE_MANIFEST.sha256']=(hashlib.sha256(manifest_bytes).hexdigest()+'\n').encode()
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('xb') as stream:
        with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for n,b in sorted(payload.items()):z.writestr(NAME+'/'+n,b)
    with zipfile.ZipFile(destination) as z:
        assert z.testzip() is None
        for n,b in payload.items():assert z.read(NAME+'/'+n)==b
    return {'archive':str(destination),'sha256':sha256_file(destination),'bytes':destination.stat().st_size,
            'files':len(payload),'package_name':NAME,'manifest_sha256':hashlib.sha256(manifest_bytes).hexdigest(),
            'context_sha256':sha256_file(context_path),'old_predictions_included_as_original_val_only':True}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('repo-root','out','approval-file'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(build(a.repo_root,a.out,a.approval_file),indent=2))


if __name__=='__main__':main()
