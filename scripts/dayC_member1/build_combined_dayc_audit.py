"""Build the source/score/metadata-only joint B2/B3 audit capsule; never run a model."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from avfuse.member1_fusion.dayc_guards import (
    ensure_inside_repo,read_strict_json,require_new_file,require_sha256,sha256_file)

NAME='AFML_DayC_B2_B3_Combined_Audit'


def build(root,output,*,input_spec,authorization,source_verification):
    root=Path(root).resolve(strict=True);output=ensure_inside_repo(root,output)
    output.relative_to(root/'artifacts/dayC/member1/colab');require_new_file(output)
    input_spec=ensure_inside_repo(root,input_spec,must_exist=True)
    authorization=ensure_inside_repo(root,authorization,must_exist=True)
    source_verification=ensure_inside_repo(root,source_verification,must_exist=True)
    spec=read_strict_json(input_spec);context=read_strict_json(root/'configs/dayC/member1/C8A_context.json')
    from avfuse.member1_fusion.dayc_freeze import check_preparation_authorization,SCORING_FILES
    check_preparation_authorization(read_strict_json(authorization),sha256_file(root/'configs/dayC/member1/C8A_context.json'))
    report=read_strict_json(source_verification)
    if report['status']!='PASS' or report['failed'] or report['errors'] or report['skipped']:raise ValueError('Source tests must pass')
    for name,h in report['source_sha256'].items():require_sha256(root/name,h)
    original=root/'artifacts/dayC/member1/colab/AFML_DayC_B2_C8A_Release.zip'
    require_sha256(original,'dbd44345b383645b12f6a03dae7778165ec0f82df6018dfec4cd27e7706b0882')
    payload={}
    with zipfile.ZipFile(original) as z:
        prefix='AFML_DayC_B2_Full_Validation/'
        for name in z.namelist():
            relative=name.removeprefix(prefix)
            if relative==name:raise ValueError('Unexpected original package prefix')
            payload['execution_project/'+relative]=z.read(name)
    logical_root=Path(spec['original_root']);bindings={}
    received=root/'artifacts/dayC/member1/day8/B2_RETURN_AUDIT_20261112T160215000947Z/received'
    for logical,expected in spec['inputs'].items():
        if logical==context['pilot_release']['canonical_manifest']:
            relative=Path(context['experiment_spec']['local_inputs']['canonical_manifest']['path'])
            source=root/relative
        else:
            relative=Path(logical).relative_to(logical_root)
            if relative.name in {'four_modes_short.jsonl','four_modes_typical.jsonl','four_modes_long.jsonl'}:
                source=received/'benchmarks'/relative.name
            else:source=root/relative
        source=ensure_inside_repo(root,source,must_exist=True);require_sha256(source,expected)
        target='execution_project/'+str(relative)
        content=source.read_bytes()
        if target in payload and payload[target]!=content:raise ValueError('Original package input bytes differ')
        payload[target]=content;bindings[logical]=target
    # Retain exact original validation-ready metadata as opaque copies, never rewrite feature_path values.
    for name in ('validation_manifest','ready_val_manifest','dayb_context'):
        item=context['experiment_spec']['local_inputs'][name];source=root/item['path'];require_sha256(source,item['sha256'])
        payload['execution_project/'+item['path']]=source.read_bytes()
    spec.update(prepare_B3_source_draft=True,execution_project_relative='execution_project',input_path_bindings=bindings,
                scope='saved validation evaluation and unapproved B3 draft only; no inference or test feature access')
    payload.update({
        'EVALUATION_INPUTS.json':(json.dumps(spec,sort_keys=True,indent=2)+'\n').encode(),
        'finish_day8_evaluation.py':(root/'scripts/dayC_member1/finish_day8_evaluation.py').read_bytes(),
        'src/avfuse/member1_fusion/dayc_research.py':(root/'src/avfuse/member1_fusion/dayc_research.py').read_bytes(),
        'src/avfuse/member1_fusion/dayc_freeze.py':(root/'src/avfuse/member1_fusion/dayc_freeze.py').read_bytes(),
        'configs/dayC/member1/C8A_context.json':(root/'configs/dayC/member1/C8A_context.json').read_bytes(),
        'assets/B3_preparation_authorization.json':Path(authorization).read_bytes(),
        'assets/B3_source_verification.json':Path(source_verification).read_bytes(),
        'assets/accepted_dayB_per_type_reference.py':(root/'scripts/dayB_finish/run_g8_g9.py').read_bytes()})
    item=context['experiment_spec']['local_inputs']['calibration'];payload[item['path']]=(root/item['path']).read_bytes()
    for name in SCORING_FILES:payload['review_sources/'+name]=(root/name).read_bytes()
    for path in (root/'tests/member1').rglob('*.py'):
        payload['review_sources/'+str(path.relative_to(root))]=path.read_bytes()
    payload['README.md']=b'Joint B2/B3 audit only. Exact saved validation scores, timings and canonical metadata; all metadata bytes and calibration hashes preserved. Includes canonical train/val/test metadata for the existing Member2 val-only/split audit, never test scoring. No weights, cached feature tensors or media. Entry point: finish_day8_evaluation.py. B3/heldout source snapshot is review-only; no final freeze or test attempt.\n'
    if any(Path(n).suffix in {'.pth','.pt','.npz','.mp4','.wav'} for n in payload):raise ValueError('Forbidden raw/model/feature payload')
    manifest={'schema_id':'afml-joint-b2-b3-audit/1.0','files':{n:hashlib.sha256(b).hexdigest() for n,b in sorted(payload.items())},
              'scope':spec['scope'],'canonical_metadata_included_as_exact_bytes':True,'test_features_included':False}
    payload['JOINT_MANIFEST.json']=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name,content in sorted(payload.items()):z.writestr(NAME+'/'+name,content)
    with zipfile.ZipFile(output) as z:
        if z.testzip() is not None:raise RuntimeError('Archive check failed')
        for name,content in payload.items():
            if z.read(NAME+'/'+name)!=content:raise RuntimeError('Archive content differs')
    return {'archive':str(output),'sha256':sha256_file(output),'bytes':output.stat().st_size,'files':len(payload),
            'package_name':NAME,'manifest_sha256':hashlib.sha256(payload['JOINT_MANIFEST.json']).hexdigest(),
            'self_contained_saved_inputs':True,'weights_or_feature_tensors_included':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('repo-root','out','input-spec','authorization','source-verification'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(build(a.repo_root,a.out,input_spec=a.input_spec,
        authorization=a.authorization,source_verification=a.source_verification),indent=2))


if __name__=='__main__':main()
