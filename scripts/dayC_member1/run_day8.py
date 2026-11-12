"""C8A-guarded full query validation wrapper; unchanged Day B loop in a child."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from avfuse.member1_fusion.dayc_guards import (
    ensure_inside_repo, read_strict_json, require_human_approval, require_new_file,
    require_sha256, sha256_file, write_new_json)
from avfuse.member1_fusion.dayc_validation import verify_validation_metadata, verify_prediction_rows
from avfuse.member1_fusion.dayc_cache_mapping import verify_mapped_validation_cache
from scripts.dayC_member1.colab_query_pilot import check_environment, verify_package


def read_rows(path):
    from avfuse.shared.jsonl import read_jsonl
    return read_jsonl(path)


def load_release(root, approval_file):
    context_path = root/'configs/dayC/member1/C8A_context.json'
    context = read_strict_json(context_path)
    require_human_approval(ensure_inside_repo(root, approval_file, must_exist=True), 'C8A',
                           context_sha256=sha256_file(context_path))
    if (root/'PACKAGE_MANIFEST.json').exists():
        verify_package(root)
    for name, expected in context['pilot_source_sha256'].items():
        require_sha256(root/name, expected)
    for name, item in context['experiment_spec']['local_inputs'].items():
        if name.startswith('source_'):
            require_sha256(root/item['path'],item['sha256'])
    release = context['pilot_release']
    require_human_approval(root/'assets/c0_approval.json' if (root/'PACKAGE_MANIFEST.json').exists()
                           else root/release['c0_approval_path'], 'C0', context_sha256=release['c0_context_sha256'])
    return context, release


def observed_runtime():
    import torch
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    return {'python':sys.version,'platform':platform.platform(),
        'versions':{k:importlib.metadata.version(k) for k in ('numpy','jsonschema','torch','scikit-learn','matplotlib','pytest')},
        'device':'cpu','torch_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads(),
        'cuda_available':torch.cuda.is_available()}


def preflight(root, approval_file, ready_path, canonical_path, upstream, output, evidence):
    context, release = load_release(root, approval_file)
    output = ensure_inside_repo(root, output)
    if output != root/'artifacts/predictions/restore_query_val.jsonl':
        raise PermissionError('Only the canonical new query validation output is allowed')
    require_new_file(output)
    evidence = ensure_inside_repo(root, evidence)
    evidence.relative_to(root/'artifacts/dayC/member1/day8')
    require_new_file(evidence)
    packaged = (root/'PACKAGE_MANIFEST.json').exists()
    for requested, expected in ((ready_path,release['ready_manifest']),
                                (canonical_path,release['canonical_manifest']), (upstream,release['upstream'])):
        if Path(requested) != Path(expected):
            raise PermissionError('Input differs from approved original path')
        if not packaged:
            ensure_inside_repo(root,requested,must_exist=True)
    current = observed_runtime()
    check_environment(current,release['environment'],release['runtime_amendment']['allowed_version_changes'])
    protected = {Path(release[k]):release[k+'_sha256'] for k in
                 ('canonical_manifest','validation_manifest','ready_manifest','g4_report','g5_context')}
    protected.update({Path(a['path']):a['sha256'] for a in release['runtime_assets']})
    spec = context['experiment_spec']
    for key in ('predictions_reference','predictions_low_precision','predictions_restore_value_output','calibration'):
        item=spec['local_inputs'][key];protected[root/item['path']]=item['sha256']
    for path, expected in protected.items():require_sha256(path,expected)
    from avfuse.member2_data.manifest import read_manifest
    canonical = read_manifest(canonical_path)
    ready = read_manifest(ready_path)
    ids = verify_validation_metadata(canonical,ready)
    for mode in ('reference','low_precision','restore_value_output'):
        item=spec['local_inputs']['predictions_'+mode]
        verify_prediction_rows(read_rows(root/item['path']),mode,ids,release['checkpoint_sha256'])
    protocol=read_strict_json(root/spec['local_inputs']['calibration']['path'])
    if protocol['manifest_sha256'] != release['canonical_manifest_sha256'] or protocol['checkpoint_sha256'] != release['checkpoint_sha256']:
        raise ValueError('Calibration provenance differs')
    g4=read_strict_json(release['g4_report'])
    if not (g4['status']=='PASS' and g4['n_verified']==g4['n_expected']==1200 and not g4['errors']):
        raise ValueError('G4 provenance incomplete')
    mapping=release['validation_cache_mapping']
    cache = verify_mapped_validation_cache(ready,g4['cache_inventory'],mapping['original_root'],mapping['mapped_root'])
    return context,release,protected,ids,ready,current,cache


def launch_checked(command, log, root):
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(root/'src')+os.pathsep+str(root),
             PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',MPLBACKEND='Agg')
    with Path(log).open('x') as stream:
        subprocess.run(command,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)


def execute(root, approval_file, ready_path, canonical_path, upstream, output, evidence):
    context,release,protected,ids,ready,current,cache = preflight(
        root,approval_file,ready_path,canonical_path,upstream,output,evidence)
    evidence=ensure_inside_repo(root,evidence);evidence.mkdir(parents=True,exist_ok=False)
    output=ensure_inside_repo(root,output);output.parent.mkdir(parents=True,exist_ok=True)
    write_new_json(evidence/'input_hashes.json',{str(p):h for p,h in protected.items()})
    write_new_json(evidence/'cache_verification.json',cache)
    write_new_json(evidence/'execution_context.json',{'runtime':current,'checkpoint_sha256':release['checkpoint_sha256'],
        'approval_sha256':sha256_file(approval_file),'C8A_context_sha256':sha256_file(root/'configs/dayC/member1/C8A_context.json'),
        'cache_mapping':release['validation_cache_mapping'],'test_features_accessed':False,
        'package_manifest_sha256':sha256_file(root/'PACKAGE_MANIFEST.json') if (root/'PACKAGE_MANIFEST.json').exists() else None,
        'canonical_metadata_scope':'labels/IDs and source/identity split audit only; no test features or predictions'})
    source={n:sha256_file(root/n) for n in context['pilot_source_sha256']}
    source.update({str(p.relative_to(root)):sha256_file(p) for p in (root/'scripts/dayC_member1').glob('*.py')})
    if (root/'PACKAGE_MANIFEST.json').exists():
        source.update({n:h for n,h in read_strict_json(root/'PACKAGE_MANIFEST.json')['files'].items() if n.endswith('.py')})
    write_new_json(evidence/'source_code_hashes.json',source)
    write_new_json(evidence/'query_pilot_trace.json',read_strict_json(root/'assets/accepted_pilot.json'))
    command=[sys.executable,'-B',str(root/'scripts/dayC_member1/day8_child.py'),'validation',
        '--approval-file',str(approval_file),'--','--manifest',str(ready_path),'--mode','restore_query','--device','cpu',
        '--upstream',str(upstream),'--repo-root',str(root),'--run-id',evidence.parent.name+'_'+evidence.name,'--out',str(output)]
    write_new_json(evidence/'attempt_started.json',{'command':command,'started_utc':datetime.now(timezone.utc).isoformat()})
    try:
        # Lock is persistent even after failure; no automatic second model attempt.
        with (root/'artifacts/dayC/member1/day8/full_validation_execution.lock').open('x') as lock:
            lock.write(str(evidence.relative_to(root))+'\n')
        launch_checked(command,evidence/'query_validation_run.log',root)
        result=verify_prediction_rows(read_rows(output),'restore_query',ids,release['checkpoint_sha256'])
        for path,expected in protected.items():require_sha256(path,expected)
        for name,expected in source.items():require_sha256(root/name,expected)
        result.update(status='COMPLETE',output_sha256=sha256_file(output),category_counts={k:300 for k in
                      ('real','audio_fake','video_fake','both_fake')},old_predictions_unchanged=True)
        write_new_json(evidence/'validation_completion.json',result)
        with (evidence/'query_validation_output.sha256').open('x') as stream:stream.write(sha256_file(output)+'\n')
        write_new_json(evidence/'attempt_status.json',{'status':'COMPLETE','full_validation_count':1200,'test_access':False})
        return evidence
    except BaseException as exc:
        write_new_json(evidence/'attempt_status.json',{'status':'FAILED','error':f'{type(exc).__name__}: {exc}',
                                                     'automatic_retry_permitted':False,'test_access':False})
        raise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('repo-root','ready-val-manifest','canonical-manifest','upstream','approval-file','out','evidence-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(argv)
    root=a.repo_root.resolve(strict=True)
    if not a.repo_root.is_absolute() or root!=Path(__file__).resolve().parents[2]:
        raise PermissionError('Wrapper must belong to the supplied project root')
    execute(root,a.approval_file,a.ready_val_manifest,a.canonical_manifest,a.upstream,a.out,a.evidence_dir)


if __name__=='__main__':main()
