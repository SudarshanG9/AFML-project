"""Build a local, source-only C8.2 package; no remote operations or inference."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

PACKAGE_NAME = 'AFML_DayC_C8_2_CPU_Pilot'
NEW_SOURCES = ('scripts/dayC_member1/__init__.py', 'scripts/dayC_member1/colab_query_pilot.py',
               'src/avfuse/member1_fusion/dayc_cache_mapping.py', 'src/avfuse/member1_fusion/dayc_guards.py', 'src/avfuse/member1_fusion/dayc_precision_audit.py')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def build_package(root, destination, release_name='colab_pilot_release.json'):
    from avfuse.member1_fusion.dayc_guards import (
        ensure_inside_repo, read_strict_json, require_new_file, require_sha256, require_human_approval)
    root = Path(root).resolve(strict=True)
    requested = Path(destination)
    destination = ensure_inside_repo(root, requested)
    destination.relative_to(root / 'artifacts/dayC/member1/colab')
    require_new_file(requested if requested.is_absolute() else root / requested)
    require_new_file(destination)
    config_path = ensure_inside_repo(root, Path('configs/dayC/member1') / release_name, must_exist=True)
    config_path.relative_to(root/'configs/dayC/member1')
    release = read_strict_json(config_path)
    package_name = release.get('package_name', PACKAGE_NAME)
    if package_name not in {PACKAGE_NAME, 'AFML_DayC_C8_2_CPU_Pilot_Drift_v2', 'AFML_DayC_C8_2_CPU_Pilot_Mapped_v3'}:
        raise PermissionError('Unapproved package name')
    extra_payload = {}
    runtime_context = sha(config_path.read_bytes())
    mapping = release.get('validation_cache_mapping')
    if mapping:
        from avfuse.member1_fusion.dayc_cache_mapping import validate_mapping_release
        parent = root/'configs/dayC/member1/colab_pilot_release_drift_v2.json'
        require_sha256(parent, mapping['runtime_parent_release_sha256'])
        validate_mapping_release(release, read_strict_json(parent))
        runtime_context = sha(parent.read_bytes())
        extra_payload['assets/runtime_parent_release.json'] = parent.read_bytes()
        for candidate in sorted((root/'artifacts/dayC/member1/approvals').glob('C8_2_CACHE_MAPPING_*.json')):
            record = read_strict_json(candidate)
            if (record.get('gate') == mapping['gate'] and record.get('ack_text') == mapping['ack_text']
                    and record.get('user_supplied') is True and record.get('context_sha256') == sha(config_path.read_bytes())):
                extra_payload['assets/cache_mapping_approval.json'] = candidate.read_bytes()
                break
        if 'assets/cache_mapping_approval.json' not in extra_payload:
            raise PermissionError('Missing user mapping instruction record')
    amendment = release.get('runtime_amendment')
    expected_gate = amendment['gate'] if amendment else 'C8.2_RUNTIME'
    expected_ack = amendment['ack_text'] if amendment else 'Approve existing Colab CPU; prepare the local pilot package'
    c0_path = ensure_inside_repo(root, Path(release['c0_approval_path']), must_exist=True)
    require_human_approval(c0_path, 'C0', context_sha256=release['c0_context_sha256'])
    approval_path = None
    for path in sorted((root/'artifacts/dayC/member1/approvals').glob('C8_2_RUNTIME_*.json'), reverse=True):
        path = ensure_inside_repo(root, path, must_exist=True)
        record = read_strict_json(path)
        if (record.get('gate') == expected_gate and record.get('user_supplied') is True
                and record.get('ack_text') == expected_ack
                and record.get('context_sha256') == runtime_context):
            approval_path = path
            break
    if approval_path is None:
        raise PermissionError('No explicit runtime authorization for this release')
    if amendment:
        anchor = release['dayb_reference_anchor']
        prediction = ensure_inside_repo(root, Path(anchor['prediction_file_repo_path']), must_exist=True)
        require_sha256(prediction, anchor['prediction_file_sha256'])
        rows = [json.loads(line) for line in prediction.read_text().splitlines() if line.strip()]
        matches = [row for row in rows if row['clip_id'] == anchor['clip_id']]
        if len(matches) != 1 or any(matches[0][key] != anchor[key] for key in
                                  ('score_fake','run_id','checkpoint_sha256','split','config')):
            raise ValueError('Stored Day B reference anchor does not match actual validation evidence')

    frozen_path = root / 'configs/dayB/frozen_g5_package_manifest.json'
    frozen = read_strict_json(frozen_path)
    expected = {item['path']: item['sha256'] for item in frozen['files']}
    originals = sorted([name for name in expected if name.startswith('src/')]
                       + ['scripts/dayB_inference.py','contracts/schemas/manifest.schema.json',
                          'contracts/schemas/prediction.schema.json'])
    payload = dict(extra_payload)
    for name in originals:
        path = ensure_inside_repo(root, Path(name), must_exist=True)
        require_sha256(path, expected[name])
        payload[name] = path.read_bytes()
    for name in NEW_SOURCES:
        payload[name] = ensure_inside_repo(root, Path(name), must_exist=True).read_bytes()
    context = root/'artifacts/dayB/final_review/DayB_FINAL_CPU_20261112T031627Z_91e5f723/received_evidence/provenance/g5/execution_context.json'
    require_sha256(context, release['g5_context_sha256'])
    payload.update({'configs/dayC/member1/colab_pilot_release.json':config_path.read_bytes(),
        'assets/accepted_g5_context.json':context.read_bytes(), 'assets/runtime_approval.json':approval_path.read_bytes(),
        'assets/c0_approval.json':c0_path.read_bytes(), 'assets/frozen_g5_package_manifest.json':frozen_path.read_bytes()})
    payload['README.md'] = b'C8.2 validation pilot only. Use the checksum-pinned local quickstart in the existing approved Colab CPU runtime. No installs, mounts, full validation or held-out access. Return C8_2_pilot_review.zip and stop.\n'
    if any(Path(name).suffix in {'.pth','.pt','.npz','.jsonl','.mp4','.wav'} for name in payload):
        raise PermissionError('Package must not contain weights, features, media or dataset rows')
    manifest = {'schema_id':'avfuse-member1-colab-pilot-package/1.0','scope':'C8.2 validation pilot only',
                'g5_original_source_count':len(originals),'files':{name:sha(data) for name,data in sorted(payload.items())}}
    data = (json.dumps(manifest,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    payload['PACKAGE_MANIFEST.json'] = data
    payload['PACKAGE_MANIFEST.sha256'] = (sha(data)+'\n').encode()
    destination.parent.mkdir(parents=True,exist_ok=True)
    # The final archive name is exclusive. No existing file can be replaced.
    with destination.open('xb') as stream:
        with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in sorted(payload.items()):
                info = zipfile.ZipInfo(package_name+'/'+name, date_time=(2026,11,12,0,0,0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info,data)
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('Archive integrity failed; preserve file and stop')
        if {n.removeprefix(package_name+'/') for n in archive.namelist()} != set(payload):
            raise RuntimeError('Package file set differs')
        for name,data in payload.items():
            if archive.read(package_name+'/'+name) != data:
                raise RuntimeError('Packaged file bytes differ: '+name)
    return {'package_name':package_name,'archive':str(destination),'sha256':sha(destination.read_bytes()),'bytes':destination.stat().st_size,
            'files':len(payload),'original_source_count':len(originals),'manifest_sha256':sha(payload['PACKAGE_MANIFEST.json']),
            'runtime_approval_sha256':sha(approval_path.read_bytes()),'release_sha256':sha(config_path.read_bytes())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--release-name',default='colab_pilot_release.json')
    args = parser.parse_args()
    if not args.repo_root.is_absolute():
        parser.error('--repo-root must be absolute')
    print(json.dumps(build_package(args.repo_root,args.out,args.release_name),indent=2))


if __name__ == '__main__':
    main()
