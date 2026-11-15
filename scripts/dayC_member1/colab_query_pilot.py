"""Manual Colab CPU C8.2 pilot only. No installation, mount or network access."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            value.update(chunk)
    return value.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def verify_package(root):
    manifest = root / "PACKAGE_MANIFEST.json"
    require(digest(manifest) == (root / "PACKAGE_MANIFEST.sha256").read_text().strip(), "Package manifest anchor changed")
    expected = load(manifest)["files"]
    for rel, sha in expected.items():
        path = root / rel
        require(not Path(rel).is_absolute() and ".." not in Path(rel).parts, "Unsafe package path")
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root.resolve()), "Missing/escaping package input: " + rel)
        require(digest(path) == sha, "Package hash mismatch: " + rel)
    actual_sources = {str(p.relative_to(root)) for base in ("src", "scripts")
                      for p in (root / base).rglob("*.py")}
    require(actual_sources == {rel for rel in expected if rel.endswith('.py')}, "Package Python source set changed")
    return expected


def check_environment(current, accepted, version_drift=None):
    expected_versions = dict(accepted['versions'])
    if version_drift is not None:
        require(set(version_drift) == {'numpy','jsonschema','pytest'}, 'Unapproved dependency amendment')
        for name, change in version_drift.items():
            require(set(change) == {'from','to'} and change['from'] == accepted['versions'][name],
                    'Invalid version amendment: ' + name)
            expected_versions[name] = change['to']
    # Torch and every non-amended setting remain exact, including dependencies
    # not named in the user's narrowly scoped acknowledgement.
    for key in ('python','platform','device','torch_threads','torch_interop_threads','cuda_available'):
        require(current[key] == accepted[key], 'Accepted CPU environment changed: ' + key + '; stop, do not install replacements')
    require(current['versions'] == expected_versions,
            'Accepted CPU environment changed: versions outside approved exact values; stop, do not install replacements')
    return {'identical_to_dayb':current['versions'] == accepted['versions'],
            'version_differences':{name:{'dayb':old,'observed':current['versions'][name]}
                for name,old in accepted['versions'].items() if current['versions'][name] != old},
            'scope':'approved pilot amendment only; no full-cohort/timing equivalence claim'}


def check_reference_anchor(actual, anchor):
    expected = anchor['score_fake']
    require(anchor['clip_id'] == '7cea6ad75b4ebf41' and anchor['split'] == 'val'
            and anchor['config'] == 'reference', 'Wrong Day B reference anchor')
    require(anchor['atol'] == 1e-6 and anchor['rtol'] == 1e-5, 'Reference-score tolerances changed')
    require(all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
                and 0 <= x <= 1 for x in (actual,expected)), 'Invalid reference score')
    difference = abs(actual-expected)
    limit = anchor['atol'] + anchor['rtol'] * abs(expected)
    require(difference <= limit, 'Native reference score differs from stored Day B anchor; stop for research review')
    return {'status':'PASS','dayb_score_fake':expected,'current_native_score_fake':actual,
            'absolute_difference':difference,'atol':anchor['atol'],'rtol':anchor['rtol'],
            'source_prediction_sha256':anchor['prediction_file_sha256'],
            'scope':'single-clip score consistency only; not population or runtime equivalence'}


def validate_cohort(validation, ready):
    counts = {"real": 300, "audio_fake": 300, "video_fake": 300, "both_fake": 300}
    require(len(validation) == len(ready) == 1200, "Full 1,200 validation metadata rows required")
    originals = {row["clip_id"]: row for row in validation}
    require(len(originals) == 1200 and len({r['clip_id'] for r in ready}) == 1200, "Duplicate validation ID")
    require(set(originals) == {r["clip_id"] for r in ready}, "Validation ID sets differ")
    require(Counter(row["manipulation_type"] for row in ready) == counts, "Validation strata differ")
    for row in ready:
        require(row['split'] == originals[row['clip_id']]['split'] == 'val', "Non-validation row; stop")
        require({k: v for k, v in row.items() if k != 'feature_path'} ==
                {k: v for k, v in originals[row['clip_id']].items() if k != 'feature_path'}, "Ready metadata differs")
    return originals


def check_cache_paths(ready, cache_root):
    val = Path(cache_root) / 'val'
    require(val.is_dir(), "Recorded validation cache root unavailable; do not rewrite paths")
    for row in ready:
        require(row['split'] == 'val', "Non-validation feature reference")
        path = Path(row['feature_path'])
        require(path == val / row['clip_id'] / 'features.npz', "Feature path differs from recorded cache layout")
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(val.resolve()), "Missing/escaping validation feature: " + row['clip_id'])
        require((path.parent / 'meta.json').is_file(), "Missing validation metadata: " + row['clip_id'])
    return {"validation_feature_files_present": len(ready), "validation_metadata_files_present": len(ready),
            "feature_contents_loaded_for_this_check": False, "all_cache_bytes_rehashed": False}


def main():
    sys.dont_write_bytecode = True
    package_hashes = verify_package(ROOT)
    release_path = ROOT / 'configs/dayC/member1/colab_pilot_release.json'
    release = load(release_path)
    sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
    mapping = release.get('validation_cache_mapping')
    runtime_context = digest(release_path)
    if mapping:
        from avfuse.member1_fusion.dayc_cache_mapping import validate_mapping_release
        parent = ROOT / 'assets/runtime_parent_release.json'
        require(digest(parent) == mapping['runtime_parent_release_sha256'], 'Runtime parent changed')
        validate_mapping_release(release, load(parent))
        mapping_approval = load(ROOT / 'assets/cache_mapping_approval.json')
        require(mapping_approval.get('gate') == mapping['gate']
                and mapping_approval.get('ack_text') == mapping['ack_text']
                and mapping_approval.get('user_supplied') is True
                and mapping_approval.get('context_sha256') == digest(release_path), 'Mapping authorization mismatch')
        runtime_context = digest(parent)
    approval = load(ROOT / 'assets/runtime_approval.json')
    amendment = release.get('runtime_amendment')
    expected_gate = amendment['gate'] if amendment else 'C8.2_RUNTIME'
    expected_ack = amendment['ack_text'] if amendment else 'Approve existing Colab CPU; prepare the local pilot package'
    require(approval.get('gate') == expected_gate and approval.get('user_supplied') is True
            and approval.get('ack_text') == expected_ack
            and approval.get('context_sha256') == runtime_context, "Runtime approval does not match release")
    c0 = load(ROOT / 'assets/c0_approval.json')
    require(c0.get('gate') == 'C0' and c0.get('user_supplied') is True
            and c0.get('context_sha256') == release['c0_context_sha256']
            and c0.get('ack_text') == 'ACK DAYC C0: ACCEPT DAYB EVIDENCE AND LOCAL BASELINE', "Missing C0 acknowledgement")
    sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
    from avfuse.member1_fusion.dayc_guards import write_new_json
    output_root = ROOT / 'artifacts/dayC/member1/day8'
    output_root.mkdir(parents=True, exist_ok=True)
    run = output_root / ('C8_2_PILOT_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8])
    run.mkdir(exist_ok=False)
    stage = 'read_only_preflight'
    print('PILOT_RUN_DIR=' + str(run), flush=True)
    try:
        import torch
        # Reproduce the accepted one-thread benchmark/inference settings in
        # this fresh subprocess before constructing tensors or a model.
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        current = {'python': sys.version, 'platform': platform.platform(),
                   'versions': {key: importlib.metadata.version(key) for key in
                       ('numpy','jsonschema','torch','scikit-learn','matplotlib','pytest')},
                   'device': 'cpu', 'torch_threads': torch.get_num_threads(),
                   'torch_interop_threads': torch.get_num_interop_threads(),
                   'cuda_available': torch.cuda.is_available()}
        write_new_json(run / 'runtime_observed.json', {'observed':current,
            'dayb_environment':release['environment'],'approved_amendment':amendment,
            'model_constructed':False,'feature_contents_read':False})
        runtime_comparison = check_environment(current, release['environment'],
            amendment['allowed_version_changes'] if amendment else None)
        context_path = Path(release['g5_context'])
        require(digest(context_path) == release['g5_context_sha256'], 'Original G5 context changed')
        require(load(context_path) == load(ROOT / 'assets/accepted_g5_context.json'), 'Wrong G5 run/context')
        protected = {Path(release[key]): release[key + '_sha256'] for key in
                     ('canonical_manifest','validation_manifest','ready_manifest','g4_report','g5_context')}
        protected.update({Path(row['path']): row['sha256'] for row in release['runtime_assets']})
        for path, sha in protected.items():
            require(digest(path) == sha, 'Approved file changed: ' + str(path))
        from avfuse.member2_data.manifest import read_manifest
        validation = read_manifest(release['validation_manifest'])
        ready = read_manifest(release['ready_manifest'])
        validate_cohort(validation, ready)
        g4 = load(release['g4_report'])
        require(g4['status'] == 'PASS' and g4['n_verified'] == g4['n_expected'] == 1200
                and g4['errors_count'] == 0 and not g4['errors'], 'G4 verification incomplete')
        require(g4['cache_root'] == release['cache_root'], 'Approved Drive root differs')
        if mapping:
            stage = 'full_validation_cache_identity'
            from avfuse.member1_fusion.dayc_cache_mapping import verify_mapped_validation_cache, ValidationCacheResolver
            cache_proof = verify_mapped_validation_cache(ready, g4['cache_inventory'],
                mapping['original_root'], mapping['mapped_root'])
            write_new_json(run / 'cache_verification.json', cache_proof)
            cache = {key: value for key, value in cache_proof.items() if key != 'records'}
            resolver = ValidationCacheResolver(mapping['original_root'], mapping['mapped_root'])
        else:
            cache = check_cache_paths(ready, release['cache_root'])
        item = release['typical_workload']
        require(item['role'] == 'typical_anchor' and item['clip_id'] == '7cea6ad75b4ebf41', 'Pilot clip changed')
        row = next(row for row in ready if row['clip_id'] == item['clip_id'])
        require(row['feature_path'] == item['feature_path'], 'Pilot path differs')
        feature = resolver.resolve(row) if mapping else Path(row['feature_path'])
        require(digest(feature) == item['feature_sha256'], 'Pilot feature bytes changed')
        inventory = {row['clip_id']: row for row in g4['cache_inventory']}
        require(digest(feature.parent / 'meta.json') == inventory[item['clip_id']]['metadata_sha256'], 'Pilot metadata changed')
        write_new_json(run / 'preflight.json', {'status':'PASS','environment':current,'cache':cache,
            'canonical_manifest_scope':'file bytes only; rows not opened','test_access':False,
            'package_manifest_sha256':digest(ROOT/'PACKAGE_MANIFEST.json'),
            'release_sha256':digest(release_path), 'validation_cache_mapping':mapping,
            'source_sha256':{path:sha for path,sha in package_hashes.items() if path.endswith('.py')},
            'approved_context_sha256':release['g5_context_sha256'],'pilot_feature_sha256':item['feature_sha256'],
            'runtime_comparison':runtime_comparison,'runtime_approval_sha256':digest(ROOT/'assets/runtime_approval.json')})
        # One explicit pilot attempt for this package. A failure never retries.
        with (output_root / 'pilot_execution.lock').open('x') as stream:
            stream.write(run.name + '\n')
        stage = 'validation_pilot'
        from scripts.dayB_inference import create_adapter, load_features
        from types import SimpleNamespace
        from avfuse.member1_fusion.dayc_precision_audit import (
            check_model_invariants, reference_parity_on_validation_feature, validate_precision_trace, _exclusive_audit)
        torch.manual_seed(42)
        adapter = create_adapter(SimpleNamespace(upstream=release['upstream'], device='cpu'))
        features = load_features(feature)
        original = {k: t.detach().clone() for k, t in features.items()}
        before = check_model_invariants(adapter, checkpoint_sha256=release['checkpoint_sha256'])
        stage = 'reference_parity'
        parity = reference_parity_on_validation_feature(adapter, features)
        require(len(parity['original_fake_scores']) == len(parity['instrumented_fake_scores']) == 1, 'Pilot must be batch one')
        reference_consistency = None
        if amendment:
            stage = 'dayb_reference_consistency'
            reference_consistency = check_reference_anchor(parity['original_fake_scores'][0], release['dayb_reference_anchor'])
            write_new_json(run / 'reference_consistency.json', reference_consistency)
        modes = {'reference': {'score_fake':parity['original_fake_scores'][0], 'parity':parity}}
        with _exclusive_audit(adapter.model), torch.inference_mode():
            for mode in ('low_precision','restore_value_output','restore_query'):
                stage = 'diagnostic_' + mode
                trace = []
                result = adapter.forward_with_mode(features, mode=mode, trace=trace)
                trace_audit = validate_precision_trace(trace, mode=mode)
                require(result['score_fake'].numel() == 1 and torch.isfinite(result['logits']).all(), 'Invalid logits/shape')
                score = float(result['score_fake'].reshape(-1)[0].item())
                require(math.isfinite(score) and 0 <= score <= 1, 'Invalid fake probability')
                modes[mode] = {'score_fake':score,'trace':trace,'trace_audit':trace_audit}
                state = check_model_invariants(adapter, checkpoint_sha256=release['checkpoint_sha256'])
                require(state['state_dict_sha256'] == before['state_dict_sha256'], 'Model state changed')
                require(set(features) == set(original) and all(torch.equal(t,features[k]) for k,t in original.items()), 'Input tensors changed')
        for path, sha in protected.items():
            require(digest(path) == sha, 'Frozen input changed during pilot: ' + str(path))
        require(digest(feature) == item['feature_sha256'], 'Feature changed during pilot')
        require(digest(feature.parent / 'meta.json') == inventory[item['clip_id']]['metadata_sha256'], 'Pilot metadata changed during execution')
        verify_package(ROOT)
        write_new_json(run / 'pilot.json', {'status':'PILOT_COMPLETE_PENDING_C8A','clip_id':item['clip_id'],
            'split':'val','checkpoint_sha256':release['checkpoint_sha256'],'modes':modes,
            'query_minus_low_precision_score':modes['restore_query']['score_fake']-modes['low_precision']['score_fake'],
            'state_and_inputs_unchanged':True,'feature_loads':1,
            'runtime_comparison':runtime_comparison,'dayb_reference_consistency':reference_consistency,
            'forward_calls':{'reference_native':1,'reference_instrumented':1,'low_precision':1,'restore_value_output':1,'restore_query':1},
            'scope':'one validation clip; no population/H3/equal-cost conclusion',
            'limitation':'all-five-layer interventions are cumulative; deeper upstream activations can differ',
            'full_validation_executed':False,'Day9_or_Day10_executed':False,'test_access':False})
    except Exception as exc:
        write_new_json(run / 'failure.json', {'status':'FAILED_STOP_NO_RETRY','stage':stage,
            'error':f'{type(exc).__name__}: {exc}','test_access':False})
        print('STOP:',type(exc).__name__,str(exc),flush=True)
    files = sorted(p for p in run.rglob('*') if p.is_file())
    write_new_json(run / 'output_hashes.json', {str(p.relative_to(run)):digest(p) for p in files})
    bundle = run / 'C8_2_pilot_review.zip'
    with zipfile.ZipFile(bundle,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files + [run / 'output_hashes.json']:
            archive.write(path, arcname=path.relative_to(run))
    print('RETURN REVIEW ZIP:',bundle,flush=True)
    print('STOP AT C8A REVIEW. No full validation, policy freeze or held-out run.',flush=True)
    if (run / 'failure.json').exists():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
