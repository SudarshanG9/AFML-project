"""Synthetic package/guard tests; never connect to Colab/Drive or real test data."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest
from scripts.dayC_member1 import colab_query_pilot as runner
from scripts.dayC_member1 import build_colab_pilot_package as builder


@pytest.fixture
def package_fixture(tmp_path):
    actual = Path(builder.__file__).resolve().parents[2]
    root = tmp_path / 'TEST_FIXTURE_repo'
    root.mkdir()
    frozen = json.loads((actual/'configs/dayB/frozen_g5_package_manifest.json').read_text())
    names = [x['path'] for x in frozen['files'] if x['path'].startswith('src/')]
    names += ['scripts/dayB_inference.py','contracts/schemas/manifest.schema.json','contracts/schemas/prediction.schema.json']
    names += list(builder.NEW_SOURCES)
    names += ['configs/dayC/member1/colab_pilot_release_mapped_v3.json', 'configs/dayC/member1/colab_pilot_release_drift_v2.json', 'artifacts/predictions/reference_val.jsonl']
    names += ['configs/dayB/frozen_g5_package_manifest.json','configs/dayC/member1/colab_pilot_release.json',
        'artifacts/dayC/member1/approvals/C0_ack_20261112T062409280108Z.json',
        'artifacts/dayB/final_review/DayB_FINAL_CPU_20261112T031627Z_91e5f723/received_evidence/provenance/g5/execution_context.json']
    names += [str(p.relative_to(actual)) for p in (actual/'artifacts/dayC/member1/approvals').glob('C8_2_RUNTIME_*.json')]
    names += [str(p.relative_to(actual)) for p in (actual/'artifacts/dayC/member1/approvals').glob('C8_2_CACHE_MAPPING_*.json')]
    for name in names:
        dest = root/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes((actual/name).read_bytes())
    return root, root/'artifacts/dayC/member1/colab/TEST_FIXTURE_package.zip'


def test_package_preserves_approved_bytes_and_contains_no_private_assets(package_fixture):
    root, archive = package_fixture
    result = builder.build_package(root, archive)
    assert result['original_source_count'] == 41
    extracted = root/'extracted'
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in bundle.namelist())
        assert not any(Path(n).suffix in {'.pth','.npz','.pt','.jsonl','.mp4','.wav'} for n in bundle.namelist())
        bundle.extractall(extracted)
    expected = runner.verify_package(extracted/builder.PACKAGE_NAME)
    assert len(expected)+2 == result['files']
    frozen = json.loads((root/'configs/dayB/frozen_g5_package_manifest.json').read_text())
    for item in frozen['files']:
        if item['path'] in expected:
            assert expected[item['path']] == item['sha256']
    before = archive.read_bytes()
    with pytest.raises(FileExistsError):
        builder.build_package(root, archive)
    assert archive.read_bytes() == before


@pytest.mark.parametrize('defect',['source','authorization','outside','symlink'])
def test_package_creation_rejects_invalid_inputs(package_fixture, defect):
    root,out = package_fixture
    if defect == 'source':
        path = root/'src/avfuse/member1_fusion/precision.py'
        path.write_bytes(path.read_bytes()+b'\n# TEST_FIXTURE changed\n')
        error = ValueError
    elif defect == 'authorization':
        for p in (root/'artifacts/dayC/member1/approvals').glob('C8_2_RUNTIME_*.json'):
            p.unlink()
        error = PermissionError
    elif defect == 'outside':
        out = root.parent/'escape.zip'
        error = PermissionError
    else:
        out.parent.mkdir(parents=True)
        out.symlink_to(out.parent/'not_created.zip')
        error = FileExistsError
    with pytest.raises(error):builder.build_package(root,out)
    if defect!='symlink':assert not out.exists()


def test_archive_source_tamper_rejected(package_fixture):
    root,archive = package_fixture
    builder.build_package(root,archive)
    with zipfile.ZipFile(archive) as z:z.extractall(root/'extract')
    package = root/'extract'/builder.PACKAGE_NAME
    source = package/'src/avfuse/member1_fusion/precision.py'
    source.write_bytes(source.read_bytes()+b'# TEST_FIXTURE changed\n')
    with pytest.raises(RuntimeError,match='hash mismatch'):runner.verify_package(package)


@pytest.mark.parametrize('key',['python','platform','versions','device','torch_threads','torch_interop_threads','cuda_available'])
def test_environment_mismatch_fails_closed(key):
    env = {'python':'TEST_FIXTURE','platform':'TEST_FIXTURE','versions':{'torch':'TEST_FIXTURE'},
           'device':'cpu','torch_threads':1,'torch_interop_threads':1,'cuda_available':False}
    runner.check_environment(env,env)
    changed=deepcopy(env);changed[key]='different'
    with pytest.raises(RuntimeError,match='environment changed'):runner.check_environment(changed,env)


def synthetic_rows():
    categories=['real','audio_fake','video_fake','both_fake']
    return [{'clip_id':f'{i:016x}','dataset':'TEST_FIXTURE','source_id':str(i),'identity_id':str(i),
             'label':int(i>=300),'split':'val','manipulation_type':categories[i//300],
             'feature_path':'TEST_FIXTURE'} for i in range(1200)]


@pytest.mark.parametrize('defect',[None,'short','duplicate','test','metadata'])
def test_validation_cohort_never_downscales_or_opens_test(defect):
    rows=synthetic_rows();ready=deepcopy(rows)
    if defect=='short':ready.pop()
    elif defect=='duplicate':ready[-1]=deepcopy(ready[0])
    elif defect=='test':ready[0]['split']='test'  # synthetic marker only
    elif defect=='metadata':ready[0]['source_id']='different'
    if defect is None:assert len(runner.validate_cohort(rows,ready))==1200
    else:
        with pytest.raises(RuntimeError):runner.validate_cohort(rows,ready)


@pytest.mark.parametrize('defect',[None,'missing','escape','metadata_missing','test'])
def test_cache_check_is_presence_only(tmp_path, monkeypatch, defect):
    base=tmp_path/'TEST_FIXTURE_cache';feature=base/'val/clip/features.npz';feature.parent.mkdir(parents=True)
    feature.write_bytes(b'not a real tensor');(feature.parent/'meta.json').write_text('{}')
    row={'clip_id':'clip','split':'val','feature_path':str(feature)}
    if defect=='missing':feature.unlink()
    elif defect=='escape':row['feature_path']=str(tmp_path/'external.npz')
    elif defect=='metadata_missing':(feature.parent/'meta.json').unlink()
    elif defect=='test':row['split']='test'
    original=Path.open
    def no_feature_open(self,*args,**kwargs):
        if self.suffix=='.npz':pytest.fail('presence check read feature contents')
        return original(self,*args,**kwargs)
    monkeypatch.setattr(Path,'open',no_feature_open)
    if defect is None:assert runner.check_cache_paths([row],base)['validation_feature_files_present']==1
    else:
        with pytest.raises(RuntimeError):runner.check_cache_paths([row],base)


@pytest.mark.parametrize('release_name', ['colab_pilot_release.json', 'colab_pilot_release_drift_v2.json', 'colab_pilot_release_mapped_v3.json'])
def test_packaged_entry_on_local_mac_stops_before_model_or_cache(package_fixture, release_name):
    root,archive=package_fixture
    built=builder.build_package(root,archive,release_name)
    with zipfile.ZipFile(archive) as z:z.extractall(root/'extract')
    package=root/'extract'/built['package_name']
    # The host intentionally differs from accepted Colab. This does not run a pilot.
    result=subprocess.run([sys.executable,'-B',str(package/'scripts/dayC_member1/colab_query_pilot.py')],
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True)
    assert result.returncode==1,result.stdout+result.stderr
    runs=list((package/'artifacts/dayC/member1/day8').glob('C8_2_PILOT_*'))
    assert len(runs)==1
    failure=json.loads((runs[0]/'failure.json').read_text())
    assert failure['stage']=='read_only_preflight' and 'environment changed' in failure['error']
    assert (runs[0]/'C8_2_pilot_review.zip').is_file()
    assert not (runs[0]/'pilot.json').exists()
    assert not (runs[0].parent/'pilot_execution.lock').exists()



def approved_drift_fixture():
    actual=Path(builder.__file__).resolve().parents[2]
    release=json.loads((actual/'configs/dayC/member1/colab_pilot_release_drift_v2.json').read_text())
    return release


def test_explicit_three_version_amendment_only():
    release=approved_drift_fixture();old=release['environment'];new=deepcopy(old)
    changes=release['runtime_amendment']['allowed_version_changes']
    for name,change in changes.items():new['versions'][name]=change['to']
    report=runner.check_environment(new,old,changes)
    assert not report['identical_to_dayb']
    assert set(report['version_differences'])=={'numpy','jsonschema','pytest'}
    assert new['versions']['torch']==old['versions']['torch']=='2.14.1+cpu'
    with pytest.raises(RuntimeError):runner.check_environment(new,old)


@pytest.mark.parametrize('defect',['torch','numpy_extra','scikit-learn','thread','extra_amendment','wrong_from'])
def test_drift_authorization_is_not_general_runtime_bypass(defect):
    release=approved_drift_fixture();old=release['environment'];new=deepcopy(old)
    changes=deepcopy(release['runtime_amendment']['allowed_version_changes'])
    for name,change in changes.items():new['versions'][name]=change['to']
    if defect=='torch':new['versions']['torch']='different'
    elif defect=='numpy_extra':new['versions']['numpy']='2.4.7'
    elif defect=='scikit-learn':new['versions']['scikit-learn']='different'
    elif defect=='thread':new['torch_threads']=2
    elif defect=='extra_amendment':changes['torch']={'from':old['versions']['torch'],'to':'different'}
    else:changes['numpy']['from']='different'
    with pytest.raises(RuntimeError):runner.check_environment(new,old,changes)


@pytest.mark.parametrize('defect',[None,'score','nan','tolerance','clip','test'])
def test_dayb_reference_score_consistency_stops_on_disagreement(defect):
    anchor=deepcopy(approved_drift_fixture()['dayb_reference_anchor']);actual=anchor['score_fake']
    if defect=='score':actual+=1e-3
    elif defect=='nan':actual=float('nan')
    elif defect=='tolerance':anchor['atol']=1.
    elif defect=='clip':anchor['clip_id']='other'
    elif defect=='test':anchor['split']='test'  # Metadata fixture, no held-out rows.
    if defect is None:
        result=runner.check_reference_anchor(actual,anchor)
        assert result['absolute_difference']==0 and 'not population' in result['scope']
    else:
        with pytest.raises(RuntimeError):runner.check_reference_anchor(actual,anchor)


def test_v2_archive_uses_new_name_and_exact_approved_amendment(package_fixture):
    root,out=package_fixture
    result=builder.build_package(root,out,'colab_pilot_release_drift_v2.json')
    assert result['package_name']=='AFML_DayC_C8_2_CPU_Pilot_Drift_v2'
    with zipfile.ZipFile(out) as z:
        assert {Path(n).parts[0] for n in z.namelist()}=={result['package_name']}
        prefix=result['package_name']+'/'
        config=json.loads(z.read(prefix+'configs/dayC/member1/colab_pilot_release.json'))
        approval=json.loads(z.read(prefix+'assets/runtime_approval.json'))
        assert config['environment']['versions']['numpy']=='2.1.3'
        assert config['runtime_amendment']['allowed_version_changes']['numpy']['to']=='2.4.6'
        assert approval['gate']=='C8.2_RUNTIME_AMENDMENT'
        assert approval['context_sha256']==result['release_sha256']
        assert not any(Path(n).suffix in {'.npz','.pth','.pt','.jsonl'} for n in z.namelist())


def test_v2_builder_rejects_unverified_baseline_reference_value(package_fixture):
    root,out=package_fixture
    prediction=root/'artifacts/predictions/reference_val.jsonl'
    prediction.write_bytes(prediction.read_bytes()+b'\n')
    with pytest.raises(ValueError):builder.build_package(root,out,'colab_pilot_release_drift_v2.json')
    assert not out.exists()


def test_mapped_package_preserves_original_runtime_approval_chain(package_fixture):
    root, archive = package_fixture
    result = builder.build_package(root, archive, 'colab_pilot_release_mapped_v3.json')
    with zipfile.ZipFile(archive) as z:
        base = result['package_name'] + '/'
        parent = z.read(base+'assets/runtime_parent_release.json')
        approval = json.loads(z.read(base+'assets/runtime_approval.json'))
        mapping_approval = json.loads(z.read(base+'assets/cache_mapping_approval.json'))
        assert approval['context_sha256'] == builder.sha(parent)
        assert mapping_approval['context_sha256'] == result['release_sha256']
        assert parent == (root/'configs/dayC/member1/colab_pilot_release_drift_v2.json').read_bytes()
        assert not any(Path(n).suffix in {'.npz','.jsonl','.pth'} for n in z.namelist())


def test_mapped_package_rejects_missing_mapping_ack(package_fixture):
    root, archive = package_fixture
    for p in (root/'artifacts/dayC/member1/approvals').glob('C8_2_CACHE_MAPPING_*.json'):
        p.unlink()
    with pytest.raises(PermissionError, match='mapping instruction'):
        builder.build_package(root, archive, 'colab_pilot_release_mapped_v3.json')
