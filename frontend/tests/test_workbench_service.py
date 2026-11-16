"""Readiness failure attribution and real backend contracts, with no licensed model needed."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test_integration import inputs as inputs
from avfuse_ui import backend_status, workbench_service as service

ROOT = Path(__file__).resolve().parents[2]


def no_cli(monkeypatch):
    monkeypatch.setattr(backend_status.subprocess, 'run', lambda *a, **k: pytest.fail('preflight launched CLI'))


@pytest.mark.parametrize('fmt', ['shared_pt', 'dimodif_npz'])
def test_exact_missing_cache_and_reusable_id(inputs, monkeypatch, fmt):
    inputs['feature_format'] = fmt
    root = Path(inputs['feature_root'])
    if fmt == 'shared_pt':
        (root / 'c.pt').unlink()
        expected_path = root / 'c.pt'
    else:
        expected_path = root / 'val/c/features.npz'
    no_cli(monkeypatch)
    for _ in range(2):
        out = backend_status.run_cached_feature_experiment(ROOT, **inputs)
        assert out.code == 6
        assert out.payload['phase'] == 'preflight'
        assert out.payload['cli_launched'] is False
        assert out.payload['asset_counts'] == dict(expected=1, found=0, missing=1)
        assert str(expected_path) in out.diagnostics
        assert 'clip c:' in out.diagnostics
        assert not (ROOT / 'artifacts/member3/runs' / inputs['run_id']).exists()


def test_npz_ignores_planned_pt_path(inputs):
    inputs['feature_format'] = 'dimodif_npz'
    path = Path(inputs['feature_root']) / 'val/c/features.npz'
    path.parent.mkdir(parents=True)
    np.savez(path, video_features=np.zeros((2, 768), dtype='float32'), audio_features=np.zeros((3, 768), dtype='float32'))
    out = service.inspect_experiment_assets(ROOT, **inputs)
    assert out.ready and out.sample_inspected
    assert out.asset_counts == dict(expected=1, found=1, missing=0)


@pytest.mark.parametrize('case', ['checkpoint_hash', 'model_config', 'source', 'duplicate', 'no_labels', 'cache_corrupt', 'dataset', 'format', 'escape'])
def test_asset_errors_do_not_launch(inputs, monkeypatch, case, tmp_path):
    if case == 'checkpoint_hash':
        monkeypatch.setattr(service, 'checkpoint_digest', lambda p: '0'*64)
    elif case == 'model_config':
        next(Path(inputs['upstream_root']).rglob('*.json')).unlink()
    elif case == 'source':
        (Path(inputs['upstream_root']) / 'src/models.py').unlink()
    elif case in ('duplicate', 'no_labels', 'dataset'):
        path = Path(inputs['manifest']); row = json.loads(path.read_text())
        if case == 'no_labels': row.pop('label')
        if case == 'dataset': row['dataset'] = 'other'
        path.write_text(json.dumps(row)+'\n' + (json.dumps(row)+'\n' if case == 'duplicate' else ''))
    elif case == 'cache_corrupt':
        (Path(inputs['feature_root']) / 'c.pt').write_bytes(b'broken')
    elif case == 'format':
        inputs['feature_format'] = 'guess'
    elif case == 'escape':
        path = Path(inputs['manifest']); row = json.loads(path.read_text()); row['feature_path'] = '../outside.pt'; path.write_text(json.dumps(row))
    no_cli(monkeypatch)
    result = backend_status.run_cached_feature_experiment(ROOT, **inputs)
    assert result.code != 0
    assert result.payload['cli_launched'] is False
    assert result.payload['phase'] == 'preflight'


def test_no_device_fallback(inputs, monkeypatch):
    import torch
    inputs['device'] = 'cuda:0'
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    no_cli(monkeypatch)
    out = backend_status.run_cached_feature_experiment(ROOT, **inputs)
    assert out.code == 2 and 'no automatic fallback' in out.diagnostics


def test_check_readiness_is_not_execution(inputs, monkeypatch):
    calls=[]
    monkeypatch.setattr(backend_status.subprocess, 'run', lambda cmd, **k: (calls.append(cmd) or SimpleNamespace(returncode=0, stdout='{"ok":true}', stderr='')))
    out = service.validate_experiment_readiness(ROOT, **inputs)
    assert out.code == 0 and service.readiness_is_current(out)
    assert len(calls) == 1 and 'validate-config' in calls[0]
    assert 'run' not in calls[0]
    assert 'inference has not been tested' in out.diagnostics
    (Path(inputs['feature_root']) / 'c.pt').write_bytes(b'changed')
    assert not service.readiness_is_current(out)


def test_real_forward_smoke_uses_existing_predict_contract(inputs, monkeypatch):
    calls=[]
    def invoke(cmd, **kwargs):
        calls.append(cmd)
        obj = {'ok': True} if 'validate-config' in cmd else dict(status='complete', published=True, n_clips=1, split='val', completed_modes=inputs['modes'])
        return SimpleNamespace(returncode=0 if len(calls)==1 else 4, stdout=json.dumps(obj), stderr='')
    monkeypatch.setattr(backend_status.subprocess, 'run', invoke)
    out = service.validate_experiment_readiness(ROOT, smoke=True, **inputs)
    assert out.code == 0 and out.payload['smoke_passed']
    assert 'predict' in calls[1] and '--max-clips' in calls[1]
    assert calls[1][calls[1].index('--split')+1] == 'val'
    assert '--require-lavdf-hash' in calls[1]
    assert not {'--unseal-test', '--overwrite'} & set(calls[1])


@pytest.mark.parametrize('stdout,code,status', [('{}',0,'invalid_response'), ('{"status":"partial","finalized":false}',0,'invalid_response'), ('',4,'incomplete'), ('',7,'checkpoint_mismatch')])
def test_execution_states_are_honest(inputs, monkeypatch, stdout, code, status):
    calls=[]
    def invoke(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0 if len(calls)==1 else code, stdout='{"ok":true}' if len(calls)==1 else stdout, stderr='')
    monkeypatch.setattr(backend_status.subprocess, 'run', invoke)
    out = backend_status.run_cached_feature_experiment(ROOT, **inputs)
    assert out.payload['phase'] == 'execution' and out.payload['cli_launched']
    assert out.payload['status'] == status
    assert out.code != 0


def test_backend_run_timeout_keeps_phase(inputs, monkeypatch):
    import subprocess
    def invoke(cmd, **kwargs):
        if 'run' in cmd: raise subprocess.TimeoutExpired(cmd, 1)
        return SimpleNamespace(returncode=0, stdout='{"ok":true}', stderr='')
    monkeypatch.setattr(backend_status.subprocess, 'run', invoke)
    out = backend_status.run_cached_feature_experiment(ROOT, **inputs)
    assert out.code == 124 and out.payload['phase'] == 'execution'
    assert out.payload['status'] == 'timeout'
