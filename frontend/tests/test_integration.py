"""Synthetic contract and wrapper checks against the consolidated backend."""
import csv
import io
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from avfuse.shared.contracts import PRECISION_MODES
from avfuse.member1_fusion.dimodif_adapter import CHECKPOINTS
from avfuse.member3_runtime.cli import build_parser
from avfuse_ui import backend_status as service
from avfuse_ui.data_access import (
    CONFIGS, authorized_path, feature_file_summary_bytes, saved_result_table,
    validate_predictions, prediction_summary, validate_benchmarks,
)

ROOT = Path(__file__).resolve().parents[2]


def test_modes_and_cli_contract():
    assert CONFIGS == tuple(PRECISION_MODES)
    for command in service.REQUIRED_COMMANDS:
        args = [command, *{"predict": ["--output", "unused"], "inspect": ["unused"]}.get(command, [])]
        assert build_parser().parse_args(args).command == command


@pytest.mark.parametrize('dirty,expected', [('', False), (' M frontend/app.py', True), (None, None)])
def test_git_dirty_boolean(monkeypatch, dirty, expected):
    monkeypatch.setattr(service.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0 if dirty is not None else 1, stdout=dirty or ''))
    assert service.git_info(ROOT)['dirty'] is expected


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    monkeypatch.setenv('AVFUSE_ASSET_ROOTS', str(tmp_path))
    upstream = tmp_path / 'upstream'
    ckpt = upstream / 'ckpt/dfd' / (CHECKPOINTS['lavdf'] + '.pth')
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b'fixture, never model loaded')
    ckpt.with_suffix('.json').write_text(json.dumps({'config': dict(task='dfd', max_length=600, d_model=64, nhead=4, d_hid=64, nlayers=5, win_size=1, feature_pyramid=True)}))
    (upstream / 'src').mkdir()
    (upstream / 'src/models.py').write_text('# synthetic source existence fixture')
    from avfuse_ui import workbench_service
    from avfuse.member2_data.readiness import DIMODIF_LAVDF_SHA256
    monkeypatch.setattr(workbench_service, 'checkpoint_digest', lambda p: DIMODIF_LAVDF_SHA256)
    features = tmp_path / 'features'
    features.mkdir()
    torch.save(dict(video_tokens=torch.zeros(2, 768), audio_tokens=torch.zeros(2, 768)), features / "c.pt")
    manifest = tmp_path / 'manifest.jsonl'
    row = dict(clip_id='c', dataset='lavdf', source_id='s', identity_id='i', split='val', label=0, manipulation_type='real', feature_path='c.pt')
    manifest.write_text(json.dumps(row) + '\n')
    return dict(upstream_root=str(upstream), checkpoint_name='lavdf', manifest=str(manifest), feature_root=str(features), feature_format='shared_pt', device='cpu', modes=['reference'], run_id='synthetic-wrapper-check')


@pytest.mark.parametrize('code', range(8))
def test_wrapper_structured_validation_only(monkeypatch, inputs, code):
    calls = []
    def run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0 if len(calls) == 1 else code,
                               stdout=json.dumps({'ok': True, 'status': 'complete', 'finalized': True}), stderr='')
    monkeypatch.setattr(service.subprocess, 'run', run)
    actual, output = service.run_cached_feature_experiment(ROOT, **inputs)
    assert actual == code
    assert len(calls) == 2
    for cmd in calls:
        assert cmd[cmd.index('--split') + 1] == 'val'
        assert cmd[cmd.index('--device') + 1] == 'cpu'
        assert '--require-lavdf-hash' in cmd and '--json' in cmd
        assert not {'--unseal-test', '--purpose', '--protocol', '--overwrite'} & set(cmd)
    assert 'Run directory:' in output


@pytest.mark.parametrize('key,value,code', [
    ('modes', ['invented'], 2), ('device', '', 2), ('device', 'auto', 2),
    ('run_id', '../escape', 2), ('experiment_id', '--unseal-test', 2),
    ('feature_root', '/nonexistent-not-authorized', 6), ('require_lavdf_hash', False, 7),
    ('checkpoint_name', 'avdeepfake1m', 7),
])
def test_wrapper_rejects_before_execution(monkeypatch, inputs, key, value, code):
    def forbidden(*a, **k):
        pytest.fail('Invalid request executed a subprocess')
    monkeypatch.setattr(service.subprocess, 'run', forbidden)
    inputs[key] = value
    assert service.run_cached_feature_experiment(ROOT, **inputs)[0] == code


@pytest.mark.parametrize('case', ['test', 'bad_json', 'missing_checkpoint', 'missing_manifest'])
def test_invalid_assets(monkeypatch, inputs, case):
    if case == 'test':
        p = Path(inputs['manifest']); p.write_text(p.read_text().replace('"val"', '"test"'))
    elif case == 'bad_json':
        Path(inputs['manifest']).write_text('bad')
    elif case == 'missing_manifest':
        Path(inputs['manifest']).unlink()
    else:
        next(Path(inputs['upstream_root']).rglob('*.pth')).unlink()
    monkeypatch.setattr(service.subprocess, 'run', lambda *a, **k: pytest.fail('Executed invalid input'))
    assert service.run_cached_feature_experiment(ROOT, **inputs)[0] in (2, 6)


def test_malformed_cli_json_and_timeout(monkeypatch, inputs):
    monkeypatch.setattr(service.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0, stdout='not-json', stderr=''))
    assert service.run_cached_feature_experiment(ROOT, **inputs)[0] == 1
    def timeout(*a, **k):
        raise subprocess.TimeoutExpired('synthetic', 300)
    monkeypatch.setattr(service.subprocess, 'run', timeout)
    assert service.run_cached_feature_experiment(ROOT, **inputs)[0] == 124


@pytest.mark.parametrize('case', ['keys', 'missing', 'rank', 'width', 'empty', 'int', 'nan', 'inf'])
def test_npz_actual_loader_rejections(case):
    arrays = dict(audio_features=np.zeros((2, 768), dtype='float32'), video_features=np.zeros((2, 768), dtype='float32'))
    if case == 'keys': arrays = {k.replace('features', 'tokens'): v for k, v in arrays.items()}
    elif case == 'missing': del arrays['audio_features']
    else:
        arrays['audio_features'] = {'rank': np.zeros((1, 2, 768)), 'width': np.zeros((2, 7)), 'empty': np.zeros((0, 768)), 'int': np.zeros((2, 768), dtype='int32'), 'nan': np.full((2, 768), np.nan), 'inf': np.full((2, 768), np.inf)}[case]
    stream = io.BytesIO(); np.savez(stream, **arrays)
    with pytest.raises(ValueError): feature_file_summary_bytes(stream.getvalue(), 'cache.npz')


@pytest.mark.parametrize('case', ['keys', 'batch', 'rank', 'int', 'empty', 'nan'])
def test_pt_actual_loader_rejections(case):
    arrays = dict(audio_tokens=torch.zeros(1, 2, 768), video_tokens=torch.zeros(1, 2, 768))
    if case == 'keys': arrays = {'audio_features': arrays['audio_tokens']}
    else:
        arrays['audio_tokens'] = {'batch': torch.zeros(2, 2, 768), 'rank': torch.zeros(2, 768), 'int': torch.zeros(1, 2, 768, dtype=torch.int32), 'empty': torch.zeros(1, 0, 768), 'nan': torch.full((1, 2, 768), float('nan'))}[case]
    stream = io.BytesIO(); torch.save(arrays, stream)
    with pytest.raises((ValueError, KeyError)): feature_file_summary_bytes(stream.getvalue(), 'cache.pt')


def test_authorized_root_and_symlink(tmp_path, monkeypatch):
    outside = tmp_path / 'outside'; outside.mkdir()
    root = tmp_path / 'repo'; root.mkdir()
    (root / 'escape').symlink_to(outside)
    with pytest.raises(ValueError): authorized_path(root, root / 'escape')
    monkeypatch.setenv('AVFUSE_ASSET_ROOTS', str(outside))
    assert authorized_path(root, outside) == outside


@pytest.mark.parametrize('name', ['test_overall_metrics.csv', 'test_manipulation_metrics.csv', 'test_paired_comparisons.csv', 'validation_overall_metrics.csv', 'cpu_model_runtime.csv', 'cpu_model_timing_samples.csv'])
def test_saved_tables_preserve_exact_records(name):
    with (ROOT / 'docs/results' / name).open() as stream:
        original = list(csv.DictReader(stream))
    assert saved_result_table(ROOT, name) == original
    if name == 'test_paired_comparisons.csv':
        for row in original:
            assert float(row['delta']) == pytest.approx(float(row['b']) - float(row['a']))


def test_predictions_group_unrelated_runs():
    row = dict(run_id='r1', split='val', clip_id='c', config='reference', score_fake=.5, checkpoint_sha256='a'*64)
    rows = [row, {**row, 'run_id': 'r2', 'checkpoint_sha256': 'b'*64}]
    assert validate_predictions(rows, ROOT / 'contracts/schemas/prediction.schema.json') == []
    assert len(prediction_summary(rows)) == 2


def test_empty_benchmark_is_not_verified():
    assert validate_benchmarks([], ROOT / 'contracts/schemas/benchmark.schema.json')


def test_backend_run_integrity_detects_corruption(tmp_path):
    from avfuse.member3_runtime.artifacts import create_run_context, verify_run_integrity
    config = json.loads((ROOT / 'configs/benchmark.example.json').read_text())
    ctx = create_run_context(config, runs_root=tmp_path, run_id='synthetic-integrity',
                             repo_root=ROOT, allow_unset_inputs=True)
    ctx.write_json_artifact('fixture.json', {'synthetic': True}, artifact_type='fixture', required=False)
    ctx.finalize()
    assert verify_run_integrity(ctx.run_dir)['ok']
    (ctx.run_dir / 'fixture.json').write_text('{}')
    assert not verify_run_integrity(ctx.run_dir)['ok']


def test_benchmark_missing_external_samples_and_cross_fields(tmp_path):
    from avfuse.member3_runtime.artifacts import create_run_context
    from avfuse.member3_runtime.benchmark import benchmark_callable
    config = json.loads((ROOT / 'configs/benchmark.example.json').read_text())
    ctx = create_run_context(config, runs_root=tmp_path, run_id='synthetic-benchmark',
                             repo_root=ROOT, allow_unset_inputs=True)
    measurement = benchmark_callable(lambda: None, device='cpu', warmup=0, repeats=2)
    row = ctx.build_record(measurement, config='reference', scope='operator', backend='sdpa',
                           batch_size=1, execution_dtype='float32', input_boundary='synthetic_tensors',
                           input_shapes={'q': [1, 1, 2, 4], 'k': [1, 1, 2, 4]})
    schema = ROOT / 'contracts/schemas/benchmark.schema.json'
    assert validate_benchmarks([row], schema) == []
    assert validate_benchmarks([{**row, 'p50_ms': -1}], schema)
    external = dict(row)
    external.pop('timing_samples_ms')
    external['timing_samples_ref'] = 'missing.json'
    assert validate_benchmarks([external], schema)
    assert validate_benchmarks([external], schema, run_dir=ctx.run_dir)


def test_duplicate_run_id_does_not_execute(tmp_path, inputs, monkeypatch):
    root = tmp_path / 'root'; root.mkdir()
    (root / 'artifacts/member3/runs' / inputs['run_id']).mkdir(parents=True)
    monkeypatch.setattr(service.subprocess, 'run', lambda *a, **k: pytest.fail('Duplicate executed'))
    assert service.run_cached_feature_experiment(root, **inputs)[0] == 2
