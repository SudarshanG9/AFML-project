"""Run association, missing evidence and safe Member 2 handoff; all fixtures synthetic."""
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from avfuse.member3_runtime.provenance import sha256_file
from avfuse_ui import workbench_service as service
from avfuse_ui.data_access import CONFIGS, saved_result_table
from avfuse_ui.workbench_results import decision_comparison, performance_rows

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def run_fixture(tmp_path):
    from avfuse.member3_runtime.artifacts import create_run_context
    root = tmp_path / 'repo'; root.mkdir()
    shutil.copytree(ROOT / 'contracts', root / 'contracts')
    release = root / 'docs/member1/dayC/releases/M2-REL-20261113-01/m2_release.json'
    release.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / 'docs/member1/dayC/releases/M2-REL-20261113-01/m2_release.json', release)
    manifest = root / 'validation.jsonl'
    rows = [dict(clip_id=f'clip{i}', dataset='lavdf', source_id=f's{i}', identity_id=f'i{i}', split='val',
                 label=int(i>=2), manipulation_type=['real','real','audio_fake','audio_fake','video_fake','video_fake','both_fake','both_fake'][i], feature_path=f'c{i}.pt') for i in range(8)]
    manifest.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    config = json.loads((ROOT / 'configs/benchmark.example.json').read_text())
    ctx = create_run_context(config, runs_root=root / 'artifacts/member3/runs', run_id='synthetic-result-fixture', repo_root=ROOT, allow_unset_inputs=True, environment_report={})
    preds = [dict(run_id=ctx.run_id, split='val', clip_id=r['clip_id'], config='reference', score_fake=.1 if not r['label'] else .9, checkpoint_sha256='a'*64) for r in rows]
    ctx.write_json_artifact('fixture_note.json', {'note': 'synthetic test fixture only'}, artifact_type='fixture', required=False)
    pred_path = ctx.run_dir / 'predictions.jsonl'
    pred_path.write_text(''.join(json.dumps(p)+'\n' for p in preds))
    ctx.register('predictions.jsonl', 'avfuse_predictions')
    summary = dict(run_id=ctx.run_id, split='val', checkpoint_sha256='a'*64, status='complete', dataset_complete=True,
                   requested_modes=['reference'], data_origin='real_cached_features',
                   features=dict(manifest=str(manifest), manifest_sha256=sha256_file(manifest)),
                   predictions=dict(n_rows=8), predictions_sha256=sha256_file(pred_path))
    ctx.write_json_artifact('execution_summary.json', summary, artifact_type='avfuse_execution_summary')
    ctx.finalize()
    protocol = root / 'protocol.json'
    protocol.write_text(json.dumps(dict(calibration_split='val', calibration_config='reference', threshold=.6627789735794068,
                                       manifest_sha256=sha256_file(manifest), checkpoint_sha256='a'*64)))
    return root, ctx.run_dir, protocol


def test_run_results_missing_modes_evaluation_and_integrity(run_fixture):
    root, run, _ = run_fixture
    out = service.load_experiment_results(root, run)
    assert out['completion_recorded'] and len(out['predictions']) == 8
    assert out['metrics'] is None and not out['benchmarks']
    assert out['missing_modes'] == list(CONFIGS[1:])
    from avfuse_ui.run_inspection import verify_with_receipt
    assert verify_with_receipt(run)['verdict']['ok']


@pytest.mark.parametrize('case', ['other_run', 'checkpoint', 'score', 'split', 'malformed', 'synthetic'])
def test_unrelated_or_invalid_scores_are_not_displayed(run_fixture, case):
    root, run, _ = run_fixture
    p = run / 'predictions.jsonl'
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    if case == 'other_run': rows[0]['run_id'] = 'unrelated'
    if case == 'checkpoint': rows[0]['checkpoint_sha256'] = 'b'*64
    if case == 'score': rows[0]['score_fake'] = 1.1
    if case == 'split': rows[0]['split'] = 'test'
    if case == 'synthetic':
        path=run/'execution_summary.json'; summary=json.loads(path.read_text()); summary['data_origin']='synthetic_smoke'; path.write_text(json.dumps(summary))
    elif case == 'malformed': p.write_text('bad')
    else: p.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    out=service.load_experiment_results(root, run)
    assert not out['predictions'] and out['errors']


@pytest.mark.parametrize('state', ['missing_summary', 'failed', 'partial', 'missing_manifest'])
def test_incomplete_run_never_establishes_completion(run_fixture, state):
    root, run, _ = run_fixture
    if state == 'missing_summary': (run/'execution_summary.json').unlink()
    elif state == 'missing_manifest': (run/'run_manifest.json').unlink()
    elif state == 'failed': (run/'failure_summary.json').write_text('{"status":"failed"}')
    else:
        path=run/'execution_summary.json'; summary=json.loads(path.read_text()); summary['status']='partial'; path.write_text(json.dumps(summary))
    out=service.load_experiment_results(root, run)
    assert not out['completion_recorded']


def test_decision_comparison_missing_reference_or_modes_stays_missing():
    rows=[dict(config='reference', clip_id='c', score_fake=.5), dict(config='low_precision', clip_id='c', score_fake=.7)]
    result=decision_comparison(rows, 'c', .6)
    assert result[1]['Δ from FP32'] == pytest.approx(.2)
    assert result[1]['Decision flip vs FP32'] is True
    assert result[2]['Fake score'] is None
    assert decision_comparison(rows[1:], 'c')[1]['Δ from FP32'] is None
    assert 'Decision flip vs FP32' not in decision_comparison(rows, 'c')[0]


def test_performance_missing_recall_not_invented():
    result=performance_rows(dict(reference=dict(auroc={'value':.5}, auprc={'value':.6}, bce={'value':.7})))
    assert result[0]['RECALL'] is None


@pytest.mark.parametrize('field,value', [('manifest_sha256','b'*64), ('checkpoint_sha256','b'*64), ('threshold', .5), ('calibration_split','test')])
def test_protocol_mismatch_cannot_launch_evaluator(run_fixture, monkeypatch, field, value):
    root, run, protocol=run_fixture
    data=json.loads(protocol.read_text()); data[field]=value; protocol.write_text(json.dumps(data))
    monkeypatch.setattr('subprocess.run', lambda *a, **k: pytest.fail('invalid evaluation launched'))
    out=service.evaluate_validation_results(root, run, str(protocol))
    assert out.code and out.payload['cli_launched'] is False
    assert 'No recalibration' in out.diagnostics


def test_missing_labels_cannot_launch_evaluator(run_fixture, monkeypatch):
    root, run, protocol=run_fixture
    manifest=root/'validation.jsonl'
    rows=[json.loads(line) for line in manifest.read_text().splitlines()]
    rows[0].pop('label'); manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    monkeypatch.setattr('subprocess.run', lambda *a, **k: pytest.fail('unlabelled evaluation launched'))
    out=service.evaluate_validation_results(root, run, str(protocol))
    assert out.code and not out.payload['cli_launched']


def test_member2_handoff_arguments_and_receipt_binding(run_fixture, monkeypatch):
    root, run, protocol=run_fixture
    calls=[]
    metric=dict(value=.8, ci_low=.7, ci_high=.9)
    def invoke(command, **kwargs):
        calls.append(command)
        output=Path(command[command.index('--out')+1]); output.mkdir(parents=True)
        report=dict(split='val', configs={'reference':dict(auroc=metric,auprc=metric,bce=metric,per_type={})},
                    manifest_sha256=sha256_file(root/'validation.jsonl'), threshold=.6627789735794068, n_boot=1000, seed=0)
        (output/'metrics.json').write_text(json.dumps(report))
        return SimpleNamespace(returncode=0, stdout='fixture evaluator', stderr='')
    monkeypatch.setattr('subprocess.run', invoke)
    out=service.evaluate_validation_results(root, run, str(protocol))
    assert out.code == 0
    command=calls[0]
    assert 'evaluate' in command and command[command.index('--split')+1]=='val'
    assert not {'--unseal-test','calibrate','--purpose'} & set(command)
    assert (run/'run_manifest.json').is_file()
    report=service.load_experiment_results(root, run)
    assert report['metrics'] is not None
    (Path(out.payload['output'])/'metrics.json').write_text('{}')
    assert service.load_experiment_results(root, run)['metrics'] is None


def test_actual_member2_validation_evaluator_on_synthetic_fixture(run_fixture):
    root, run, protocol=run_fixture
    (root/'scripts').symlink_to(ROOT/'scripts', target_is_directory=True)
    (root/'src').symlink_to(ROOT/'src', target_is_directory=True)
    out=service.evaluate_validation_results(root, run, str(protocol))
    assert out.code == 0, out.diagnostics
    report=service.load_experiment_results(root, run)['metrics']
    assert report is not None and report['split']=='val'
    assert set(report['configs']) == {'reference'}
    assert report['configs']['reference']['n'] == 8
    from avfuse_ui.run_inspection import verify_with_receipt
    assert verify_with_receipt(run)['verdict']['ok']  # evaluation did not mutate the bundle


def test_saved_results_compact_table_chart_and_source_labels():
    app=AppTest.from_file(str(ROOT/'frontend/app.py'), default_timeout=30).run()
    app.sidebar.radio[0].set_value('Results').run()
    assert not app.exception
    assert len(app.dataframe[0].value)==4
    expected=next(float(row['value']) for row in saved_result_table(ROOT,'test_overall_metrics.csv') if row['mode']=='reference' and row['metric']=='auroc')
    assert app.dataframe[0].value.iloc[0]['AUROC']==expected
    assert len(app.get('plotly_chart')) == 1
    assert any('No new inference' in i.value for i in app.info)
    app.radio(key='result_source').set_value('New validation run').run()
    assert not app.exception
    assert not app.dataframe and app.info


def test_new_run_ui_absent_evaluation_and_missing_configs(run_fixture):
    root, run, _=run_fixture
    app=AppTest.from_string(f"from pathlib import Path\nfrom avfuse_ui.workbench_results import render\nfrom avfuse_ui.data_access import discover_files\nroot=Path({str(root)!r})\nrender(root, discover_files(root))", default_timeout=30).run()
    app.radio(key='result_source').set_value('New validation run').run()
    assert not app.exception
    assert any('Label-grounded evaluation has not been performed' in i.value for i in app.info)
    assert any('Missing configuration' in str(df.value) for df in app.dataframe)


@pytest.mark.parametrize('case', ['timeout', 'nonzero', 'malformed'])
def test_evaluator_failures_remain_absent(run_fixture, monkeypatch, case):
    import subprocess
    root, run, protocol=run_fixture
    def invoke(command, **kwargs):
        if case == 'timeout': raise subprocess.TimeoutExpired(command, 1)
        if case == 'malformed':
            output=Path(command[command.index('--out')+1]); output.mkdir(parents=True)
            (output/'metrics.json').write_text('{}')
        return SimpleNamespace(returncode=2 if case=='nonzero' else 0, stdout='', stderr='fixture failure')
    monkeypatch.setattr('subprocess.run', invoke)
    out=service.evaluate_validation_results(root, run, str(protocol))
    assert out.code != 0 and out.payload['phase']=='evaluation'
    assert service.load_experiment_results(root, run)['metrics'] is None


@pytest.mark.parametrize('score', [float('nan'), float('inf'), -.1, 1.1, True])
def test_nonfinite_or_invalid_probabilities_rejected(score):
    from avfuse_ui.data_access import validate_predictions
    row=dict(run_id='r', split='val', clip_id='c', config='reference', score_fake=score, checkpoint_sha256='a'*64)
    assert validate_predictions([row], ROOT/'contracts/schemas/prediction.schema.json')


@pytest.mark.parametrize('field,value', [('recall', None), ('recall', {'value':'bad','ci_low':0,'ci_high':1}), ('auroc',{'value':1.2,'ci_low':1,'ci_high':1.3})])
def test_malformed_metric_blocks_rejected(field, value):
    from avfuse_ui.data_access import validate_metrics_report
    metric=dict(value=.5,ci_low=.4,ci_high=.6)
    block=dict(auroc=metric,auprc=metric,bce=metric,per_type={}); block[field]=value
    assert validate_metrics_report(dict(split='val',configs={'reference':block}))
