"""Synthetic run states must remain inspectable without implicit hashing or inference."""
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from avfuse_ui.run_inspection import optional_metadata, verify_with_receipt, receipt_is_current

ROOT = Path(__file__).resolve().parents[2]


def make_run(tmp_path):
    from avfuse.member3_runtime.artifacts import create_run_context
    cfg = json.loads((ROOT / 'configs/benchmark.example.json').read_text())
    ctx = create_run_context(cfg, runs_root=tmp_path / 'artifacts/runs', run_id='synthetic-phase5',
                             repo_root=ROOT, allow_unset_inputs=True, environment_report={})
    ctx.write_json_artifact('execution_summary.json', {'status': 'complete'}, artifact_type='fixture', required=False)
    ctx.finalize()
    return ctx.run_dir


def render(tmp_path):
    source = f'''from pathlib import Path
from avfuse_ui.ui import render_runs
from avfuse_ui.data_access import discover_files
root = Path({str(tmp_path)!r})
render_runs(root, discover_files(root))
'''
    app = AppTest.from_string(source, default_timeout=30).run()
    app.selectbox(key='run_select').select('artifacts/runs/synthetic-phase5').run()
    return app


@pytest.mark.parametrize('state', ['empty', 'incomplete', 'failed', 'finalized', 'corrupt',
                                  'bad_manifest', 'bad_summary', 'bad_optional', 'missing_optional'])
def test_run_states_preserve_stage_explorer(tmp_path, monkeypatch, state):
    if state in ('empty', 'incomplete', 'failed'):
        run = tmp_path / 'artifacts/runs/synthetic-phase5'; run.mkdir(parents=True)
        if state == 'failed': (run / 'failure_summary.json').write_text('{"status":"failed"}')
        if state == 'incomplete': (run / 'execution_summary.json').write_text('{"status":"partial"}')
    else:
        run = make_run(tmp_path)
        if state == 'corrupt': (run / 'execution_summary.json').write_text('{"status":"partial"}')
        if state == 'bad_manifest': (run / 'run_manifest.json').write_text('broken')
        if state == 'bad_summary': (run / 'execution_summary.json').write_text('broken')
        if state == 'bad_optional': (run / 'provenance.json').write_text('broken')
        if state == 'missing_optional': (run / 'execution_summary.json').unlink()
    from avfuse.member3_runtime import artifacts
    monkeypatch.setattr(artifacts, 'verify_run_integrity', lambda *a: pytest.fail('Implicit verification'))
    app = render(tmp_path)
    assert not app.exception
    assert app.selectbox(key='stage_profile_select') is not None
    assert any('Not verified' in item.value for item in app.info)
    assert not app.success
    if state.startswith('bad_'):
        assert any('Could not read' in item.value for item in app.error)
    if state == 'bad_summary': assert (run / 'execution_summary.json').read_text() == 'broken'


@pytest.mark.parametrize('filename', ['execution_summary.json','failure_summary.json','provenance.json','resolved_config.json'])
def test_unreadable_optional_metadata_keeps_page_usable(tmp_path, monkeypatch, filename):
    run = make_run(tmp_path)
    (run / filename).write_text('{}')
    from avfuse_ui import run_inspection
    real = run_inspection.load_json
    def read(path):
        if path.name == filename: raise PermissionError('synthetic unreadable file')
        return real(path)
    monkeypatch.setattr(run_inspection, 'load_json', read)
    app = render(tmp_path)
    assert not app.exception
    assert any(filename in item.value and 'PermissionError' in item.value for item in app.error)
    assert app.selectbox(key='stage_profile_select') is not None


@pytest.mark.parametrize('corrupt', [False, True])
def test_explicit_integrity_action_and_past_receipt(tmp_path, corrupt):
    run = make_run(tmp_path)
    if corrupt: (run / 'execution_summary.json').write_text('{}')
    app = render(tmp_path)
    app.button(key='verify_run').click().run()
    assert not app.exception
    assert app.selectbox(key='stage_profile_select') is not None
    if corrupt: assert any('integrity failed' in item.value for item in app.error)
    else:
        assert any('integrity passed' in item.value for item in app.success)
        (run / 'execution_summary.json').write_text('{"status":"partial"}')
        app.run()
        assert any('Not verified for current files' in item.value for item in app.warning)
        assert not app.success


def test_receipt_detects_changes_without_rehashing(tmp_path):
    run = make_run(tmp_path)
    receipt = verify_with_receipt(run)
    assert receipt['verdict']['ok']
    assert receipt_is_current(run, receipt)
    (run / 'new.json').write_text('{}')
    assert not receipt_is_current(run, receipt)


def test_malformed_stage_entry_does_not_crash(tmp_path):
    run = make_run(tmp_path)
    (run / 'stage_profile.json').write_text('{"stages":[null,1,"bad"]}')
    app = render(tmp_path)
    app.selectbox(key='stage_profile_select').select('artifacts/runs/synthetic-phase5/stage_profile.json').run()
    assert not app.exception
    assert any('Invalid stage_profile.json entry' in item.value for item in app.error)


def test_optional_metadata_no_replacement(tmp_path):
    p = tmp_path / 'bad.json'; p.write_text('not-json')
    value, error = optional_metadata(p)
    assert value is None and 'bad.json' in error
    assert p.read_text() == 'not-json'


def test_discovery_refreshes_all_runtime_artifacts(tmp_path):
    from avfuse_ui.data_access import discover_files
    assert not discover_files(tmp_path)['run_dirs']
    run = make_run(tmp_path)
    (run / 'predictions.jsonl').write_text('{}\n')
    (run / 'stage_profile.json').write_text('{}')
    files = discover_files(tmp_path)
    assert run in files['run_dirs']
    assert run / 'predictions.jsonl' in files['predictions']
    assert run / 'stage_profile.json' in files['stage_profiles']
    assert run / 'environment_report.json' in files['environment_reports']


def test_interface_failure_never_means_runnable(monkeypatch):
    from avfuse_ui.backend_status import inspect_backend
    from avfuse.member3_runtime import cli
    monkeypatch.setattr(cli, 'build_parser', lambda: (_ for _ in ()).throw(AttributeError('synthetic parser failure')))
    state = inspect_backend(ROOT)
    assert state['source_available']
    assert not state['interfaces_imported']
    assert not state['execution_interface_available']
    assert state['assets_available'].startswith('Not checked')


def test_current_repository_instructions_match_app():
    text = (ROOT / 'README.md').read_text()
    assert 'Frontend integration and full final-release validation remain deferred' not in text
    assert 'The frontend is untouched' not in text
    app = (ROOT / 'frontend/app.py').read_text()
    assert 'AVFUSE_REPO_ROOT' in app
    ui = (ROOT / 'frontend/avfuse_ui/ui.py').read_text()
    assert 'edit the root in the sidebar' not in ui
    assert 'does not transmit dataset clips or experiment files to a server' not in ui
