"""Render all pages without loading private models or executing experiments."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]
PAGES = ['Overview', 'Dataset & features', 'Fusion diagnostics', 'Demo & predictions',
         'Evaluation', 'Benchmarks', 'Runs & system', 'Artifact import']


@pytest.mark.parametrize('page', PAGES)
def test_every_page_renders_without_model_assets(page, monkeypatch):
    from avfuse_ui import backend_status
    def forbidden(*args, **kwargs):
        pytest.fail('Page render attempted model execution')
    monkeypatch.setattr(backend_status, 'run_cached_feature_experiment', forbidden)
    app = AppTest.from_file(str(ROOT / 'frontend/app.py'), default_timeout=30).run()
    assert not app.exception
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception
    if page == 'Evaluation':
        assert sum(len(table.value) for table in app.dataframe) == 20 + 36 + 80 + 20
    if page == 'Benchmarks':
        assert sum(len(table.value) for table in app.dataframe) == 12 + 600


@pytest.mark.parametrize('page,label', [('Evaluation', 'Imported evaluation artifact'), ('Benchmarks', 'Imported benchmark records')])
def test_empty_import_mode_does_not_mix_saved_results(page, label):
    app = AppTest.from_file(str(ROOT / 'frontend/app.py'), default_timeout=30).run()
    app.sidebar.radio[0].set_value(page).run()
    next(r for r in app.radio if r.label.endswith('source')).set_value(label).run()
    assert not app.exception
    assert not app.dataframe
    assert app.info


def test_refresh_and_dirty_status():
    app = AppTest.from_file(str(ROOT / 'frontend/app.py'), default_timeout=30).run()
    next(b for b in app.button if b.label == 'Refresh discovered artifacts').click().run()
    assert not app.exception
    assert any('Working tree dirty:' in item.value and 'unknown' not in item.value.split('Working tree dirty:')[1] for item in app.markdown)


def test_malformed_multi_file_import_blocks_comparison(tmp_path, monkeypatch):
    import json
    from avfuse_ui import ui
    good = tmp_path / 'good.jsonl'
    good.write_text(json.dumps(dict(run_id='synthetic', split='val', clip_id='c', config='reference', score_fake=.5, checkpoint_sha256='a'*64))+'\n')
    bad = tmp_path / 'broken.jsonl'; bad.write_text('not json')
    monkeypatch.setattr(ui, 'choose_many_local_or_upload', lambda *a, **k: ([good, bad], None))
    app = AppTest.from_file(str(ROOT / 'frontend/app.py'), default_timeout=30).run()
    app.sidebar.radio[0].set_value('Demo & predictions').run()
    assert not app.exception
    assert any('broken.jsonl' in item.value for item in app.error)
    assert not app.success
    assert not app.dataframe


def test_imported_metrics_are_not_replaced_by_saved_tables(tmp_path, monkeypatch):
    import json
    from avfuse_ui import ui
    path = tmp_path / 'metrics.json'
    metric = dict(value=.5, ci_low=.4, ci_high=.6)
    path.write_text(json.dumps(dict(split='val', configs={'reference': dict(auroc=metric, auprc=metric, bce=metric, per_type={})})))
    monkeypatch.setattr(ui, 'choose_local_or_upload', lambda *a, **k: (path, None))
    app = AppTest.from_file(str(ROOT / 'frontend/app.py'), default_timeout=30).run()
    app.sidebar.radio[0].set_value('Evaluation').run()
    next(r for r in app.radio if r.label == 'Evaluation source').set_value('Imported evaluation artifact').run()
    assert not app.exception
    assert len(app.dataframe) == 1
    assert len(app.dataframe[0].value) == 1
    assert any(str(path) in caption.value for caption in app.caption)


def test_malformed_metrics_object_has_useful_error(tmp_path, monkeypatch):
    from avfuse_ui import ui
    path = tmp_path / 'metrics.json'; path.write_text('{"split":"val","configs":{"reference":null}}')
    monkeypatch.setattr(ui, 'choose_local_or_upload', lambda *a, **k: (path, None))
    app = AppTest.from_file(str(ROOT / 'frontend/app.py'), default_timeout=30).run()
    app.sidebar.radio[0].set_value('Evaluation').run()
    next(r for r in app.radio if r.label == 'Evaluation source').set_value('Imported evaluation artifact').run()
    assert not app.exception
    assert app.error
    assert not app.success
