import sys
import pytest
from scripts import dayB_inference


def test_manifest_requires_out_before_touching_model(monkeypatch, capsys):
    def reject_model_load(_args):
        pytest.fail('Argument validation must happen before model loading')
    monkeypatch.setattr(dayB_inference, 'create_adapter', reject_model_load)
    monkeypatch.setattr(sys, 'argv', [
        'dayB_inference.py', '--manifest', 'missing.jsonl',
        '--mode', 'reference',
    ])
    with pytest.raises(SystemExit) as error:
        dayB_inference.main()
    assert error.value.code == 2
    assert '--out is required for manifest inference' in capsys.readouterr().err
