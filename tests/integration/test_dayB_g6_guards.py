"""Audit guards only; no model or cache is loaded."""
import math
import pytest
from scripts.dayB_g6.run_g6 import finite_tree, validate_rows


def predictions():
    return [{'run_id': 'unit', 'clip_id': str(i), 'split': 'val', 'config': 'low_precision',
             'score_fake': 0.5, 'checkpoint_sha256': 'approved'} for i in range(1200)]


def test_exact_cohort_accepted():
    rows = predictions()
    validate_rows(rows, {str(i) for i in range(1200)}, 'low_precision', 'approved')


@pytest.mark.parametrize('mutation', ['duplicate', 'subset', 'extra', 'checkpoint', 'test', 'label', 'score', 'config'])
def test_invalid_evidence_rejected(mutation):
    rows = predictions()
    if mutation == 'duplicate': rows[0]['clip_id'] = '1'
    elif mutation == 'subset': rows.pop()
    elif mutation == 'extra': rows[0]['clip_id'] = 'outside'
    elif mutation == 'checkpoint': rows[0]['checkpoint_sha256'] = 'different'
    elif mutation == 'test': rows[0]['split'] = 'test'
    elif mutation == 'label': rows[0]['label'] = 1
    elif mutation == 'score': rows[0]['score_fake'] = math.nan
    elif mutation == 'config': rows[0]['config'] = 'restore_query'
    with pytest.raises(RuntimeError):
        validate_rows(rows, {str(i) for i in range(1200)}, 'low_precision', 'approved')


def test_undefined_ci_rejected():
    with pytest.raises(RuntimeError):
        finite_tree({'metrics': [{'ci_low': math.nan}]})
