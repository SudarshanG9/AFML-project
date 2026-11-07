import numpy as np

from avfuse.member2_data.shortcut_control import control_report, matched_subset
from avfuse.member2_data.shortcuts import shortcut_auroc


def test_matching_removes_the_shortcut():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(400), np.ones(400)].astype(int)
    v = np.r_[rng.normal(40, 15, 400), rng.normal(70, 15, 400)]   # fakes have more trailing silence
    assert shortcut_auroc(y, v) > 0.85
    m = matched_subset(y, v, n_bins=10)
    assert (y[m] == 0).sum() == (y[m] == 1).sum() > 50
    assert shortcut_auroc(y[m], v[m]) < 0.62                      # the feature no longer separates


def _rows(y, s):
    return [{"clip_id": f"c{i}", "label": int(t), "score_fake": float(p)} for i, (t, p) in enumerate(zip(y, s))]


def test_detector_using_only_the_shortcut_collapses():
    rng = np.random.default_rng(1)
    y = np.r_[np.zeros(400), np.ones(400)].astype(int)
    v = np.r_[rng.normal(40, 15, 400), rng.normal(70, 15, 400)]
    rows = _rows(y, 1 / (1 + np.exp(-(v - 55) / 5)))               # score depends only on the shortcut
    rep = control_report(rows, {r["clip_id"]: x for r, x in zip(rows, v)}, {r["clip_id"]: r["clip_id"] for r in rows},
                         n_boot=100)
    assert rep["detector_auroc_all"]["value"] > 0.85 and rep["detector_auroc_matched"]["value"] < 0.65


def test_detector_independent_of_the_shortcut_holds():
    rng = np.random.default_rng(2)
    y = np.r_[np.zeros(400), np.ones(400)].astype(int)
    v = np.r_[rng.normal(40, 15, 400), rng.normal(70, 15, 400)]
    s = 1 / (1 + np.exp(-(y * 3 - 1.5 + rng.normal(0, 1, 800))))   # real signal, unrelated to v
    rows = _rows(y, s)
    rep = control_report(rows, {r["clip_id"]: x for r, x in zip(rows, v)}, {r["clip_id"]: r["clip_id"] for r in rows},
                         n_boot=100)
    assert abs(rep["drop"]) < 0.05
