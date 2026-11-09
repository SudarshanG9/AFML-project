import pytest

import synthetic
from avfuse.member2_data.predictions import ContractError
from avfuse.member2_data.readiness import check_predictions, train_overlap


def _setup(configs=("reference",)):
    m = synthetic.manifest()
    return m, synthetic.predictions(m, configs)


def test_good_predictions_pass_with_orientation_ok():
    m, p = _setup()
    rep = check_predictions(m, p, expected_sha=synthetic.SYNTHETIC_SHA, n_boot=100)
    r = rep["configs"]["reference"]["val"]
    assert r["orientation"] == "ok"
    assert set(r["auroc_per_type_vs_real"]) == {"audio_fake", "video_fake", "both_fake"}
    assert rep["configs"]["reference"]["test"]["n"] > 0 and "auroc" not in rep["configs"]["reference"]["test"]


def test_inverted_scores_are_flagged():
    m, p = _setup()
    for r in p:
        r["score_fake"] = 1 - r["score_fake"]
    rep = check_predictions(m, p, expected_sha=synthetic.SYNTHETIC_SHA, n_boot=100)
    assert rep["configs"]["reference"]["val"]["orientation"] == "INVERTED"


def test_wrong_checkpoint_rejected():
    m, p = _setup()
    with pytest.raises(ContractError, match="agreed checkpoint"):
        check_predictions(m, p, n_boot=10)


def test_missing_and_duplicate_rejected():
    m, p = _setup()
    val = [r for r in p if r["split"] == "val"]
    with pytest.raises(ContractError, match="no prediction"):
        check_predictions(m, val[1:], expected_sha=synthetic.SYNTHETIC_SHA, n_boot=10)
    with pytest.raises(ContractError, match="predicted twice"):
        check_predictions(m, val + val[:1], expected_sha=synthetic.SYNTHETIC_SHA, n_boot=10)


def test_label_field_in_predictions_rejected():
    m, p = _setup()
    p[0]["label"] = 1
    with pytest.raises(ContractError):
        check_predictions(m, p, expected_sha=synthetic.SYNTHETIC_SHA, n_boot=10)


def _meta():
    return [
        {"file": "train/000001.mp4", "split": "train"},
        {"file": "train/000002.mp4", "split": "train", "original": "train/000001.mp4"},
        {"file": "dev/000010.mp4", "split": "dev"},
        {"file": "dev/000011.mp4", "split": "dev", "original": "dev/000010.mp4"},
    ]


def _row(cid, src, split="val", t="real"):
    return {"clip_id": cid, "source_id": src, "split": split, "manipulation_type": t}


def test_overlap_clean():
    rep = train_overlap(_meta(), [_row("a", "lavdf:000010"), _row("b", "lavdf:000010", t="both_fake")])
    assert rep["clean"] and rep["splits"]["val"]["n_sources"] == 1


def test_overlap_detected():
    rep = train_overlap(_meta(), [_row("a", "lavdf:000010"), _row("c", "lavdf:000001", "test", "audio_fake")])
    assert not rep["clean"]
    assert rep["splits"]["test"]["videos_whose_source_is_in_train"] == 1
    assert rep["splits"]["test"]["by_type"] == {"audio_fake": 1}
