import numpy as np
import pytest

from avfuse.member2_data.evaluate import bce
from avfuse.member2_data.paired import compare, measured_extra_ms, paired_delta, restoration_table
from avfuse.member2_data.splits import group_clusters

from synthetic import manifest, predictions  # noqa: E402


def test_identical_runs_give_zero():
    y = np.array([0, 1, 0, 1]); s = np.array([.2, .7, .4, .9])
    d = paired_delta(bce, y, s, s, ["a", "a", "b", "b"], n_boot=100)
    assert d["delta"] == 0 and d["ci_low"] == 0 and not d["excludes_zero"]


def test_low_precision_damage_detected():
    m = manifest()
    c = compare(m, predictions(m), "reference", "low_precision", group_clusters(m), "val", 0.5, n_boot=300)
    assert c["metrics"]["bce"]["delta"] > 0 and c["metrics"]["bce"]["excludes_zero"]


def test_restoration_table_identifies_value_output():
    m = manifest()
    bench = [{"run_id": "x", "config": cfg, "device": "cpu", "backend": "sdpa", "scope": "cached_feature_model",
              "batch_size": 1, "repeats": 50, "p50_ms": ms, "p95_ms": ms * 1.2, "peak_memory_bytes": None}
             for cfg, ms in (("reference", 2.0), ("low_precision", 2.5), ("restore_value_output", 2.9),
                             ("restore_query", 2.6))]
    t = restoration_table(m, predictions(m), group_clusters(m), bench, n_boot=300)
    rows = {r["config"]: r for r in t["rows"]}
    vo, q = rows["restore_value_output"], rows["restore_query"]
    assert vo["positive_evidence"] and not q["positive_evidence"]
    assert vo["R"] == pytest.approx(t["L_low_precision"] - vo["L"])
    assert vo["R_per_type"]["audio_fake"] > vo["R_per_type"]["video_fake"]
    assert vo["extra_p50_ms"] == pytest.approx(0.4) and vo["R_per_ms"] == pytest.approx(vo["R"] / 0.4)
    assert "label-grounded support" in t["verdict"]


def test_costs_from_mixed_devices_rejected():
    from avfuse.member2_data.predictions import ContractError
    b = [{"config": "low_precision", "device": "cpu", "scope": "cached_feature_model", "p50_ms": 1},
         {"config": "reference", "device": "cuda", "scope": "cached_feature_model", "p50_ms": 1}]
    with pytest.raises(ContractError):
        measured_extra_ms(b)


def test_by_type_uses_one_fake_type_plus_all_real():
    from avfuse.member2_data.paired import compare_by_type
    m = manifest()
    p = predictions(m)
    bt = compare_by_type(m, p, "reference", "low_precision", group_clusters(m), "val", 0.5, n_boot=200)
    assert set(bt) == {"audio_fake", "video_fake", "both_fake"}
    val = [r for r in m if r["split"] == "val"]
    n_real = sum(r["manipulation_type"] == "real" for r in val)
    for ft, c in bt.items():
        assert c["fake_type"] == ft and c["n_real"] == n_real
        assert c["n_fake"] == sum(r["manipulation_type"] == ft for r in val)
        assert set(c["metrics"]) == {"auroc", "auprc", "bce", "recall", "real_fpr"}
        d = c["metrics"]["bce"]
        assert d["ci_low"] <= d["delta"] <= d["ci_high"]


def test_by_type_point_estimate_matches_manual_subset():
    from avfuse.member2_data.paired import aligned, compare
    m = manifest()
    p = predictions(m)
    c = compare(m, p, "reference", "restore_value_output", group_clusters(m), "val", 0.5, n_boot=50, fake_type="video_fake")
    y, ids, t, sc = aligned(m, p, ["reference", "restore_value_output"], "val")
    k = (t == "real") | (t == "video_fake")
    assert c["metrics"]["bce"]["delta"] == pytest.approx(bce(y[k], sc["restore_value_output"][k]) - bce(y[k], sc["reference"][k]))
    # the real-clip FPR is the same set of real clips in every subset
    assert c["metrics"]["real_fpr"]["a"] == pytest.approx(float((sc["reference"][t == "real"] >= 0.5).mean()))


def test_identical_configs_give_zero_in_every_type():
    from avfuse.member2_data.paired import compare_by_type
    m = manifest()
    p = [r for r in predictions(m) if r["config"] == "reference"]
    twin = [{**r, "config": "low_precision"} for r in p]
    for c in compare_by_type(m, p + twin, "reference", "low_precision", group_clusters(m), "val", 0.5, n_boot=50).values():
        assert all(d["delta"] == 0 and not d["excludes_zero"] for d in c["metrics"].values())
