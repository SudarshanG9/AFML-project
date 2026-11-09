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
