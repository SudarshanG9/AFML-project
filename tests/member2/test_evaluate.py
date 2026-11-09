import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from avfuse.member2_data.evaluate import cluster_resamples, evaluate_config, threshold_at_fpr
from avfuse.member2_data.predictions import ContractError, by_config, join, validate_predictions
from avfuse.member2_data.splits import group_clusters
from avfuse.shared.jsonl import write_jsonl

from synthetic import manifest, predictions  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
Y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
S = np.array([.10, .80, .40, .35, .20, .90, .60, .70, .05, .55])


def test_threshold_hand_example():
    t = threshold_at_fpr(Y, S, 0.2)   # 1 of 5 reals may be flagged -> just above 0.40
    assert 0.40 < t["threshold"] < 0.4000001 and t["achieved_fpr"] == 0.2 and t["recall"] == 0.8
    t0 = threshold_at_fpr(Y, S, 0.0)  # just above 0.60
    assert t0["achieved_fpr"] == 0 and t0["recall"] == pytest.approx(0.6)


def test_threshold_never_exceeds_target():
    rng = np.random.default_rng(1)
    for _ in range(200):
        y, s = rng.integers(0, 2, 60), np.round(rng.random(60), 2)
        if (y == 0).any():
            for target in (0.0, 0.01, 0.05, 0.2):
                assert threshold_at_fpr(y, s, target)["achieved_fpr"] <= target + 1e-12


def test_cluster_resamples_keep_groups_whole():
    c = np.array(list("aabcccd"))
    size = {k: (c == k).sum() for k in set(c)}
    for idx in cluster_resamples(c, 100):
        for k in set(c[idx]):
            assert (c[idx] == k).sum() % size[k] == 0


def test_evaluate_config_matches_sklearn_and_strata():
    m = manifest()
    rows = join(m, by_config(predictions(m))["reference"], "val")
    res = evaluate_config(rows, group_clusters(m), threshold=0.5, n_boot=100)
    y = [r["label"] for r in rows]
    s = [r["score_fake"] for r in rows]
    assert res["auroc"]["value"] == pytest.approx(roc_auc_score(y, s))
    assert res["auprc"]["value"] == pytest.approx(average_precision_score(y, s))
    real = [r for r in rows if r["label"] == 0]
    af = [r for r in rows if r["manipulation_type"] == "audio_fake"]
    assert res["per_type"]["audio_fake"]["n_real"] == len(real)  # vs ALL reals
    assert res["per_type"]["audio_fake"]["auroc"]["value"] == pytest.approx(
        roc_auc_score([0] * len(real) + [1] * len(af), [r["score_fake"] for r in real + af]))


def test_predictions_contract():
    m = manifest(20)
    p = predictions(m)
    validate_predictions(p)
    with pytest.raises(ContractError, match="Additional properties"):
        validate_predictions([dict(p[0], label=1)])          # labels can't ride along
    with pytest.raises(ContractError):
        validate_predictions([dict(p[0], score_fake=1.2)])
    with pytest.raises(ContractError, match="twice"):
        validate_predictions(p + p[:1])
    with pytest.raises(ContractError, match="checkpoint"):
        validate_predictions(p[:1] + [dict(p[1], checkpoint_sha256="other")])
    ref = by_config(p)["reference"]
    with pytest.raises(ContractError, match="no prediction"):
        join(m, [r for r in ref if r["clip_id"] != next(x["clip_id"] for x in ref if x["split"] == "val")], "val")
    with pytest.raises(ContractError, match="different split"):
        join(m, [dict(ref[0], split="test" if ref[0]["split"] == "val" else "val")] + ref[1:], "val")


def test_cli_calibrate_and_evaluate(tmp_path):
    m = manifest()
    write_jsonl(tmp_path / "manifest.jsonl", m)
    write_jsonl(tmp_path / "preds.jsonl", predictions(m))
    env = {"PYTHONPATH": str(ROOT / "src"), "PATH": ""}
    run = lambda *args: subprocess.run([sys.executable, str(ROOT / "scripts/m2_evaluate.py"), *args],
                                       capture_output=True, text=True, env=env)
    r = run("calibrate", "--manifest", str(tmp_path / "manifest.jsonl"), "--predictions",
            str(tmp_path / "preds.jsonl"), "--out", str(tmp_path / "protocol.json"))
    assert r.returncode == 0, r.stderr
    r = run("evaluate", "--manifest", str(tmp_path / "manifest.jsonl"), "--predictions", str(tmp_path / "preds.jsonl"),
            "--protocol", str(tmp_path / "protocol.json"), "--split", "val", "--n-boot", "50", "--out", str(tmp_path / "val"))
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "val" / "summary.md").read_text().count("### `") == 4
