"""Checks to run on Member 1's prediction files before any metric is reported,
plus the audit of overlap between our evaluation videos and the checkpoint's
training data.

check_predictions: contract (schema, duplicates, one checkpoint), the expected
checkpoint hash, complete join to the manifest, and score orientation (a
higher score_fake must mean "more likely fake"; an AUROC well below 0.5
means the score is inverted and must be fixed on Member 1's side, never
silently flipped here).

train_overlap: DiMoDif's LAV-DF checkpoint was trained on the official train
split and selected on dev. A val/test video is contaminated if its source video
(the real original it was made from) also appears in train, as a real video or
as the base of a fake.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np

from avfuse.member2_data.evaluate import _auroc, cluster_ci
from avfuse.member2_data.predictions import ContractError, by_config, join, validate_predictions
from avfuse.member2_data.splits import group_clusters

DIMODIF_LAVDF_SHA256 = "e1e59006d1bb6ad36a3859af5486a6649ed41bba23258b7c41e24f873be7948e"
FAKE_TYPES = ("audio_fake", "video_fake", "both_fake")


def check_predictions(manifest: Sequence[Mapping], preds: Sequence[Mapping],
                      expected_sha: str = DIMODIF_LAVDF_SHA256, n_boot: int = 500, seed: int = 0) -> dict:
    """Raises ContractError on a hard failure; returns a report otherwise."""
    validate_predictions(preds)
    sha = preds[0]["checkpoint_sha256"]
    if expected_sha and sha != expected_sha:
        raise ContractError(f"checkpoint_sha256 {sha[:16]}... is not the agreed checkpoint {expected_sha[:16]}...")
    clusters = group_clusters(manifest)
    rep = {"checkpoint_sha256": sha, "run_ids": sorted({p["run_id"] for p in preds}), "configs": {}}
    for cfg, rows in by_config(preds).items():
        for split in sorted({r["split"] for r in rows}):
            if split == "test":   # the test split stays sealed: count rows only, compute nothing
                rep["configs"].setdefault(cfg, {})["test"] = {"n": sum(r["split"] == "test" for r in rows)}
                continue
            j = join(manifest, rows, split)
            y = np.array([r["label"] for r in j])
            s = np.array([r["score_fake"] for r in j])
            g = [clusters[r["clip_id"]] for r in j]
            auc = cluster_ci(_auroc, y, s, g, n_boot, seed=seed)
            per_type = {}
            for t in FAKE_TYPES:
                m = np.array([r["manipulation_type"] in ("real", t) for r in j])
                per_type[t] = _auroc(y[m], s[m])
            if auc["ci_high"] < 0.5:
                orientation = "INVERTED"
            elif auc["ci_low"] > 0.5:
                orientation = "ok"
            else:
                orientation = "unclear (AUROC CI contains 0.5)"
            rep["configs"].setdefault(cfg, {})[split] = {
                "n": len(j), "n_real": int((y == 0).sum()), "n_fake": int(y.sum()),
                "auroc": auc, "auroc_per_type_vs_real": per_type, "orientation": orientation,
                "score_real": _quantiles(s[y == 0]), "score_fake": _quantiles(s[y == 1]),
                "n_distinct_scores": int(len(np.unique(s))),
            }
    return rep


def _quantiles(x) -> dict:
    q = np.quantile(x, [0.05, 0.5, 0.95]) if len(x) else [np.nan] * 3
    return {"p05": float(q[0]), "median": float(q[1]), "p95": float(q[2])}


def _stem(path: str) -> str:
    return "lavdf:" + os.path.splitext(os.path.basename(path))[0]


def train_overlap(metadata: Sequence[Mapping], manifest: Sequence[Mapping]) -> dict:
    """Which evaluation sources also occur in LAV-DF's official train split."""
    train_sources, train_files = set(), set()
    for e in metadata:
        if e.get("split") == "train":
            train_sources.add(_stem(e.get("original") or e["file"]))
            train_files.add(_stem(e["file"]))
    rep = {"n_train_videos": len(train_files), "n_train_sources": len(train_sources), "splits": {}}
    for split in sorted({m["split"] for m in manifest}):
        rows = [m for m in manifest if m["split"] == split]
        hit = [m for m in rows if m["source_id"] in train_sources or m["source_id"] in train_files]
        rep["splits"][split] = {
            "n_videos": len(rows), "n_sources": len({m["source_id"] for m in rows}),
            "videos_whose_source_is_in_train": len(hit),
            "by_type": dict(Counter(m["manipulation_type"] for m in hit)),
            "examples": sorted({m["source_id"] for m in hit})[:10],
        }
    rep["clean"] = all(v["videos_whose_source_is_in_train"] == 0 for v in rep["splits"].values())
    return rep


def read_metadata(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)
