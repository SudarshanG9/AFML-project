"""Synthetic manifest + predictions for Member 2 tests. Not results.

Each source = 1 real + 1-3 fakes in one split. low_precision hurts audio_fake
most; restore_value_output recovers it, restore_query does not. Every
prediction row is schema-valid (the schema allows no 'notes' field, so the
synthetic checkpoint name marks them instead).
"""
import zlib

import numpy as np

FAKES = ("audio_fake", "video_fake", "both_fake")
SEP = {"reference": {"audio_fake": 3.0, "video_fake": 3.0, "both_fake": 3.5},
       "low_precision": {"audio_fake": 1.0, "video_fake": 2.6, "both_fake": 3.0},
       "restore_value_output": {"audio_fake": 2.7, "video_fake": 2.6, "both_fake": 3.1},
       "restore_query": {"audio_fake": 1.0, "video_fake": 2.6, "both_fake": 3.0}}
SYNTHETIC_SHA = "SYNTHETIC-NOT-A-CHECKPOINT"


def manifest(n_sources=200, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sources):
        split = ("train", "val", "test")[0 if s % 10 < 6 else (1 if s % 10 < 8 else 2)]
        src = f"src{s:04d}"
        types = ["real"] + list(rng.choice(FAKES, size=rng.integers(1, 4), replace=False))
        for j, t in enumerate(types):
            rows.append({"clip_id": f"{src}_{j}", "dataset": "synthetic", "source_id": src, "identity_id": src,
                         "split": split, "label": int(t != "real"), "manipulation_type": str(t),
                         "feature_path": f"data/cache/{src}_{j}.pt"})
    return rows


def predictions(rows, configs=tuple(SEP), seed=0):
    rng = np.random.default_rng(seed)
    base = {r["clip_id"]: rng.standard_normal() for r in rows}
    out = []
    for cfg in configs:
        jit = np.random.default_rng(zlib.crc32(cfg.encode()))
        for r in rows:
            if r["split"] == "train":
                continue
            z = -1.5 + (SEP[cfg][r["manipulation_type"]] if r["label"] else 0) + base[r["clip_id"]] \
                + 0.05 * jit.standard_normal()
            out.append({"run_id": "synthetic", "clip_id": r["clip_id"], "split": r["split"], "config": cfg,
                        "score_fake": float(1 / (1 + np.exp(-z))), "checkpoint_sha256": SYNTHETIC_SHA})
    return out
