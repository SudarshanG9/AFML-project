"""Bookkeeping for DiMoDif feature extraction (NumPy only; the heavy part is
scripts/m2_dimodif_features.py).

Layout written per video, mirroring DiMoDif's own `features.npz`:
    <out>/<split>/<video_id>/features.npz   video_features (T_v, D_v), audio_features (T_a, D_a)
    <out>/<split>/<video_id>/meta.json      shapes, frame counts, per-stage timings, status
DiMoDif's loader reads exactly these two arrays, truncates both to min(T_v, T_a)
and zero-pads to 600 frames, so T_v and T_a are recorded separately, never assumed equal.
"""
from __future__ import annotations

import json
import os

import numpy as np

STAGES = ("landmarks_ms", "video_load_crop_ms", "audio_load_ms", "video_encode_ms", "audio_encode_ms")


def out_dir(out_root: str, split: str, video_id: str) -> str:
    return os.path.join(out_root, split, video_id)


def is_done(out_root: str, split: str, video_id: str) -> bool:
    d = out_dir(out_root, split, video_id)
    return os.path.exists(os.path.join(d, "features.npz")) and os.path.exists(os.path.join(d, "meta.json"))


def pending(videos: list[dict], out_root: str, split: str | None = None, limit: int | None = None) -> list[dict]:
    """Videos still to do (lets a disconnected Colab session resume where it stopped)."""
    todo = [v for v in videos if (split in (None, "all") or v["split"] == split)
            and not is_done(out_root, v["split"], v["video_id"])]
    return todo[:limit] if limit else todo


def write_meta(out_root: str, split: str, video_id: str, meta: dict) -> None:
    d = out_dir(out_root, split, video_id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def _stats(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return None
    return {"min": float(x.min()), "median": float(np.median(x)), "p95": float(np.percentile(x, 95)),
            "max": float(x.max()), "mean": float(x.mean())}


def summarize(out_root: str) -> dict:
    """Feature-cache metadata report (shapes, token counts, alignment, timings)."""
    metas = []
    for root, _, files in os.walk(out_root):
        if "meta.json" in files:
            with open(os.path.join(root, "meta.json")) as f:
                metas.append(json.load(f))
    ok = [m for m in metas if m.get("status") == "ok"]
    failed = [m for m in metas if m.get("status") != "ok"]
    rep = {
        "n_ok": len(ok), "n_failed": len(failed),
        "failed_examples": [{"video_id": m.get("video_id"), "error": m.get("error", "")[:200]} for m in failed[:10]],
        "per_split_ok": {s: sum(m["split"] == s for m in ok) for s in sorted({m["split"] for m in ok})},
    }
    if ok:
        rep.update({
            "video_dim": sorted({m["D_v"] for m in ok}), "audio_dim": sorted({m["D_a"] for m in ok}),
            "dtype": sorted({m["dtype"] for m in ok}),
            "T_v": _stats([m["T_v"] for m in ok]), "T_a": _stats([m["T_a"] for m in ok]),
            "T_v_minus_T_a": _stats([m["T_v"] - m["T_a"] for m in ok]),
            "T_v_minus_video_frames": _stats([m["T_v"] - m["n_frames"] for m in ok]),
            "longer_than_600": sum(min(m["T_v"], m["T_a"]) > 600 for m in ok),
            "frames_without_landmarks": _stats([m["frames_no_landmarks"] for m in ok]),
            "timings_ms": {s: _stats([m["timings_ms"][s] for m in ok]) for s in STAGES},
            "environment": ok[0].get("environment"),
        })
    return rep
