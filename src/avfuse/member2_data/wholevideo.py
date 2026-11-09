"""Whole-video units for DiMoDif (val + test).

DiMoDif (the team's chosen detector) reads an entire video, up to 600 frames,
not a fixed 2 s window. So for DiMoDif one manifest row = one whole LAV-DF video.

The videos are exactly the ones the clip pipeline sampled (same caps, same seed,
same `sample_per_split`); only the unit changes:
  * no 4 s cut, no "edit outside the window" exclusion: a whole video always
    contains its edit, so every sampled fake is kept;
  * whole-video IDs are `md5("whole:" + path)[:16]`, a separate namespace, so they
    can never collide with the 4 s clip IDs (a real clip starts at 0 s and would
    otherwise get the same ID as its whole video).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os

from avfuse.member2_data.lavdf import FPS, index_lavdf, sample_per_split
from avfuse.member2_data.manifest import IDENTITY_NOTE

MAX_FRAMES = 600  # DiMoDif's maximum sequence length (paper: "maximum sequence length f to 600")
WHOLE_NOTE = IDENTITY_NOTE + "; unit = whole video (DiMoDif input)"
VIDEO_FIELDS = ["video_id", "split", "manipulation_type", "label", "source_id", "duration_s",
                "n_frames", "path"]


def video_id(path: str) -> str:
    return hashlib.md5(f"whole:{path}".encode()).hexdigest()[:16]


def _metadata(root: str) -> dict:
    meta = next((os.path.join(root, n) for n in ("metadata.json", "metadata.min.json")
                 if os.path.exists(os.path.join(root, n))), None)
    if meta is None:
        raise FileNotFoundError(f"no metadata.json or metadata.min.json in {root}")
    with open(meta) as f:
        return {os.path.join(root, e["file"]): e for e in json.load(f)}


def whole_video_records(root: str, caps: dict[str, int], seed: int = 0) -> list[dict]:
    """Same sample as the clip pipeline, one record per whole video."""
    meta = _metadata(root)
    out = []
    for r in sample_per_split(index_lavdf(root), caps, seed):
        e = meta[r["path"]]
        duration = float(e.get("duration") or 0.0)
        n_frames = int(e.get("video_frames") or round(duration * FPS))
        out.append({
            "video_id": video_id(r["path"]), "split": r["split"],
            "manipulation_type": r["manipulation_type"], "label": r["label"],
            "source_id": r["identity"], "duration_s": round(duration, 3),
            "n_frames": n_frames, "path": r["path"],
        })
    return out


def manifest_row(v: dict, cache_dir: str = "data/cache") -> dict:
    return {
        "clip_id": v["video_id"], "dataset": "lavdf",
        "source_id": v["source_id"], "identity_id": v["source_id"],
        "split": v["split"], "label": int(v["label"]), "manipulation_type": v["manipulation_type"],
        "feature_path": f"{cache_dir}/{v['video_id']}.pt", "duration_s": v["duration_s"],
        "notes": WHOLE_NOTE,
    }


def write_videos_csv(rows: list[dict], path: str) -> None:
    """Index of raw video paths for the feature-extraction step (stays in data/, not git)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=VIDEO_FIELDS)
        w.writeheader()
        w.writerows({k: r[k] for k in VIDEO_FIELDS} for r in rows)


def read_videos_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["label"], r["n_frames"], r["duration_s"] = int(r["label"]), int(r["n_frames"]), float(r["duration_s"])
    return rows
