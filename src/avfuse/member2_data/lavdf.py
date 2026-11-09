"""LAV-DF: index, sample, decode to raw clips, and the window rules (Days 3-4).

Pipeline (NumPy + ffmpeg only, no PyTorch):
    metadata.json --index_lavdf--> records --sample_per_split--> chosen
    chosen --extract_record--> data/clips/<clip_id>/{frames.npy,audio.npy} + clips.csv

clips.csv is the raw-clip index (paths to .npy files). It is NOT the shared
manifest: scripts/m2_build_manifest.py turns it into contracts-valid
manifest.jsonl. This is the same format the earlier Colab pipeline wrote, so
clips already preprocessed there can be used directly.

The frozen-encoder feature cache (data/cache/<clip_id>.pt) follows the Lead's
checkpoint preprocessing and is a separate, later step (preprocess.py).
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass
from typing import Optional

import numpy as np

from avfuse.member2_data.labels import BY_FLAGS

FPS = 25
SAMPLE_RATE = 16000
SAMPLES_PER_FRAME = SAMPLE_RATE // FPS  # 640
LAVDF_SPLITS = {"train": "train", "dev": "val", "test": "test"}
CATEGORY_OF_TYPE = {"real": "RealVideo-RealAudio", "audio_fake": "RealVideo-FakeAudio",
                    "video_fake": "FakeVideo-RealAudio", "both_fake": "FakeVideo-FakeAudio"}


# ---------------------------------------------------------------- index + sample

def _periods_seconds(entry: dict) -> list[list[float]]:
    """fake_periods are seconds in the release we checked; if an end time is far
    past the clip's duration they must be frame numbers instead."""
    periods = [list(map(float, p)) for p in entry.get("fake_periods") or []]
    duration = float(entry.get("duration") or 0)
    if periods and duration and max(p[1] for p in periods) > duration + 1.0:
        periods = [[s / FPS, e / FPS] for s, e in periods]
    return periods


def index_lavdf(root: str, clip_seconds: float = 4.0) -> list[dict]:
    """One record per video, official split (dev -> val). Fakes are cut centred on
    their first edited segment; real clips start at 0 s."""
    meta = next((os.path.join(root, n) for n in ("metadata.json", "metadata.min.json")
                 if os.path.exists(os.path.join(root, n))), None)
    if meta is None:
        raise FileNotFoundError(f"no metadata.json or metadata.min.json in {root}")
    with open(meta) as f:
        entries = json.load(f)
    out = []
    for e in entries:
        split = LAVDF_SPLITS.get(e.get("split"))
        if split is None:
            continue
        vf, af = int(bool(e.get("modify_video"))), int(bool(e.get("modify_audio")))
        mtype = BY_FLAGS[(vf, af)]
        start, fake_rel = 0.0, (None, None)
        periods = _periods_seconds(e)
        if mtype != "real" and periods:
            centre = (periods[0][0] + periods[0][1]) / 2.0
            duration = float(e.get("duration") or 0)
            start = max(0.0, centre - clip_seconds / 2.0)
            if duration:
                start = min(start, max(0.0, duration - clip_seconds))
            fake_rel = (round(periods[0][0] - start, 3), round(periods[0][1] - start, 3))
        source = e.get("original") or e["file"]
        out.append({
            "path": os.path.join(root, e["file"]), "dataset": "lavdf", "split": split,
            "category": CATEGORY_OF_TYPE[mtype], "manipulation_type": mtype,
            "identity": "lavdf:" + os.path.splitext(os.path.basename(source))[0],
            "video_fake": vf, "audio_fake": af, "label": int(mtype != "real"),
            "start_seconds": round(start, 3), "fake_start": fake_rel[0], "fake_end": fake_rel[1],
        })
    return out


def sample_per_split(records: list[dict], caps: dict[str, int], seed: int = 0) -> list[dict]:
    """At most caps[split] clips per (split, category). Each split is sampled on
    its own, so the test sample doesn't depend on the train/val caps."""
    out = []
    for split in ("train", "val", "test"):
        if split not in caps:
            continue
        rng = np.random.default_rng(seed)  # fresh per split: same draws as the earlier Colab code
        groups: dict = {}
        for r in records:
            if r["split"] == split:
                groups.setdefault(r["category"], []).append(r)
        for cat in sorted(groups):
            g = groups[cat]
            if len(g) > caps[split]:
                idx = rng.choice(len(g), size=caps[split], replace=False)
                g = [g[i] for i in sorted(idx)]
            out.extend(g)
    return out


def parse_caps(items: list[str]) -> dict[str, int]:
    """["train=1500", "val=300"] -> {"train": 1500, "val": 300}"""
    caps = {}
    for it in items:
        split, _, n = it.partition("=")
        if split not in ("train", "val", "test") or not n.isdigit():
            raise ValueError(f"bad cap {it!r}; use e.g. train=1500 val=300 test=500")
        caps[split] = int(n)
    return caps


# ---------------------------------------------------------------- decode

def clip_id(record: dict) -> str:
    key = f"{record['path']}@{float(record.get('start_seconds', 0.0)):.3f}"
    return hashlib.md5(key.encode()).hexdigest()[:16]


def _ffmpeg(args: list[str]) -> bytes:
    return subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error"] + args,
                          check=True, capture_output=True).stdout


def extract_clip(mp4: str, size: int = 96, scale: int = 112, max_seconds: float = 4.0,
                 start_seconds: float = 0.0):
    """-> frames uint8 (T, size, size, 3) at 25 fps, audio float32 mono 16 kHz.
    The same seek is applied to both commands so the streams stay aligned."""
    seek = ["-ss", f"{start_seconds:.3f}"] if start_seconds else []
    t = ["-t", str(max_seconds)] if max_seconds else []
    raw = _ffmpeg(seek + ["-i", mp4] + t + [
        "-vf", f"fps={FPS},scale={scale}:{scale}:flags=bicubic,crop={size}:{size}",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, size, size, 3)
    raw = _ffmpeg(seek + ["-i", mp4] + t + ["-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"])
    return frames, np.frombuffer(raw, dtype=np.float32).copy()


def extract_record(record: dict, out_dir: str, **kw) -> Optional[dict]:
    """Write frames.npy / audio.npy for one clip; return its clips.csv row."""
    cid = clip_id(record)
    d = os.path.join(out_dir, cid)
    fpath, apath = os.path.join(d, "frames.npy"), os.path.join(d, "audio.npy")
    if not (os.path.exists(fpath) and os.path.exists(apath)):
        try:
            frames, audio = extract_clip(record["path"], start_seconds=float(record["start_seconds"]), **kw)
        except subprocess.CalledProcessError as e:
            print(f"skip {record['path']}: {e.stderr.decode(errors='ignore')[:200]}")
            return None
        if len(frames) == 0 or len(audio) == 0:
            print(f"skip {record['path']}: empty stream")
            return None
        os.makedirs(d, exist_ok=True)
        np.save(fpath, frames)
        np.save(apath, audio.astype(np.float16))
    n_frames = int(np.load(fpath, mmap_mode="r").shape[0])
    n_samples = int(np.load(apath, mmap_mode="r").shape[0])
    row = {k: record[k] for k in ("dataset", "split", "category", "identity", "label",
                                  "video_fake", "audio_fake", "start_seconds")}
    row.update({"clip_id": cid, "fake_start": record["fake_start"], "fake_end": record["fake_end"],
                "source": record["path"], "frames": fpath, "audio": apath,
                "n_frames": n_frames, "n_samples": n_samples})
    return row


CLIPS_FIELDS = ["clip_id", "dataset", "split", "category", "identity", "label", "video_fake",
                "audio_fake", "n_frames", "n_samples", "start_seconds", "fake_start", "fake_end",
                "frames", "audio", "source"]


def write_clips_csv(rows, path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CLIPS_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: "" if r.get(k) is None else r.get(k) for k in CLIPS_FIELDS})


def read_clips_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("label", "video_fake", "audio_fake", "n_frames", "n_samples"):
            r[k] = int(r[k])
        for k in ("start_seconds", "fake_start", "fake_end"):
            v = r.get(k)
            r[k] = float(v) if v not in (None, "") else (0.0 if k == "start_seconds" else None)
    return rows


# ---------------------------------------------------------------- window rules

@dataclass
class WindowSpec:
    clip_frames: int = 50          # 2 s seen by the model
    max_offset_ms: float = 480.0   # margin kept for the optional time-shift diagnostic


def _usable(n_frames: int, n_samples: int) -> int:
    return min(n_frames, n_samples // SAMPLES_PER_FRAME)


def _margin(spec: WindowSpec) -> int:
    return int(math.ceil(spec.max_offset_ms / 1000.0 * FPS))


def long_enough(row: dict, spec: WindowSpec = WindowSpec()) -> bool:
    return _usable(row["n_frames"], row["n_samples"]) >= spec.clip_frames + 2 * _margin(spec)


def eval_window(row: dict, spec: WindowSpec = WindowSpec()) -> tuple[int, int]:
    """Deterministic evaluation window: centred, or centred on the edit for
    LAV-DF fakes. Returns (first video frame, first audio sample)."""
    lo, hi = _margin(spec), _usable(row["n_frames"], row["n_samples"]) - spec.clip_frames - _margin(spec)
    if hi < lo:
        raise ValueError("clip too short; filter with long_enough()")
    if row.get("fake_start") is not None and row.get("fake_end") is not None:
        focus = (row["fake_start"] + row["fake_end"]) / 2.0 * FPS
        vs = int(min(max(round(focus - spec.clip_frames / 2), lo), hi))
    else:
        vs = (lo + hi) // 2
    return vs, vs * SAMPLES_PER_FRAME


def window_covers_fake(row: dict, spec: WindowSpec = WindowSpec()) -> bool:
    """For clips with an edited segment: does the evaluation window contain any of it?"""
    if row.get("fake_start") is None or row.get("fake_end") is None:
        return True
    vs, _ = eval_window(row, spec)
    return row["fake_end"] * FPS > vs and row["fake_start"] * FPS < vs + spec.clip_frames
