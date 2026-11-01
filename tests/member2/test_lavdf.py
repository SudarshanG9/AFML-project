import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from avfuse.member2_data.lavdf import (index_lavdf, long_enough, parse_caps, sample_per_split,
                                       window_covers_fake, write_clips_csv)
from avfuse.member2_data.manifest import read_manifest

ROOT = Path(__file__).resolve().parents[2]


def _meta(tmp_path):
    entries = [
        {"file": "test/000001.mp4", "split": "test", "modify_video": False, "modify_audio": False,
         "fake_periods": [], "duration": 6.0, "original": None},
        {"file": "test/000002.mp4", "split": "test", "modify_video": False, "modify_audio": True,
         "fake_periods": [[3.0, 3.4]], "duration": 6.0, "original": "test/000001.mp4"},
        {"file": "dev/000003.mp4", "split": "dev", "modify_video": True, "modify_audio": True,
         "fake_periods": [[75, 85]], "duration": 6.0, "original": "dev/000009.mp4"},  # frames, not s
    ]
    (tmp_path / "metadata.json").write_text(json.dumps(entries))
    return str(tmp_path)


def test_index_maps_types_sources_and_cuts(tmp_path):
    recs = {r["path"].split("/")[-1]: r for r in index_lavdf(_meta(tmp_path))}
    real, fake, both = recs["000001.mp4"], recs["000002.mp4"], recs["000003.mp4"]
    assert real["manipulation_type"] == "real" and real["start_seconds"] == 0.0
    assert fake["manipulation_type"] == "audio_fake" and fake["identity"] == real["identity"]
    assert fake["start_seconds"] == 1.2 and (fake["fake_start"], fake["fake_end"]) == (1.8, 2.2)
    assert both["split"] == "val" and both["manipulation_type"] == "both_fake"
    assert both["fake_start"] == 1.8  # 75-85 frames = 3.0-3.4 s; 4 s cut starts at 1.2 s


def test_sampling_caps_per_split_category():
    recs = [{"split": s, "category": c, "x": i} for i in range(20) for s in ("train", "test") for c in "AB"]
    out = sample_per_split(recs, parse_caps(["train=3", "test=5"]))
    assert len(out) == 3 * 2 + 5 * 2
    test_only = sample_per_split(recs, parse_caps(["test=5"]))
    assert [r for r in out if r["split"] == "test"] == test_only  # independent of the train cap


def test_window_rules():
    row = {"n_frames": 100, "n_samples": 64000, "fake_start": 2.0, "fake_end": 2.4}
    assert long_enough(row) and window_covers_fake(row)
    assert not window_covers_fake(dict(row, fake_start=0.05, fake_end=0.2))
    assert not long_enough(dict(row, n_frames=40, n_samples=40 * 640))


def test_build_manifest_script(tmp_path):
    rows = []
    for i in range(8):
        mtype = ["real", "audio_fake"][i % 2]
        rows.append({"clip_id": f"c{i}", "dataset": "lavdf", "split": "test" if i < 4 else "val",
                     "category": ["RealVideo-RealAudio", "RealVideo-FakeAudio"][i % 2],
                     "identity": f"lavdf:{i // 2}", "label": i % 2, "video_fake": 0, "audio_fake": i % 2,
                     "n_frames": 100, "n_samples": 64000, "start_seconds": 0.0,
                     "fake_start": 2.0 if i % 2 else None, "fake_end": 2.4 if i % 2 else None,
                     "frames": "x", "audio": "y", "source": f"/raw/{i}.mp4"})
    rows.append(dict(rows[1], clip_id="short", n_frames=40, n_samples=25600))
    write_clips_csv(rows, str(tmp_path / "clips.csv"))
    out = tmp_path / "manifest.jsonl"
    r = subprocess.run([sys.executable, str(ROOT / "scripts/m2_build_manifest.py"), "--clips",
                        str(tmp_path / "clips.csv"), "--out", str(out)], capture_output=True, text=True,
                       env={"PYTHONPATH": str(ROOT / "src"), "PATH": ""})
    assert r.returncode == 0, r.stderr
    assert "excluded 1 too short" in r.stdout
    m = read_manifest(out)
    assert len(m) == 8 and m[1]["manipulation_type"] == "audio_fake" and m[1]["feature_path"] == "data/cache/c1.pt"
