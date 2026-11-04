"""Whole-video shortcut audit on real (tiny, generated) mp4 files. Skipped if ffmpeg is missing."""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from avfuse.member2_data.shortcuts import shortcut_audit, whole_video_features
from avfuse.member2_data.wholevideo import write_videos_csv

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                reason="needs ffmpeg/ffprobe")


def _mp4(path, lead_silence_s):
    """2 s, 25 fps test video; audio = tone delayed by lead_silence_s (silence before it)."""
    delay = int(lead_silence_s * 1000)
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "testsrc=size=64x64:rate=25:duration=2",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000:duration=2",
                    "-filter_complex", f"[1:a]adelay={delay}|{delay},atrim=0:2[a]", "-map", "0:v", "-map", "[a]",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)], check=True)


def _setup(tmp_path, silence_for, n=16):
    videos, manifest = [], []
    for i in range(n):
        fake = i % 2
        p = tmp_path / f"v{i}.mp4"
        _mp4(p, silence_for(i, fake))
        vid = f"vid{i:02d}"
        videos.append({"video_id": vid, "split": "test", "manipulation_type": "audio_fake" if fake else "real",
                       "label": fake, "source_id": f"lavdf:{i}", "duration_s": 2.0, "n_frames": 50, "path": str(p)})
        manifest.append({"clip_id": vid, "split": "test", "label": fake,
                         "manipulation_type": "audio_fake" if fake else "real"})
    return videos, manifest


def test_features_from_real_file(tmp_path):
    videos, _ = _setup(tmp_path, lambda i, f: 0.5, n=1)
    f = whole_video_features(videos[0])
    assert 450 <= f["leading_silence_ms"] <= 560          # ~0.5 s of silence before the tone
    assert abs(f["audio_duration_s"] - 2.0) < 0.1
    assert f["video_codec"] == "h264" and f["audio_codec"] == "aac" and f["video_resolution"] == "64x64"
    assert f["container_bitrate_kbps"] > 0 and f["file_size_mb"] > 0


def test_flags_silence_shortcut_on_whole_videos(tmp_path):
    videos, manifest = _setup(tmp_path, lambda i, fake: 0.6 if fake else 0.0)
    rep = shortcut_audit(manifest, {v["video_id"]: whole_video_features(v) for v in videos})
    assert any("/all: leading_silence_ms" in f for f in rep["flags"])


def test_script_runs_end_to_end(tmp_path):
    from avfuse.member2_data.manifest import IDENTITY_NOTE  # noqa: F401  (manifest helpers importable)
    from avfuse.shared.jsonl import write_jsonl
    videos, _ = _setup(tmp_path, lambda i, fake: 0.0, n=8)
    write_videos_csv(videos, str(tmp_path / "videos.csv"))
    write_jsonl(tmp_path / "m.jsonl", [{"clip_id": v["video_id"], "dataset": "lavdf", "source_id": v["source_id"],
                                        "identity_id": v["source_id"], "split": "test", "label": v["label"],
                                        "manipulation_type": v["manipulation_type"],
                                        "feature_path": f"data/cache/{v['video_id']}.pt"} for v in videos])
    r = subprocess.run([sys.executable, str(ROOT / "scripts/m2_audit_whole.py"), "--videos", str(tmp_path / "videos.csv"),
                        "--manifest", str(tmp_path / "m.jsonl"), "--out", str(tmp_path / "audit"), "--workers", "1"],
                       capture_output=True, text=True, env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/usr/local/bin:/bin"})
    assert r.returncode == 0, r.stderr
    assert "8 videos, 0 audio read error(s)" in r.stdout
    assert (tmp_path / "audit" / "audit.md").read_text().startswith("# Shortcut (leakage) audit: whole videos")
