import json
import subprocess
import sys
from pathlib import Path

from avfuse.member2_data.lavdf import clip_id, index_lavdf, sample_per_split
from avfuse.member2_data.manifest import read_manifest
from avfuse.member2_data.wholevideo import read_videos_csv, video_id, whole_video_records

ROOT = Path(__file__).resolve().parents[2]


def _meta(tmp_path, n=40):
    entries = []
    for i in range(n):
        split = ["dev", "test"][i % 2]
        vf, af = (i // 2) % 2, (i // 4) % 2
        fake = vf or af
        entries.append({"file": f"{split}/{i:06d}.mp4", "split": split, "modify_video": bool(vf),
                        "modify_audio": bool(af), "duration": 6.0 + i % 5, "video_frames": int((6.0 + i % 5) * 25),
                        "fake_periods": [[0.1, 0.4]] if fake else [],  # edit at the very start
                        "original": f"{split}/{split}{i // 4:06d}_orig.mp4" if fake else None})
    (tmp_path / "metadata.json").write_text(json.dumps(entries))
    return str(tmp_path)


def test_same_videos_as_clip_pipeline_and_no_exclusions(tmp_path):
    root = _meta(tmp_path)
    caps = {"val": 3, "test": 3}
    whole = whole_video_records(root, caps)
    clips = sample_per_split(index_lavdf(root), caps)
    assert [v["path"] for v in whole] == [c["path"] for c in clips]  # identical selection
    assert len(whole) == len(clips)  # an edit at 0.1 s would be cut by the clip window; whole videos keep it


def test_ids_never_collide_with_clip_ids(tmp_path):
    root = _meta(tmp_path)
    recs = index_lavdf(root)
    real = next(r for r in recs if r["manipulation_type"] == "real")
    assert real["start_seconds"] == 0.0
    assert video_id(real["path"]) != clip_id(real)


def test_script_writes_valid_manifest(tmp_path):
    root = _meta(tmp_path)
    out, vids = tmp_path / "m.jsonl", tmp_path / "v.csv"
    r = subprocess.run([sys.executable, str(ROOT / "scripts/m2_whole_video_manifest.py"), "--root", root,
                        "--cap", "val=3", "test=3", "--out", str(out), "--videos-out", str(vids)],
                       capture_output=True, text=True, env={"PYTHONPATH": str(ROOT / "src"), "PATH": ""})
    assert r.returncode == 0, r.stderr
    m = read_manifest(out)
    v = read_videos_csv(str(vids))
    assert len(m) == len(v) > 0 and {x["split"] for x in m} == {"val", "test"}
    assert all("whole video" in x["notes"] and x["feature_path"] == f"data/cache/{x['clip_id']}.pt" for x in m)
    assert v[0]["n_frames"] == int(v[0]["duration_s"] * 25)
    r = subprocess.run([sys.executable, str(ROOT / "scripts/m2_whole_video_manifest.py"), "--root", root,
                        "--cap", "test=3", "--out", str(out), "--videos-out", str(vids), "--check-files"],
                       capture_output=True, text=True, env={"PYTHONPATH": str(ROOT / "src"), "PATH": ""})
    assert r.returncode != 0 and "missing on disk" in r.stderr
