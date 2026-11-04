"""Shortcut (leakage) audit on whole videos, the DiMoDif unit (Member 2). CPU only.

    python scripts/m2_audit_whole.py --videos data/manifests/videos_whole.csv \
        --manifest data/manifests/manifest_whole.jsonl --out artifacts/reports/audit_whole --workers 2

For every video: leading/trailing silence and loudness of the whole audio track, duration,
frame count, and (ffprobe) codecs, sample rate, fps, resolution, bitrate and file size.
Each feature is flagged if it alone separates real from fake with AUROC >= 0.65 (rule fixed in
advance), per split, overall and per fake type. Writes shortcut_audit.json and audit.md.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

from avfuse.member2_data.manifest import read_manifest
from avfuse.member2_data.report import save_json
from avfuse.member2_data.shortcuts import audit_markdown, shortcut_audit, whole_video_features
from avfuse.member2_data.wholevideo import read_videos_csv


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--no-probe", action="store_true", help="skip ffprobe (codec/bitrate) features")
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()

    manifest = read_manifest(a.manifest)
    wanted = {m["clip_id"] for m in manifest}
    videos = [v for v in read_videos_csv(a.videos) if v["video_id"] in wanted]
    print(f"computing features for {len(videos)} whole videos...")
    feats = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (v, f) in enumerate(zip(videos, ex.map(partial(whole_video_features, probe=not a.no_probe),
                                                       videos, chunksize=16)), 1):
            feats[v["video_id"]] = f
            if i % 400 == 0:
                print(f"{i}/{len(videos)}")
    errors = sum("audio_error" in f for f in feats.values())
    rep = shortcut_audit(manifest, feats)
    rep["n_videos"] = len(videos)
    rep["audio_errors"] = errors
    out = Path(a.out)
    save_json(rep, out / "shortcut_audit.json")
    (out / "audit.md").write_text(audit_markdown(rep, "Shortcut (leakage) audit: whole videos"))
    print(f"{len(videos)} videos, {errors} audio read error(s)")
    for split, block in rep["splits"].items():
        print(f"--- {split} (all types)")
        for n, s in block.get("all", {}).items():
            print(f"  {n:24s} shortcut AUROC {s['shortcut_auroc']:.3f}{'  FLAG' if s['flag'] else ''}")
    print(f"{len(rep['flags'])} flag(s)")
    for f in rep["flags"]:
        print("  FLAG", f)
    print(f"wrote {out}/audit.md and {out}/shortcut_audit.json")
    print(json.dumps({"passed": rep["passed"], "n_flags": len(rep["flags"])}))


if __name__ == "__main__":
    main()
