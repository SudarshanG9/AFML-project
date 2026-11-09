"""Whole-video manifest for DiMoDif (Member 2). Metadata only: no decoding.

    python scripts/m2_whole_video_manifest.py --root data/raw/LAV-DF --cap val=300 test=500 \
        --out data/manifests/manifest_whole.jsonl --videos-out data/manifests/videos_whole.csv \
        [--check-files]

Picks the same videos as scripts/m2_lavdf.py (same --cap and --seed), keeps each video whole,
runs the split audit and validates against the shared schema. --check-files also confirms every
raw video is on disk (use it after downloading, before feature extraction).
"""
import argparse
import json
import os
import sys
from collections import Counter

from avfuse.member2_data.lavdf import parse_caps
from avfuse.member2_data.manifest import write_manifest
from avfuse.member2_data.splits import split_audit
from avfuse.member2_data.wholevideo import MAX_FRAMES, manifest_row, whole_video_records, write_videos_csv


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="folder with LAV-DF metadata.json")
    p.add_argument("--cap", nargs="+", default=["val=300", "test=500"])
    p.add_argument("--out", required=True)
    p.add_argument("--videos-out", required=True)
    p.add_argument("--check-files", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    videos = whole_video_records(a.root, parse_caps(a.cap), a.seed)
    too_long = [v for v in videos if v["n_frames"] > MAX_FRAMES]
    missing = [v for v in videos if a.check_files and not os.path.exists(v["path"])]
    rows = [manifest_row(v) for v in videos]
    audit = split_audit(rows)
    if not audit["passed"]:
        sys.exit(f"{audit['groups_crossing_splits']} source groups cross splits; not writing the manifest")
    if missing:
        sys.exit(f"{len(missing)} raw videos missing on disk, e.g. {missing[0]['path']}. Download them first.")
    write_manifest(rows, a.out)
    write_videos_csv(videos, a.videos_out)
    report = os.path.splitext(a.out)[0] + "_report.json"
    durations = sorted(v["duration_s"] for v in videos)
    summary = {
        "n_videos": len(videos), "counts": audit["counts"], "groups_per_split": audit["groups_per_split"],
        "split_audit_passed": audit["passed"], "longer_than_max_frames": len(too_long),
        "max_frames": MAX_FRAMES, "files_checked": a.check_files,
        "duration_s": {"min": durations[0], "median": durations[len(durations) // 2], "max": durations[-1]},
    }
    with open(report, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"{len(videos)} whole videos (no exclusions needed)")
    for s in ("train", "val", "test"):
        if s in audit["counts"]:
            print(f"  {s:5s} {dict(sorted(audit['counts'][s].items()))}")
    print(f"split audit passed: {audit['passed']}  groups per split: {audit['groups_per_split']}")
    print(f"duration s: min {summary['duration_s']['min']}  median {summary['duration_s']['median']}  "
          f"max {summary['duration_s']['max']};  longer than {MAX_FRAMES} frames: {len(too_long)}")
    print("raw files checked: all present" if a.check_files else "raw files not checked (add --check-files)")
    print(f"wrote {a.out}, {a.videos_out}, {report}")
    print(Counter(v["split"] for v in videos))


if __name__ == "__main__":
    main()
