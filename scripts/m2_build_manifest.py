"""clips.csv -> contracts-valid manifest.jsonl, with the split audit (Member 2).

    python scripts/m2_build_manifest.py --clips data/clips/clips.csv \
        --out data/manifests/manifest.jsonl [--resplit]

Also accepts the manifest.csv written by the earlier Colab pipeline (same columns).
Clips the model cannot score are excluded with the loader's rules (too short, or
a LAV-DF edit outside the 2 s window) and the counts are printed: report them.
If any source/identity group crosses splits the manifest is refused unless
--resplit is given (moves whole groups; test is never redrawn).
"""
import argparse
import json
import os
import sys

from avfuse.member2_data.lavdf import long_enough, read_clips_csv, window_covers_fake
from avfuse.member2_data.manifest import manifest_row_from_clip, write_manifest
from avfuse.member2_data.splits import resplit_by_group, split_audit


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clips", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--resplit", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    clips = read_clips_csv(a.clips)
    short = [c for c in clips if not long_enough(c)]
    ok = [c for c in clips if long_enough(c)]
    outside = [c for c in ok if not window_covers_fake(c)]
    ok = [c for c in ok if window_covers_fake(c)]
    print(f"{len(clips)} clips: excluded {len(short)} too short, {len(outside)} fakes with the edit "
          f"outside the evaluation window -> {len(ok)} kept")

    rows = [manifest_row_from_clip(c) for c in ok]
    audit = split_audit(rows)
    if not audit["passed"]:
        if not a.resplit:
            sys.exit(f"{audit['groups_crossing_splits']} source/identity groups cross splits "
                     f"({audit['clips_in_crossing_groups']} clips). Rerun with --resplit.")
        rows = resplit_by_group(rows, seed=a.seed)
        audit = {"before": audit, "after": split_audit(rows)}
        print(f"re-split: moved groups; now passed={audit['after']['passed']}")
    write_manifest(rows, a.out)
    report = os.path.splitext(a.out)[0] + "_split_audit.json"
    with open(report, "w") as f:
        json.dump({"excluded_too_short": len(short), "excluded_edit_outside_window": len(outside),
                   "audit": audit}, f, indent=2)
    final = audit.get("after", audit)
    print(f"split audit passed: {final['passed']}  groups per split: {final['groups_per_split']}")
    for s in ("train", "val", "test"):
        if s in final["counts"]:
            print(f"  {s:5s} {final['counts'][s]}")
    print(f"wrote {a.out} and {report}")


if __name__ == "__main__":
    main()
