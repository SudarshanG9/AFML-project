"""Leakage / shortcut audit on real clips (Member 2, Day 11; rerun before final sign-off).

    python scripts/m2_audit.py --clips data/clips/clips.csv \
        --manifest data/manifests/manifest.jsonl --out artifacts/reports/audit [--probe-media]

Reads every audio.npy listed in clips.csv. --probe-media also runs ffprobe on the
raw videos (they must still be on disk). Writes shortcut_audit.json and audit.md.
If a flag appears, tell the team the same day; for the expected silence flag the
fix is to cut real clips from the middle of the video as well.
"""
import argparse
from pathlib import Path

from avfuse.member2_data.lavdf import read_clips_csv
from avfuse.member2_data.manifest import read_manifest
from avfuse.member2_data.report import save_json
from avfuse.member2_data.shortcuts import FLAG_AUROC, clip_features, shortcut_audit


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clips", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--probe-media", action="store_true")
    a = p.parse_args()
    manifest = read_manifest(a.manifest)
    wanted = {m["clip_id"] for m in manifest}
    clips = [c for c in read_clips_csv(a.clips) if c["clip_id"] in wanted]
    print(f"computing features for {len(clips)} clips...")
    feats = {c["clip_id"]: clip_features(c, a.probe_media) for c in clips}
    rep = shortcut_audit(manifest, feats)
    out = Path(a.out)
    save_json(rep, out / "shortcut_audit.json")
    lines = ["# Shortcut (leakage) audit", "", f"Rule fixed in advance: a feature is flagged if it alone separates "
             f"real from fake with AUROC ≥ {FLAG_AUROC} (0.5 = useless, 1.0 = perfect shortcut).", ""]
    for split, block in rep["splits"].items():
        lines += [f"## {split}", "", "| Feature | real median | fake median | shortcut AUROC | flag |",
                  "| --- | --- | --- | --- | --- |"]
        for n, s in block.get("all", {}).items():
            fmt = lambda v: "" if v is None else f"{v:.3g}"
            lines.append(f"| {n} | {fmt(s.get('real_median'))} | {fmt(s.get('fake_median'))} | "
                         f"{s['shortcut_auroc']:.3f} | {'**FLAG**' if s['flag'] else ''} |")
        lines.append("")
    lines += ["## Flags", ""] + ([f"- {f}" for f in rep["flags"]] or ["- none"])
    (out / "audit.md").write_text("\n".join(lines) + "\n")
    print(f"{len(rep['flags'])} flag(s)")
    for f in rep["flags"]:
        print("  FLAG", f)
    print(f"wrote {out}/audit.md")


if __name__ == "__main__":
    main()
