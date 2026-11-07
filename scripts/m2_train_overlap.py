"""Audit overlap between our evaluation videos and the checkpoint's training data.

    python scripts/m2_train_overlap.py --metadata data/raw/LAV-DF/metadata.json \
        --manifest data/manifests/manifest_whole.jsonl --out artifacts/reports/train_overlap.json

DiMoDif's LAV-DF checkpoint was trained on LAV-DF's official train split. An
evaluation video counts as overlapping if its source (the real original it was
made from) also appears in train. LAV-DF has no speaker IDs, so speaker-level
overlap cannot be checked; val = LAV-DF dev, which was used to select the checkpoint.
"""
import argparse
import json
import sys
from pathlib import Path

from avfuse.member2_data.manifest import read_manifest
from avfuse.member2_data.readiness import read_metadata, train_overlap


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata", required=True, help="LAV-DF metadata.json or metadata.min.json")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    rep = train_overlap(read_metadata(a.metadata), read_manifest(a.manifest))
    print(f"LAV-DF train: {rep['n_train_videos']} videos from {rep['n_train_sources']} sources")
    for split, r in rep["splits"].items():
        print(f"{split}: {r['n_videos']} videos / {r['n_sources']} sources; "
              f"source also in train: {r['videos_whose_source_is_in_train']} {r['by_type'] or ''}")
    print("CLEAN: no evaluation source appears in train" if rep["clean"] else "OVERLAP FOUND (see examples)")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, indent=2))
        print(f"wrote {a.out}")
    sys.exit(0 if rep["clean"] else 1)


if __name__ == "__main__":
    main()
