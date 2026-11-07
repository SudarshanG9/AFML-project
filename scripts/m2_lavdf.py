"""LAV-DF raw-clip stage (Member 2, Days 3-4). NumPy + ffmpeg, no PyTorch.

1) list only the videos we need (so the 25.6 GB download keeps ~9,200 files):
    python scripts/m2_lavdf.py select --root data/raw/LAV-DF \
        --cap train=1500 val=300 test=500 --out data/raw/lavdf_wanted.txt
2) decode them into 4 s clips + clips.csv:
    python scripts/m2_lavdf.py clips --root data/raw/LAV-DF \
        --cap train=1500 val=300 test=500 --out data/clips --workers 2
Same --cap/--seed in both steps gives the same selection.
"""
import argparse
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from avfuse.member2_data.lavdf import extract_record, index_lavdf, parse_caps, sample_per_split, write_clips_csv


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["select", "clips"])
    p.add_argument("--root", required=True, help="folder with metadata.json")
    p.add_argument("--cap", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--prefix", default="LAV-DF", help="top folder inside LAV-DF.tar (select)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    chosen = sample_per_split(index_lavdf(a.root), parse_caps(a.cap), seed=a.seed)
    print(f"{len(chosen)} videos selected")
    for (s, c), n in sorted(Counter((r["split"], r["manipulation_type"]) for r in chosen).items()):
        print(f"  {s:5s} {c:11s} {n}")

    if a.cmd == "select":
        with open(a.out, "w") as f:
            for name in ("metadata.json", "metadata.min.json"):
                f.write(f"{a.prefix}/{name}\n")
            for r in chosen:
                f.write(f"{a.prefix}/{os.path.relpath(r['path'], a.root)}\n")
        print(f"file list -> {a.out} (use with: tar -x -T {a.out})")
        return

    clip_dir = os.path.join(a.out, "clips")
    os.makedirs(clip_dir, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, row in enumerate(ex.map(partial(extract_record, out_dir=clip_dir), chosen, chunksize=8)):
            if row:
                rows.append(row)
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{len(chosen)} decoded")
    path = os.path.join(a.out, "clips.csv")
    write_clips_csv(rows, path)
    print(f"wrote {len(rows)} clips -> {path}")


if __name__ == "__main__":
    main()
