"""Check Member 1's prediction files before evaluating them. No PyTorch.

    python scripts/m2_check_predictions.py --manifest data/manifests/manifest_whole.jsonl \
        --predictions reference_val.jsonl --out artifacts/reports/check_reference_val.json

Fails on: schema problems, duplicate rows, a different checkpoint, predicted clips
missing from the manifest or manifest clips with no prediction. Reports score
orientation per config: "INVERTED" means higher scores mean more real, which must be
fixed in Member 1's scoring and never flipped here. Test rows are only counted.
"""
import argparse
import json
import sys
from pathlib import Path

from avfuse.member2_data.manifest import read_manifest
from avfuse.member2_data.predictions import ContractError
from avfuse.member2_data.readiness import DIMODIF_LAVDF_SHA256, check_predictions
from avfuse.shared.jsonl import read_jsonl


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--predictions", nargs="+", required=True)
    p.add_argument("--expected-sha", default=DIMODIF_LAVDF_SHA256, help="'' to skip the checkpoint check")
    p.add_argument("--n-boot", type=int, default=500)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    preds = [r for f in a.predictions for r in read_jsonl(f)]
    try:
        rep = check_predictions(read_manifest(a.manifest), preds, a.expected_sha, a.n_boot)
    except ContractError as e:
        sys.exit(f"FAILED: {e}")
    print(f"contract OK: {len(preds)} rows, checkpoint {rep['checkpoint_sha256'][:16]}..., runs {rep['run_ids']}")
    for cfg, splits in rep["configs"].items():
        for split, r in splits.items():
            if split == "test":
                print(f"{cfg:22s} test: {r['n']} rows (sealed, not scored)")
                continue
            t = r["auroc_per_type_vs_real"]
            print(f"{cfg:22s} {split}: n={r['n']} ({r['n_real']} real / {r['n_fake']} fake)  "
                  f"AUROC {r['auroc']['value']:.4f} [{r['auroc']['ci_low']:.4f}, {r['auroc']['ci_high']:.4f}]  "
                  f"orientation: {r['orientation']}")
            print(f"{'':22s} per type vs real: " + "  ".join(f"{k} {v:.4f}" for k, v in t.items()))
            print(f"{'':22s} median score real {r['score_real']['median']:.4f} / fake {r['score_fake']['median']:.4f}"
                  f"  distinct scores {r['n_distinct_scores']}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, indent=2))
        print(f"wrote {a.out}")
    if any(r.get("orientation") == "INVERTED" for s in rep["configs"].values() for r in s.values()):
        sys.exit("score orientation INVERTED for at least one config: ask Member 1 to fix score_fake")


if __name__ == "__main__":
    main()
