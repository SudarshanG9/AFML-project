"""Shortcut control check: detector AUROC on a feature-matched subset (Member 2).

Needs real predictions (from Member 1) and video_features.csv from m2_audit_whole.py.

    python scripts/m2_shortcut_control.py --manifest data/manifests/manifest_whole.jsonl \
        --predictions artifacts/predictions/*.jsonl --features artifacts/reports/audit_whole/video_features.csv \
        --feature trailing_silence_ms --config reference --split val --out artifacts/reports/control_val
"""
import argparse
import csv
import sys
from pathlib import Path

from avfuse.member2_data.manifest import read_manifest
from avfuse.member2_data.predictions import ContractError, by_config, join, read_predictions
from avfuse.member2_data.report import ci, save_json
from avfuse.member2_data.seal import SealedTestError, open_test
from avfuse.member2_data.shortcut_control import control_report
from avfuse.member2_data.splits import group_clusters


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--predictions", nargs="+", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--feature", default="trailing_silence_ms")
    p.add_argument("--config", default="reference")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--protocol", default=None, help="needed for --split test")
    p.add_argument("--unseal-test", action="store_true")
    p.add_argument("--purpose", default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    try:
        if a.split == "test":
            open_test(a.protocol, a.purpose, [a.config], a.unseal_test)
        manifest = read_manifest(a.manifest)
        preds = by_config(read_predictions(*a.predictions))
        if a.config not in preds:
            raise ContractError(f"config {a.config!r} not in predictions; have {sorted(preds)}")
        rows = join(manifest, preds[a.config], a.split)
    except (ContractError, SealedTestError) as e:
        sys.exit(f"ERROR: {e}")
    with open(a.features, newline="") as f:
        feat = {r["video_id"]: float(r[a.feature]) if r.get(a.feature) not in (None, "") else float("nan")
                for r in csv.DictReader(f)}
    rep = control_report(rows, feat, group_clusters(manifest), a.bins, a.n_boot)
    rep.update({"feature": a.feature, "config": a.config, "split": a.split})
    out = Path(a.out)
    save_json(rep, out / "shortcut_control.json")
    lines = [f"# Shortcut control: `{a.feature}` ({a.config}, {a.split})", "",
             f"Matched subset: {rep['n_matched']} of {rep['n_all']} videos, equal real/fake counts in each of "
             f"{a.bins} `{a.feature}` bins.", "",
             "| | all videos | matched subset |", "| --- | --- | --- |",
             f"| `{a.feature}` alone (shortcut AUROC) | {rep['feature_shortcut_auroc_all']:.3f} | "
             f"{rep['feature_shortcut_auroc_matched']:.3f} |",
             f"| detector AUROC [95% CI] | {ci(rep['detector_auroc_all'])} | {ci(rep['detector_auroc_matched'])} |",
             "", f"Drop in detector AUROC: {rep['drop']:+.3f}"]
    (out / "shortcut_control.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
