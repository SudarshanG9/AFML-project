"""Member 2 forensic evaluation: manifest + prediction JSONL -> metrics. No PyTorch.

1) Freeze the operating threshold on VALIDATION (reference config, 5% real FPR):
    python scripts/m2_evaluate.py calibrate --manifest data/manifests/manifest.jsonl \
        --predictions artifacts/predictions/*.jsonl --out artifacts/reports/protocol.json
2) Evaluate every config on validation with that frozen threshold:
    python scripts/m2_evaluate.py evaluate --manifest data/manifests/manifest.jsonl \
        --predictions artifacts/predictions/*.jsonl --protocol artifacts/reports/protocol.json \
        --split val --out artifacts/reports/val
"""
import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np

from avfuse.member2_data.evaluate import evaluate_config, threshold_at_fpr
from avfuse.member2_data.manifest import read_manifest
from avfuse.member2_data.predictions import ContractError, by_config, join, read_predictions
from avfuse.member2_data.report import config_markdown, save_json, sha256_file
from avfuse.member2_data.splits import group_clusters
from avfuse.shared.contracts import PRECISION_MODES


def cmd_calibrate(a, manifest, preds):
    cfgs = by_config(preds)
    if a.config not in cfgs:
        raise ContractError(f"config {a.config!r} not in predictions; have {sorted(cfgs)}")
    rows = join(manifest, cfgs[a.config], "val")
    cal = threshold_at_fpr([r["label"] for r in rows], [r["score_fake"] for r in rows], a.target_fpr)
    proto = {"protocol_version": "m2-protocol/0.1", "created": dt.datetime.now().isoformat(timespec="seconds"),
             "calibration_split": "val", "calibration_config": a.config, **cal,
             "manifest_sha256": sha256_file(a.manifest),
             "checkpoint_sha256": preds[0]["checkpoint_sha256"]}
    save_json(proto, a.out)
    print(f"threshold {cal['threshold']:.6f} from {a.config} on val: real FPR {cal['achieved_fpr']:.4f} "
          f"(target {a.target_fpr}), recall {cal['recall']:.4f}  [{cal['n_real']} real / {cal['n_fake']} fake]")
    print(f"frozen in {a.out}")


def load_protocol(a):
    import json
    proto = json.loads(Path(a.protocol).read_text())
    if proto["manifest_sha256"] != sha256_file(a.manifest):
        sys.exit("manifest changed since the threshold was frozen; refusing to evaluate")
    return proto


def cmd_evaluate(a, manifest, preds):
    proto = load_protocol(a) if a.protocol else None
    if a.split == "test":
        sys.exit("test evaluation is added with the Day 9 protocol freeze")
    if proto and proto["checkpoint_sha256"] != preds[0]["checkpoint_sha256"]:
        sys.exit("predictions come from a different checkpoint than the frozen protocol")
    clusters = group_clusters(manifest)
    thr = proto["threshold"] if proto else None
    results, md = {}, []
    for cfg, rows in sorted(by_config(preds).items(), key=lambda kv: PRECISION_MODES.index(kv[0])):
        joined = join(manifest, rows, a.split)
        results[cfg] = evaluate_config(joined, clusters, thr, a.n_boot, a.seed)
        md.append(config_markdown(cfg, a.split, results[cfg]))
        m = results[cfg]
        rec = f"  recall {m['recall']['value']:.4f}" if "recall" in m else ""
        print(f"{cfg:22s} AUROC {m['auroc']['value']:.4f} [{m['auroc']['ci_low']:.4f}, {m['auroc']['ci_high']:.4f}]"
              f"  AUPRC {m['auprc']['value']:.4f}{rec}")
    out = Path(a.out)
    save_json({"split": a.split, "threshold": thr, "n_boot": a.n_boot, "seed": a.seed,
               "manifest_sha256": sha256_file(a.manifest), "configs": results}, out / "metrics.json")
    (out / "summary.md").write_text(f"# Forensic evaluation ({a.split})\n\n" + "\n".join(md))
    print(f"wrote {out}/metrics.json and summary.md")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("calibrate", "evaluate"):
        s = sub.add_parser(name)
        s.add_argument("--manifest", required=True)
        s.add_argument("--predictions", nargs="+", required=True)
        s.add_argument("--out", required=True)
    c = sub.choices["calibrate"]
    c.add_argument("--config", default="reference")
    c.add_argument("--target-fpr", type=float, default=0.05)
    e = sub.choices["evaluate"]
    e.add_argument("--protocol", default=None)
    e.add_argument("--split", choices=["val", "test"], default="val")
    e.add_argument("--n-boot", type=int, default=1000)
    e.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    try:
        manifest = read_manifest(a.manifest)
        preds = read_predictions(*a.predictions)
        (cmd_calibrate if a.cmd == "calibrate" else cmd_evaluate)(a, manifest, preds)
    except ContractError as err:
        sys.exit(f"CONTRACT ERROR: {err}")


if __name__ == "__main__":
    main()
