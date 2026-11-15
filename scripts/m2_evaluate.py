"""Member 2 forensic evaluation: manifest + prediction JSONL -> metrics. No PyTorch.

1) Freeze the operating threshold on VALIDATION (reference config, 5% real FPR):
    python scripts/m2_evaluate.py calibrate --manifest data/manifests/manifest.jsonl \
        --predictions artifacts/predictions/*.jsonl --out artifacts/reports/protocol.json
   Freezing = this file records the threshold, the manifest hash and the
   checkpoint hash; evaluate refuses to run if any of them changes.
2) Evaluate every config on validation with that frozen threshold:
    python scripts/m2_evaluate.py evaluate --manifest data/manifests/manifest.jsonl \
        --predictions artifacts/predictions/*.jsonl --protocol artifacts/reports/protocol.json \
        --split val --out artifacts/reports/val
3) Once, after the Day 9 freeze: held-out test (logged in test_access.jsonl):
    python scripts/m2_evaluate.py evaluate ... --split test --unseal-test \
        --purpose "final held-out run" --out artifacts/reports/test
"""
import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np

from avfuse.member2_data.paired import compare, compare_by_type
from avfuse.member2_data.evaluate import evaluate_config, threshold_at_fpr
from avfuse.member2_data.manifest import read_manifest
from avfuse.member2_data.predictions import ContractError, by_config, join, read_predictions
from avfuse.member2_data.seal import SealedTestError, accesses, open_test
from avfuse.member2_data.report import config_markdown, paired_by_type_markdown, paired_markdown, plot_curves, plot_recall_by_type, save_json, sha256_file
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
        open_test(a.protocol, a.purpose, by_config(preds), a.unseal_test)
    if proto and proto["checkpoint_sha256"] != preds[0]["checkpoint_sha256"]:
        sys.exit("predictions come from a different checkpoint than the frozen protocol")
    clusters = group_clusters(manifest)
    thr = proto["threshold"] if proto else None
    results, md, curves = {}, [], {}
    for cfg, rows in sorted(by_config(preds).items(), key=lambda kv: PRECISION_MODES.index(kv[0])):
        joined = join(manifest, rows, a.split)
        results[cfg] = evaluate_config(joined, clusters, thr, a.n_boot, a.seed)
        curves[cfg] = ([r["label"] for r in joined], [r["score_fake"] for r in joined])
        md.append(config_markdown(cfg, a.split, results[cfg]))
        m = results[cfg]
        rec = f"  recall {m['recall']['value']:.4f}" if "recall" in m else ""
        print(f"{cfg:22s} AUROC {m['auroc']['value']:.4f} [{m['auroc']['ci_low']:.4f}, {m['auroc']['ci_high']:.4f}]"
              f"  AUPRC {m['auprc']['value']:.4f}{rec}")
    # Paired contrasts on the same clips (Day 10): the damage, and what each restoration recovers.
    have = set(results)
    pairs = [(x, y) for x, y in (("reference", "low_precision"), ("low_precision", "restore_value_output"),
                                 ("low_precision", "restore_query"), ("restore_query", "restore_value_output"))
             if x in have and y in have]
    paired = [compare(manifest, preds, x, y, clusters, a.split, thr, a.n_boot, a.seed) for x, y in pairs]
    for c in paired:
        for k, d in c["metrics"].items():
            print(f"  {c['b']} - {c['a']}  {k:6s} {d['delta']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]"
                  f"{'  *' if d['excludes_zero'] else ''}")
        md.append(paired_markdown(c))
    # The same contrasts inside each analysis subset (one fake type vs all real).
    paired_by_type = []
    for x, y in pairs:
        bt = compare_by_type(manifest, preds, x, y, clusters, a.split, thr, a.n_boot, a.seed)
        paired_by_type.append({"a": x, "b": y, "by_type": bt})
        md.append(paired_by_type_markdown(bt))
        for ft, c in bt.items():
            flags = [k for k, d in c["metrics"].items() if d["excludes_zero"]]
            print(f"  {y} - {x}  [{ft}] bce {c['metrics']['bce']['delta']:+.4f} "
                  f"[{c['metrics']['bce']['ci_low']:+.4f}, {c['metrics']['bce']['ci_high']:+.4f}]"
                  f"{'  CI excludes 0: ' + ', '.join(flags) if flags else ''}")
    out = Path(a.out)
    save_json({"split": a.split, "paired": paired, "paired_by_type": paired_by_type,
               "paired_by_type_method": "analysis subsets (each fake type + all real); whole source groups resampled "
                                        "with replacement inside the subset; same n_boot and seed",
               "threshold": thr,
               "test_accesses_so_far": len(accesses(a.protocol)) if a.protocol else 0, "n_boot": a.n_boot, "seed": a.seed,
               "manifest_sha256": sha256_file(a.manifest), "configs": results}, out / "metrics.json")
    (out / "summary.md").write_text(f"# Forensic evaluation ({a.split})\n\n" + "\n".join(md))
    plot_curves(curves, out / "curves.png")
    plot_recall_by_type(results, out / "recall_by_type.png")
    print(f"wrote {out}/metrics.json, summary.md and figures")


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
    e.add_argument("--unseal-test", action="store_true")
    e.add_argument("--purpose", default=None)
    e.add_argument("--n-boot", type=int, default=1000)
    e.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    try:
        manifest = read_manifest(a.manifest)
        preds = read_predictions(*a.predictions)
        (cmd_calibrate if a.cmd == "calibrate" else cmd_evaluate)(a, manifest, preds)
    except ContractError as err:
        sys.exit(f"CONTRACT ERROR: {err}")
    except SealedTestError as err:
        sys.exit(f"SEALED: {err}")


if __name__ == "__main__":
    main()
