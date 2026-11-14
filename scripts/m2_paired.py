"""Paired validation comparisons and restoration evidence (Member 2, Days 7-8).

    python scripts/m2_paired.py --manifest data/manifests/manifest.jsonl \
        --predictions artifacts/predictions/*.jsonl \
        [--protocol artifacts/reports/protocol.json] \
        [--benchmarks artifacts/benchmarks/*.jsonl] --out artifacts/reports/paired_val

Validation only by design: this is where restoration is judged and the result
feeds the Day 9 freeze. --benchmarks are Member 3's measured records; without
them no cost column is shown (never type in estimated timings).
"""
import argparse
import json
import sys
from pathlib import Path

from avfuse.member2_data.manifest import read_manifest
from avfuse.member2_data.paired import compare, compare_by_type, restoration_table
from avfuse.member2_data.predictions import ContractError, by_config, read_predictions
from avfuse.member2_data.report import paired_by_type_markdown, save_json, sha256_file
from avfuse.member2_data.splits import group_clusters
from avfuse.shared.jsonl import read_jsonl


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--predictions", nargs="+", required=True)
    p.add_argument("--protocol", default=None)
    p.add_argument("--benchmarks", nargs="*", default=None)
    p.add_argument("--scope", default="cached_feature_model")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    try:
        manifest = read_manifest(a.manifest)
        preds = read_predictions(*a.predictions)
        thr = json.loads(Path(a.protocol).read_text())["threshold"] if a.protocol else None
        bench = [r for f in (a.benchmarks or []) for r in read_jsonl(f)]
        clusters = group_clusters(manifest)
        have = by_config(preds)
        comps = [compare(manifest, preds, "reference", c, clusters, "val", thr, a.n_boot, a.seed)
                 for c in ("low_precision", "restore_value_output", "restore_query") if c in have]
        by_type = [{"a": "reference", "b": c, "by_type": compare_by_type(manifest, preds, "reference", c, clusters, "val",
                                                                        thr, a.n_boot, a.seed)}
                   for c in ("low_precision", "restore_value_output", "restore_query") if c in have]
        by_type += [{"a": "low_precision", "b": c, "by_type": compare_by_type(manifest, preds, "low_precision", c, clusters,
                                                                            "val", thr, a.n_boot, a.seed)}
                    for c in ("restore_value_output", "restore_query") if c in have and "low_precision" in have]
        if "restore_query" in have and "restore_value_output" in have:
            by_type.append({"a": "restore_query", "b": "restore_value_output",
                            "by_type": compare_by_type(manifest, preds, "restore_query", "restore_value_output",
                                                       clusters, "val", thr, a.n_boot, a.seed)})
        table = restoration_table(manifest, preds, clusters, bench, a.scope, a.n_boot, a.seed)
    except ContractError as e:
        sys.exit(f"CONTRACT ERROR: {e}")

    out = Path(a.out)
    save_json({"manifest_sha256": sha256_file(a.manifest), "comparisons_vs_reference": comps,
               "comparisons_by_type": by_type, "restoration": table}, out / "paired.json")
    lines = ["# Paired validation comparisons", "", "Deltas are b − a on the same clips; 95% paired cluster-bootstrap CI.", "",
             "| a | b | metric | a | b | delta [95% CI] | CI excludes 0 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for c in comps:
        for k, d in c["metrics"].items():
            lines.append(f"| {c['a']} | {c['b']} | {k} | {d['a']:.4f} | {d['b']:.4f} | "
                         f"{d['delta']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}] | {'yes' if d['excludes_zero'] else 'no'} |")
    t = table
    lines += ["", "# Restoration evidence (validation)", "",
              f"L(low_precision) = {t['L_low_precision']:.4f}, L(reference) = {t['L_reference']:.4f}, "
              f"damage = {t['quantization_damage']:+.4f} [{t['damage_ci']['ci_low']:+.4f}, {t['damage_ci']['ci_high']:+.4f}]", "",
              "| config | R = L_low − L_cfg [95% CI] | share of damage | extra p50 ms | R per ms | positive evidence |",
              "| --- | --- | --- | --- | --- | --- |"]
    for r in t["rows"]:
        share = "" if r["share_of_damage_recovered"] is None else f"{r['share_of_damage_recovered']:.0%}"
        ms = "" if r["extra_p50_ms"] is None else f"{r['extra_p50_ms']:.3f}"
        rpm = "" if r.get("R_per_ms") is None else f"{r['R_per_ms']:.4f}"
        lines.append(f"| {r['config']} | {r['R']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | {share} | {ms} | {rpm} | "
                     f"{'yes' if r['positive_evidence'] else 'no'} |")
    lines += ["", f"**Verdict:** {t['verdict']}", "", "# Paired comparisons by fake type (each type vs all real)", ""]
    lines += [paired_by_type_markdown(c["by_type"]) for c in by_type]
    (out / "paired.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
