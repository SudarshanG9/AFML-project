"""Paired validation harness and restoration evidence (plan Days 7-8).

Paired: two configurations scored on the SAME clips. Every bootstrap round
draws the same groups for both, so clip-to-clip difficulty cancels and only
the effect of the configuration change is left.

Restoration evidence (validation only):
    R_cfg   = L(low_precision) - L(cfg)          L = mean BCE vs manifest labels
    damage  = L(low_precision) - L(reference)
    share   = R_cfg / damage                       part of the damage recovered
    R/ms    = R_cfg / (p50_ms(cfg) - p50_ms(low_precision))  measured by Member 3
"Positive evidence" = R_cfg > 0 and its 95% CI excludes 0. The README warns
that restore_value_output and restore_query don't cost the same, so they are
compared per MEASURED ms, never assumed equal.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from avfuse.member2_data.evaluate import _auprc, _auroc, bce, cluster_resamples, recall_at
from avfuse.member2_data.predictions import ContractError, by_config, join


def paired_delta(metric, y, s_a, s_b, clusters, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> dict:
    """metric(b) - metric(a) with a paired cluster-bootstrap CI."""
    y, a, b = np.asarray(y), np.asarray(s_a, dtype=float), np.asarray(s_b, dtype=float)
    ds = [d for idx in cluster_resamples(clusters, n_boot, seed)
          if not np.isnan(d := metric(y[idx], b[idx]) - metric(y[idx], a[idx]))]
    lo, hi = np.quantile(ds, [alpha / 2, 1 - alpha / 2]) if ds else (np.nan, np.nan)
    va, vb = metric(y, a), metric(y, b)
    return {"a": float(va), "b": float(vb), "delta": float(vb - va), "ci_low": float(lo), "ci_high": float(hi),
            "excludes_zero": bool(ds) and bool(lo > 0 or hi < 0)}


def aligned(manifest, preds, configs, split):
    cfgs = by_config(preds)
    missing = [c for c in configs if c not in cfgs]
    if missing:
        raise ContractError(f"configs {missing} missing from predictions; have {sorted(cfgs)}")
    joined = {c: join(manifest, cfgs[c], split) for c in configs}  # each sorted by clip_id, complete
    base = joined[configs[0]]
    return (np.array([int(r["label"]) for r in base]), [r["clip_id"] for r in base],
            np.array([r["manipulation_type"] for r in base]),
            {c: np.array([r["score_fake"] for r in joined[c]]) for c in configs})


def compare(manifest, preds, a: str, b: str, clusters: Mapping[str, str], split: str,
            threshold: float | None = None, n_boot: int = 1000, seed: int = 0) -> dict:
    y, ids, _, sc = aligned(manifest, preds, [a, b], split)
    g = np.array([clusters[i] for i in ids])
    fns = {"auroc": _auroc, "auprc": _auprc, "bce": bce}
    if threshold is not None:
        fns["recall"] = lambda yy, ss: recall_at(yy, ss, threshold)
    return {"a": a, "b": b, "split": split, "n": int(len(y)),
            "metrics": {k: paired_delta(f, y, sc[a], sc[b], g, n_boot, seed=seed) for k, f in fns.items()}}


def measured_extra_ms(benchmarks: Sequence[Mapping] | None, scope: str = "cached_feature_model") -> dict:
    """p50_ms(config) - p50_ms(low_precision) per config, same scope and device."""
    if not benchmarks:
        return {}
    rows = [b for b in benchmarks if b["scope"] == scope]
    devices = {b["device"] for b in rows}
    if len(devices) > 1:
        raise ContractError(f"benchmarks from several devices for scope {scope}: {sorted(devices)}")
    p50 = {b["config"]: b["p50_ms"] for b in rows}
    if "low_precision" not in p50:
        return {}
    return {c: v - p50["low_precision"] for c, v in p50.items() if c != "low_precision"}


def restoration_table(manifest, preds, clusters: Mapping[str, str], benchmarks=None,
                      scope: str = "cached_feature_model", n_boot: int = 1000, seed: int = 0) -> dict:
    restores = [c for c in ("restore_value_output", "restore_query") if c in by_config(preds)]
    y, ids, t, sc = aligned(manifest, preds, ["low_precision", "reference", *restores], "val")
    g = np.array([clusters[i] for i in ids])
    L_low, L_ref = bce(y, sc["low_precision"]), bce(y, sc["reference"])
    damage = L_low - L_ref
    extra = measured_extra_ms(benchmarks, scope)
    rows = []
    for c in restores:
        d = paired_delta(bce, y, sc[c], sc["low_precision"], g, n_boot, seed=seed)  # L_low - L_c
        per_type = {ft: bce(y[m], sc["low_precision"][m]) - bce(y[m], sc[c][m])
                    for ft in ("audio_fake", "video_fake", "both_fake")
                    if (m := (t == "real") | (t == ft)).any() and (t == ft).any()}
        row = {"config": c, "R": d["delta"], "ci_low": d["ci_low"], "ci_high": d["ci_high"], "L": d["a"],
               "share_of_damage_recovered": d["delta"] / damage if damage > 0 else None,
               "positive_evidence": bool(d["delta"] > 0 and d["ci_low"] > 0), "R_per_type": per_type,
               "extra_p50_ms": extra.get(c)}
        if extra.get(c):
            row["R_per_ms"] = d["delta"] / extra[c] if extra[c] > 0 else None
        rows.append(row)
    return {"split": "val", "n": int(len(y)), "loss": "mean binary cross-entropy",
            "L_low_precision": L_low, "L_reference": L_ref, "quantization_damage": damage,
            "damage_ci": paired_delta(bce, y, sc["reference"], sc["low_precision"], g, n_boot, seed=seed),
            "cost_scope": scope if extra else None, "rows": rows,
            "verdict": ("no measurable degradation: report robustness, do not invent a rescue (Day 9 gate)"
                        if damage <= 0 else
                        "restoration has label-grounded support" if any(r["positive_evidence"] for r in rows)
                        else "no label-grounded rescue: report the negative finding")}
