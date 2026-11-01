"""Forensic metrics (Member 2).

metrics_for_fake_type: the starter's single-stratum metric (kept unchanged).
Below it: validation threshold at a fixed false-positive rate, cluster
bootstrap confidence intervals, and the full per-configuration block.

Why cluster bootstrap: clips that share a source video or a person are not
independent examples (EXPERIMENT_PROTOCOL.md), so each bootstrap round
resamples whole groups (splits.group_clusters), not single clips. Resampling
clips would treat near-duplicates as independent evidence and give intervals
that are too narrow.
"""
from collections.abc import Sequence, Mapping
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score

FAKE_STRATA = ("audio_fake", "video_fake", "both_fake")


def metrics_for_fake_type(
    predictions: Sequence[Mapping], manifest: Sequence[Mapping],
    fake_type: str, threshold: float,
) -> dict[str, float | int | str]:
    """Compare *real* against one fake type; join by clip_id, no test tuning."""
    if fake_type not in FAKE_STRATA:
        raise ValueError("Expected one of the three fake manipulation types")
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("Threshold must lie within [0, 1]")
    pred_by_clip = {p["clip_id"]: p for p in predictions}
    selected = [m for m in manifest if m["manipulation_type"] in ("real", fake_type) and m["clip_id"] in pred_by_clip]
    labels = np.asarray([int(m["label"]) for m in selected], dtype=int)
    scores = np.asarray([float(pred_by_clip[m["clip_id"]]["score_fake"]) for m in selected], dtype=float)
    if len(set(labels.tolist())) != 2:
        raise ValueError("Need both real and fake examples to estimate AUROC/AUPRC")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Scores must be finite probabilities in [0, 1]")
    hard_preds = (scores >= threshold).astype(int)
    return {
        "fake_type": fake_type,
        "n_real": int((labels == 0).sum()),
        "n_fake": int((labels == 1).sum()),
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "recall": float(recall_score(labels, hard_preds)),
        "real_fpr": float(hard_preds[labels == 0].mean()),
        "threshold": threshold,
    }


# ---------------------------------------------------------------- Member 2 additions

EPS = 1e-7


def _auroc(y, s):
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def _auprc(y, s):
    return float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def bce(y, s) -> float:
    """Mean binary cross-entropy against labels: the ground-truth loss L."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(s, dtype=float), EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def recall_at(y, s, t) -> float:
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    return float((s[y == 1] >= t).mean()) if (y == 1).any() else float("nan")


def fpr_at(y, s, t) -> float:
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    return float((s[y == 0] >= t).mean()) if (y == 0).any() else float("nan")


def threshold_at_fpr(y, s, target_fpr: float = 0.05) -> dict:
    """Lowest threshold flagging at most target_fpr of the REAL clips given.

    With n real clips at most k = floor(target * n) may score >= t. Sorting real
    scores high to low, t sits just above the (k+1)-th, so only the top k (fewer
    if tied) are flagged. Call it on VALIDATION scores only, then freeze t.
    """
    y, s = np.asarray(y).astype(int), np.asarray(s, dtype=float)
    real = np.sort(s[y == 0])[::-1]
    if len(real) == 0:
        raise ValueError("no real clips: cannot fix a false-positive rate")
    k = int(np.floor(target_fpr * len(real) + 1e-12))
    t = float(np.nextafter(real[k], np.inf)) if k < len(real) else float(s.min())
    return {"threshold": t, "target_fpr": target_fpr, "achieved_fpr": fpr_at(y, s, t),
            "recall": recall_at(y, s, t), "n_real": int(len(real)), "n_fake": int(y.sum())}


def cluster_resamples(clusters, n_boot: int, seed: int = 0):
    """Yield index arrays, each made of whole clusters drawn with replacement."""
    c = np.asarray(clusters)
    _, inv = np.unique(c, return_inverse=True)
    order = np.argsort(inv, kind="mergesort")
    bounds = np.r_[0, np.cumsum(np.bincount(inv))]
    members = [order[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])


def cluster_ci(metric, y, s, clusters, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> dict:
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    vals = [v for idx in cluster_resamples(clusters, n_boot, seed)
            if not np.isnan(v := metric(y[idx], s[idx]))]
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2]) if vals else (np.nan, np.nan)
    return {"value": float(metric(y, s)), "ci_low": float(lo), "ci_high": float(hi),
            "n_boot_valid": len(vals), "n_clusters": int(len(set(np.asarray(clusters).tolist())))}


def evaluate_config(rows: Sequence[Mapping], clusters: Mapping[str, str], threshold: float | None,
                    n_boot: int = 1000, seed: int = 0) -> dict:
    """Overall and per-attack-type metrics for one configuration on one split.

    rows: predictions.join output. clusters: clip_id -> group (from the WHOLE
    manifest, so groups match the split audit). Each fake type is compared
    against ALL real clips of the split, as the protocol requires.
    """
    y = np.array([int(r["label"]) for r in rows])
    s = np.array([r["score_fake"] for r in rows], dtype=float)
    g = np.array([clusters[r["clip_id"]] for r in rows])
    t = np.array([r["manipulation_type"] for r in rows])
    out = {"n": int(len(y)), "n_real": int((y == 0).sum()), "n_fake": int(y.sum()),
           "n_clusters": int(len(set(g.tolist()))),
           "auroc": cluster_ci(_auroc, y, s, g, n_boot, seed=seed),
           "auprc": cluster_ci(_auprc, y, s, g, n_boot, seed=seed),
           "bce": cluster_ci(bce, y, s, g, n_boot, seed=seed)}
    if threshold is not None:
        out["threshold"] = threshold
        out["recall"] = cluster_ci(lambda a, b: recall_at(a, b, threshold), y, s, g, n_boot, seed=seed)
        out["real_fpr"] = cluster_ci(lambda a, b: fpr_at(a, b, threshold), y, s, g, n_boot, seed=seed)
    out["per_type"] = {}
    for ft in FAKE_STRATA:
        if not (t == ft).any():
            continue
        m = (t == "real") | (t == ft)
        block = {"n_fake": int((t == ft).sum()), "n_real": int((t == "real").sum()),
                 "auroc": cluster_ci(_auroc, y[m], s[m], g[m], n_boot, seed=seed),
                 "auprc": cluster_ci(_auprc, y[m], s[m], g[m], n_boot, seed=seed)}
        if threshold is not None:
            only = t == ft
            block["recall"] = cluster_ci(lambda a, b: recall_at(a, b, threshold),
                                         y[only], s[only], g[only], n_boot, seed=seed)
        out["per_type"][ft] = block
    return out
