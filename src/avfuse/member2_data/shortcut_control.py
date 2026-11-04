"""Control check for a flagged shortcut feature (after the shortcut audit).

Question: does the detector still separate real from fake when the flagged
feature (e.g. trailing silence) carries no information?

Method (feature-matched subset):
  1. Split the videos of one split into quantile bins of the feature (pooled
     over real and fake).
  2. Inside each bin keep the same number of real and fake videos (random,
     seeded), so the feature has the same distribution in both classes.
  3. On that matched subset, compute the detector's AUROC, and also the
     feature's own shortcut AUROC (should drop to about 0.5: proof that the
     matching worked).
  4. Compare with the detector's AUROC on all videos.

Reading it: if the detector's AUROC stays close to its full value, it is not
relying on the shortcut. If it falls sharply, the shortcut explains part of
the score and the report must say so.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from avfuse.member2_data.evaluate import _auroc, cluster_ci
from avfuse.member2_data.shortcuts import shortcut_auroc


def matched_subset(labels, values, n_bins: int = 10, seed: int = 0) -> np.ndarray:
    """Indices of a subset where real and fake have equal counts in every feature bin."""
    y = np.asarray(labels).astype(int)
    v = np.asarray(values, dtype=float)
    ok = np.flatnonzero(np.isfinite(v))
    edges = np.unique(np.quantile(v[ok], np.linspace(0, 1, n_bins + 1)))
    bins = np.clip(np.searchsorted(edges, v[ok], side="right") - 1, 0, max(len(edges) - 2, 0))
    rng = np.random.default_rng(seed)
    keep = []
    for b in np.unique(bins):
        idx = ok[bins == b]
        real, fake = idx[y[idx] == 0], idx[y[idx] == 1]
        k = min(len(real), len(fake))
        if k:
            keep += list(rng.choice(real, k, replace=False)) + list(rng.choice(fake, k, replace=False))
    return np.sort(np.asarray(keep, dtype=int))


def control_report(rows: Sequence[Mapping], feature: Mapping[str, float], clusters: Mapping[str, str],
                   n_bins: int = 10, n_boot: int = 1000, seed: int = 0) -> dict:
    """rows: predictions.join output for one config and split. feature: clip_id -> value."""
    y = np.array([int(r["label"]) for r in rows])
    s = np.array([r["score_fake"] for r in rows], dtype=float)
    v = np.array([feature.get(r["clip_id"], np.nan) for r in rows], dtype=float)
    g = np.array([clusters[r["clip_id"]] for r in rows])
    m = matched_subset(y, v, n_bins, seed)
    full = cluster_ci(_auroc, y, s, g, n_boot, seed=seed)
    matched = cluster_ci(_auroc, y[m], s[m], g[m], n_boot, seed=seed)
    return {
        "n_all": int(len(y)), "n_matched": int(len(m)), "n_bins": n_bins,
        "feature_shortcut_auroc_all": shortcut_auroc(y, v),
        "feature_shortcut_auroc_matched": shortcut_auroc(y[m], v[m]),
        "detector_auroc_all": full, "detector_auroc_matched": matched,
        "drop": full["value"] - matched["value"],
    }
