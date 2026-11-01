"""Basic stratified performance metrics. Paired grouped CIs are a later task."""
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
