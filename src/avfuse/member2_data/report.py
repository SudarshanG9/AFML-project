"""Markdown tables for Member 2's evaluation outputs."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=float))


def ci(b, fmt="{:.3f}") -> str:
    if not b or b.get("value") is None or math.isnan(b["value"]):
        return "n/a"
    return f"{fmt.format(b['value'])} [{fmt.format(b['ci_low'])}, {fmt.format(b['ci_high'])}]"


def config_markdown(config: str, split: str, m: dict) -> str:
    lines = [f"### `{config}` on {split}", "",
             f"{m['n']} clips ({m['n_real']} real, {m['n_fake']} fake) in {m['n_clusters']} source/identity "
             "groups. Brackets: 95% cluster-bootstrap interval.", "",
             "| Metric | Value [95% CI] |", "| --- | --- |",
             f"| AUROC | {ci(m['auroc'])} |", f"| AUPRC | {ci(m['auprc'])} |", f"| BCE loss | {ci(m['bce'])} |"]
    if "recall" in m:
        lines += [f"| Recall @ frozen threshold {m['threshold']:.4f} | {ci(m['recall'])} |",
                  f"| Real false-positive rate @ threshold | {ci(m['real_fpr'])} |"]
    lines += ["", "| Attack type (vs all real) | n fake | AUROC | AUPRC | Recall |", "| --- | --- | --- | --- | --- |"]
    for ft, b in m["per_type"].items():
        lines.append(f"| {ft} | {b['n_fake']} | {ci(b['auroc'])} | {ci(b['auprc'])} | {ci(b.get('recall'))} |")
    return "\n".join(lines) + "\n"


def paired_markdown(c: dict) -> str:
    lines = [f"### Paired: `{c['b']}` − `{c['a']}` ({c['split']}, {c['n']} clips)", "",
             "| Metric | a | b | b − a [95% paired CI] | CI excludes 0 |", "| --- | --- | --- | --- | --- |"]
    for k, d in c["metrics"].items():
        lines.append(f"| {k} | {d['a']:.4f} | {d['b']:.4f} | {d['delta']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}] | "
                     f"{'yes' if d['excludes_zero'] else 'no'} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- figures (matplotlib optional)

def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def plot_curves(curves: dict, path) -> bool:
    """curves: config -> (labels, scores). ROC and precision-recall side by side."""
    plt = _plt()
    if plt is None:
        return False
    import numpy as np
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for cfg, (y, s) in curves.items():
        o = np.argsort(-np.asarray(s), kind="mergesort")
        y = np.asarray(y)[o]
        tp, fp = np.cumsum(y), np.cumsum(1 - y)
        ax[0].plot(np.r_[0, fp / fp[-1]], np.r_[0, tp / tp[-1]], lw=1.6, label=cfg)
        ax[1].plot(tp / tp[-1], tp / np.arange(1, len(y) + 1), lw=1.6, label=cfg)
    ax[0].plot([0, 1], [0, 1], ls=":", c="grey", lw=1)
    ax[0].set(xlabel="Real false-positive rate", ylabel="Recall", title="ROC")
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall", ylim=(0, 1.02))
    for a in ax:
        a.grid(alpha=0.25)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def plot_recall_by_type(results: dict, path) -> bool:
    plt = _plt()
    cfgs = [c for c in results if "recall" in results[c]]
    if plt is None or not cfgs:
        return False
    import numpy as np
    types = [t for t in ("audio_fake", "video_fake", "both_fake") if t in results[cfgs[0]]["per_type"]]
    w = 0.8 / len(cfgs)
    fig, ax = plt.subplots(figsize=(2 + 1.8 * len(types), 3.8))
    for i, c in enumerate(cfgs):
        b = [results[c]["per_type"][t]["recall"] for t in types]
        v = np.array([x["value"] for x in b])
        err = np.array([[x["value"] - x["ci_low"] for x in b], [x["ci_high"] - x["value"] for x in b]])
        ax.bar(np.arange(len(types)) + i * w, v, w, yerr=err, capsize=3, label=c)
    ax.set_xticks(np.arange(len(types)) + w * (len(cfgs) - 1) / 2, types)
    ax.set(ylabel="Recall at frozen threshold", ylim=(0, 1.05), title="Recall by attack type (95% cluster CI)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True
