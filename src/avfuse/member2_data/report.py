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
