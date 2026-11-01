"""Read, validate and join prediction JSONL (Lead -> Member 2).

Rows must match contracts/schemas/prediction.schema.json exactly. That schema
has additionalProperties: false, so a row carrying a label (or any field not in
the contract) is rejected: ground truth reaches the evaluator only through the
manifest.
"""
from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from jsonschema import Draft202012Validator

from avfuse.shared.jsonl import read_jsonl

SCHEMA = Path(__file__).resolve().parents[3] / "contracts" / "schemas" / "prediction.schema.json"


class ContractError(ValueError):
    pass


def validate_predictions(rows: Sequence[Mapping]) -> None:
    with SCHEMA.open(encoding="utf-8") as f:
        validator = Draft202012Validator(json.load(f))
    problems, seen = [], set()
    for i, r in enumerate(rows):
        for err in validator.iter_errors(dict(r)):
            problems.append(f"row {i} (clip {r.get('clip_id')!r}): {err.message}")
        key = (r.get("run_id"), r.get("config"), r.get("clip_id"))
        if key in seen:
            problems.append(f"row {i}: clip {r.get('clip_id')!r} predicted twice for config {r.get('config')!r}")
        seen.add(key)
    shas = {r.get("checkpoint_sha256") for r in rows}
    if len(shas) > 1:
        problems.append(f"more than one checkpoint_sha256 in one evaluation: {sorted(map(str, shas))[:3]}")
    if problems:
        raise ContractError(f"{len(problems)} prediction problem(s):\n  " + "\n  ".join(problems[:20]))


def read_predictions(*paths: str | Path) -> list[dict]:
    rows = [r for p in paths for r in read_jsonl(p)]
    validate_predictions(rows)
    return rows


def by_config(rows: Sequence[Mapping]) -> dict[str, list[dict]]:
    out = defaultdict(list)
    for r in rows:
        out[r["config"]].append(dict(r))
    return dict(out)


def join(manifest: Sequence[Mapping], preds: Sequence[Mapping], split: str) -> list[dict]:
    """Manifest rows of `split` + score_fake, sorted by clip_id.

    Errors (never silently dropped, because dropping clips can flatter metrics):
    a predicted clip_id missing from the manifest, a split disagreeing with the
    manifest, or a manifest clip of this split without a prediction.
    """
    m_by_id = {m["clip_id"]: m for m in manifest}
    unknown = [p["clip_id"] for p in preds if p["clip_id"] not in m_by_id]
    if unknown:
        raise ContractError(f"{len(unknown)} predicted clip_id(s) not in the manifest, e.g. {unknown[:3]}")
    wrong = [p["clip_id"] for p in preds if m_by_id[p["clip_id"]]["split"] != p["split"]]
    if wrong:
        raise ContractError(f"{len(wrong)} prediction(s) label a clip with a different split than the manifest")
    got = {p["clip_id"]: p["score_fake"] for p in preds if p["split"] == split}
    expected = [m for m in manifest if m["split"] == split]
    missing = [m["clip_id"] for m in expected if m["clip_id"] not in got]
    if missing:
        raise ContractError(f"{len(missing)} of {len(expected)} {split} clips have no prediction, e.g. {missing[:3]}")
    return sorted((dict(m, score_fake=float(got[m["clip_id"]])) for m in expected), key=lambda r: r["clip_id"])
