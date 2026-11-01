"""Manifest verification before any model experiment."""
from collections import defaultdict
from collections.abc import Sequence, Mapping
from pathlib import Path
import json
from jsonschema import Draft202012Validator

SCHEMA_PATH = Path(__file__).resolve().parents[3] / "contracts" / "schemas" / "manifest.schema.json"


def schema_path() -> Path:
    """Use repo schema during development; package consumers can supply override later."""
    return SCHEMA_PATH


def verify_manifest(rows: Sequence[Mapping], schema_file: str | Path | None = None) -> None:
    path = Path(schema_file) if schema_file else schema_path()
    with path.open(encoding="utf-8") as fh:
        schema = json.load(fh)
    validator = Draft202012Validator(schema)
    if not rows:
        raise ValueError("Manifest is empty")
    seen_clips = set()
    split_by_source = defaultdict(set)
    split_by_identity = defaultdict(set)
    for row in rows:
        validator.validate(dict(row))
        clip = row["clip_id"]
        if clip in seen_clips:
            raise ValueError(f"Duplicate clip_id: {clip}")
        seen_clips.add(clip)
        split_by_source[(row["dataset"], row["source_id"])].add(row["split"])
        split_by_identity[(row["dataset"], row["identity_id"])].add(row["split"])
    for kind, group in (("source", split_by_source), ("identity", split_by_identity)):
        leaked = [key for key, splits in group.items() if len(splits) > 1]
        if leaked:
            raise ValueError(f"{kind}-group leakage across splits: {leaked[:4]}")
