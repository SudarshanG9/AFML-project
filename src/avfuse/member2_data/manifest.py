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


# ---------------------------------------------------------------- build + I/O (Member 2)

IDENTITY_NOTE = "identity_id = source_id: LAV-DF has no speaker metadata"


def manifest_row_from_clip(clip: Mapping, cache_dir: str = "data/cache") -> dict:
    """Raw-clip index row (lavdf.py / earlier Colab pipeline) -> contracts-valid row.

    feature_path is where the frozen-encoder cache for this clip WILL live
    (data/cache/<clip_id>.pt); it is written once the Lead's checkpoint is locked.
    """
    from avfuse.member2_data.labels import BY_CATEGORY
    mtype = clip.get("manipulation_type") or BY_CATEGORY[clip["category"]]
    row = {
        "clip_id": clip["clip_id"],
        "dataset": clip.get("dataset") or "lavdf",
        "source_id": clip["identity"],
        "identity_id": clip["identity"],
        "split": clip["split"],
        "label": int(clip["label"]),
        "manipulation_type": mtype,
        "feature_path": f"{cache_dir}/{clip['clip_id']}.pt",
        "notes": IDENTITY_NOTE,
    }
    if clip.get("n_samples"):
        row["duration_s"] = round(int(clip["n_samples"]) / 16000.0, 3)
    return row


def read_manifest(path: str | Path) -> list[dict]:
    from avfuse.shared.jsonl import read_jsonl
    rows = read_jsonl(path)
    verify_manifest(rows)
    return rows


def write_manifest(rows: Sequence[Mapping], path: str | Path) -> None:
    from avfuse.shared.jsonl import write_jsonl
    verify_manifest(rows)
    write_jsonl(path, rows)
