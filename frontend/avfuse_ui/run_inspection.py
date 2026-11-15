"""Bounded optional metadata reads and dated, non-continuous integrity receipts."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .data_access import load_json


def optional_metadata(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        if not path.exists():
            return None, None
        return load_json(path), None
    except (OSError, ValueError, TypeError) as exc:
        return None, f"Could not read {path.name}: {type(exc).__name__}: {exc}"


def manifest_structure_valid(value: dict[str, Any] | None) -> bool:
    from avfuse.member3_runtime.artifacts import ARTIFACT_TYPES
    return bool(value and value.get("artifact_type") == ARTIFACT_TYPES["manifest"]
                and isinstance(value.get("run_id"), str)
                and isinstance(value.get("artifacts"), list)
                and all(isinstance(entry, dict) and isinstance(entry.get("path"), str)
                        for entry in value["artifacts"]))


def run_file_stamp(run_dir: Path) -> list[tuple]:
    """Cheap change detection, not an integrity check; never read large file bodies."""
    result = []
    for path in sorted(run_dir.rglob("*")):
        stat = path.lstat()
        result.append((str(path.relative_to(run_dir)), stat.st_mode, stat.st_size,
                       stat.st_mtime_ns, stat.st_ctime_ns))
    return result


def verify_with_receipt(run_dir: Path) -> dict[str, Any]:
    from avfuse.member3_runtime.artifacts import verify_run_integrity
    before = run_file_stamp(run_dir)
    verdict = verify_run_integrity(run_dir)
    after = run_file_stamp(run_dir)
    return {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "verdict": verdict, "file_stamp": after, "changed_during_check": before != after}


def receipt_is_current(run_dir: Path, receipt: dict[str, Any]) -> bool:
    try:
        return (not receipt.get("changed_during_check", True)
                and receipt.get("file_stamp") == run_file_stamp(run_dir))
    except OSError:
        return False
