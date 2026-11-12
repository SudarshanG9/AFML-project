"""Fail-closed local orchestration guards; no model or research decision code.

Call ensure_inside_repo before passing paths to I/O helpers. Human declarations
are validated, not authenticated: callers must retain the actual user's response.
This module never produces an approved record or modifies a finalized artifact.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any

GATE_ACKS = {
    "C0": "ACK DAYC C0: ACCEPT DAYB EVIDENCE AND LOCAL BASELINE",
    "C8A": "ACK DAYC C8A: AUTHORIZE FULL VALIDATION QUERY CONTROL",
    "C8B": "ACK DAYC C8B: ACCEPT DAY8 VALIDATION EVIDENCE",
    "C9": "ACK DAYC C9: FREEZE PRECISION POLICY AND TEST PROTOCOL",
    "C10": "ACK DAYC C10: AUTHORIZE ONE FOUR-MODE HELD-OUT TEST RUN",
    "C10R": "ACK DAYC C10R: ACCEPT MEMBER1 DAY10 HANDOFF",
}
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def ensure_inside_repo(repo_root: Path, path: Path, *, must_exist: bool = False) -> Path:
    root = Path(repo_root).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PermissionError(f"Path escapes repository: {candidate}") from exc
    if must_exist and not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def sha256_file(path: Path) -> str:
    path = Path(path)
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f"Expected regular file, not directory or symlink: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_digest(value: str) -> None:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise ValueError("Expected a lowercase 64-character SHA-256")


def require_sha256(path: Path, expected_hex: str) -> None:
    _require_digest(expected_hex)
    if sha256_file(path) != expected_hex:
        raise ValueError(f"SHA-256 mismatch: {path}")


def require_new_file(path: Path) -> None:
    if os.path.lexists(path):
        raise FileExistsError(path)


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _write_new_bytes(path: Path, data: bytes) -> str:
    path = Path(path)
    require_new_file(path)
    # Parent is caller-created inside the validated repository; never use /tmp.
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                        dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        # Exclusive even if another process creates path after require_new_file.
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        expected = hashlib.sha256(data).hexdigest()
        require_sha256(path, expected)
        return expected
    finally:
        # Only our own temporary sibling is removed, including failed commits.
        temporary.unlink(missing_ok=True)


def write_new_json(path: Path, payload: dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        raise TypeError("JSON artifact must be an object")
    return _write_new_bytes(path, _json_bytes(payload))


def write_new_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise TypeError("JSONL rows must be a list of objects")
    # Serialize all rows first, so invalid data never produces a final file.
    return _write_new_bytes(path, b"".join(_json_bytes(row) for row in rows))


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_strict_json(path: Path) -> Any:
    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")
    with Path(path).open(encoding="utf-8") as stream:
        result = json.load(stream, object_pairs_hook=_unique_object,
                           parse_constant=reject_constant)
    # Also rejects overflow such as 1e999, which parse_constant does not see.
    _json_bytes(result)
    return result


def require_human_approval(approval_path: Path, gate: str, *, context_sha256: str) -> dict:
    if gate not in GATE_ACKS:
        raise ValueError(f"Unknown research gate: {gate}")
    _require_digest(context_sha256)
    record = read_strict_json(approval_path)
    if not isinstance(record, dict):
        raise PermissionError("Human approval must be a JSON object")
    if (record.get("gate") != gate or record.get("ack_text") != GATE_ACKS[gate]
            or record.get("context_sha256") != context_sha256
            or record.get("user_supplied") is not True):
        raise PermissionError("Missing, stale, or mismatched human acknowledgement")
    try:
        approved_at = datetime.fromisoformat(record["approved_at"].replace("Z", "+00:00"))
    except (KeyError, AttributeError, TypeError, ValueError) as exc:
        raise PermissionError("Approval needs an ISO-8601 timestamp with timezone") from exc
    if approved_at.utcoffset() is None or approved_at > datetime.now(timezone.utc):
        raise PermissionError("Approval timestamp is naive or in the future")
    return record


def require_allowed_split(split: str, *, phase: str, test_approved: bool) -> None:
    if phase not in {"baseline", "day8", "day9", "day10-readonly", "day10"}:
        raise PermissionError(f"Unknown phase: {phase}")
    if split == "val":
        return
    if split == "test" and phase == "day10" and test_approved is True:
        return
    raise PermissionError(f"Split {split!r} is forbidden in phase {phase!r}")


def require_frozen_inputs(expected: dict[str, str], paths: dict[str, Path]) -> None:
    if not expected or set(expected) != set(paths):
        raise ValueError("Frozen input names and supplied path names must match exactly")
    for name in sorted(expected):
        require_sha256(paths[name], expected[name])
