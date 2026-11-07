"""Trace every final measurement to exact inputs and environment.

Segment D extends this module with deterministic content identities. Nothing here runs
inside a timed benchmark callable: hash inputs *before* or *after* measuring.

Identities kept deliberately distinct (``HashRecord.kind``):

``config``                 canonical-JSON hash of the resolved configuration
``checkpoint``             file hash (or directory hash) of the model checkpoint
``cache_manifest``         hash of the cache *manifest file only*
``feature_cache_content``  directory hash of the actual feature-cache files
``run_manifest``           hash of ``run_manifest.json`` (see ``artifacts.py``)
``artifact``               hash of one stored artifact file

A manifest digest is never presented as a feature-cache content digest: the record's
``verified_scope`` says exactly what was read.

Algorithms (all SHA-256)
------------------------
``sha256-file``            raw file bytes, read in 1 MiB chunks.
``sha256-directory-v1``    see :func:`sha256_directory_detailed`.
``sha256-json-sorted-v1``  see :func:`canonical_json_bytes`. This is *not* RFC 8785 (JCS).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import stat
import unicodedata
from pathlib import Path
from typing import Any

import torch

CHUNK_BYTES = 1024 * 1024

ALGORITHM_FILE = "sha256-file"
ALGORITHM_DIRECTORY = "sha256-directory-v1"
ALGORITHM_JSON = "sha256-json-sorted-v1"

KIND_CONFIG = "config"
KIND_CHECKPOINT = "checkpoint"
KIND_CACHE_MANIFEST = "cache_manifest"
KIND_CACHE_CONTENT = "feature_cache_content"
KIND_RUN_MANIFEST = "run_manifest"
KIND_ARTIFACT = "artifact"
HASH_KINDS = (
    KIND_CONFIG, KIND_CHECKPOINT, KIND_CACHE_MANIFEST, KIND_CACHE_CONTENT,
    KIND_RUN_MANIFEST, KIND_ARTIFACT,
)

SCOPE_FILE_BYTES = "file_bytes"
SCOPE_DIRECTORY_CONTENTS = "directory_contents"
SCOPE_MANIFEST_ONLY = "manifest_file_only"
SCOPE_CANONICAL_JSON = "canonical_json"

_DIRECTORY_MAGIC = b"avfuse-directory-hash-v1\n"


class ProvenanceError(Exception):
    """Base class for provenance/hashing failures."""


class MissingInputError(ProvenanceError, FileNotFoundError):
    """A hash was requested but the input is absent or unset."""


class SymlinkNotAllowedError(ProvenanceError):
    """A symbolic link was found where only regular files and directories are allowed."""


class UnsupportedFileTypeError(ProvenanceError):
    """A directory contains something that is neither a regular file nor a directory."""


class FileChangedDuringHashError(ProvenanceError):
    """A file or directory changed while it was being hashed; the digest is untrustworthy."""


class CanonicalJsonError(ProvenanceError, ValueError):
    """A value cannot be represented in the sorted-key canonical JSON form."""


def sha256_file(path: str | Path) -> str:
    """SHA-256 of a file, read in chunks (never the whole file in memory).

    Only regular files are accepted; symlinks and special files are rejected before reading.
    Reads use nonblocking/no-follow flags where the platform supports them. Raises
    ``FileNotFoundError`` (missing), ``IsADirectoryError`` (directory), ``PermissionError``
    (unreadable), :class:`UnsupportedFileTypeError` (FIFO/device/socket), and
    :class:`FileChangedDuringHashError` if file identity or content metadata changes.
    File content is never logged or included in error messages.
    """
    p = Path(path)
    try:
        before = p.lstat()
    except FileNotFoundError:
        raise MissingInputError(f"File to hash does not exist: {p.name!r}") from None
    if stat.S_ISLNK(before.st_mode):
        raise SymlinkNotAllowedError(f"Refusing to hash a symbolic link: {p.name!r}")
    if stat.S_ISDIR(before.st_mode):
        raise IsADirectoryError(f"Expected a file but found a directory: {p.name!r}")
    if not stat.S_ISREG(before.st_mode):
        raise UnsupportedFileTypeError(
            f"Only regular files can be hashed: {p.name!r} "
            f"(filesystem mode {stat.S_IFMT(before.st_mode):#o})"
        )

    # O_NONBLOCK prevents a path swapped to a FIFO between lstat/open from hanging this
    # process; O_NOFOLLOW prevents following a symlink where the platform supports it.
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(p, flags)
    h = hashlib.sha256()
    total = 0
    with os.fdopen(fd, "rb") as fh:
        opened = os.fstat(fh.fileno())
        if not stat.S_ISREG(opened.st_mode):
            raise UnsupportedFileTypeError(
                f"Opened object is not a regular file: {p.name!r}"
            )
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise FileChangedDuringHashError(
                f"File identity changed before hashing: {p.name!r}"
            )
        for block in iter(lambda: fh.read(CHUNK_BYTES), b""):
            h.update(block)
            total += len(block)
        after = os.fstat(fh.fileno())
    if (
        total != opened.st_size
        or (opened.st_size, opened.st_mtime_ns, opened.st_dev, opened.st_ino)
        != (after.st_size, after.st_mtime_ns, after.st_dev, after.st_ino)
    ):
        raise FileChangedDuringHashError(
            f"File changed while being hashed: {p.name!r} "
            f"(size {opened.st_size}->{after.st_size}, read {total} bytes)"
        )
    return h.hexdigest()


def environment_info() -> dict[str, str]:
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "cuda_runtime": str(torch.version.cuda),
        "mps_available": str(torch.backends.mps.is_available()),
        "cuda_available": str(torch.cuda.is_available()),
    }


# --------------------------------------------------------------------- directory hashing
def _normalize_name(name: str) -> str:
    if "\\" in name:
        raise ProvenanceError(
            f"File name contains a backslash and cannot be hashed portably: {name!r}"
        )
    return unicodedata.normalize("NFC", name)


def _scan_directory(root: Path) -> list[tuple[str, Path, int]]:
    """Return ``(relative_posix_path, path, size)`` for every file, sorted by UTF-8 bytes."""
    found: dict[str, tuple[Path, int]] = {}

    def walk(directory: Path, prefix: str) -> None:
        with os.scandir(directory) as it:
            entries = sorted(it, key=lambda e: e.name)
        for entry in entries:
            name = _normalize_name(entry.name)
            rel = f"{prefix}/{name}" if prefix else name
            if entry.is_symlink():
                raise SymlinkNotAllowedError(
                    f"Symbolic links are not allowed in hashed directories: {rel!r}"
                )
            if entry.is_dir(follow_symlinks=False):
                walk(Path(entry.path), rel)
            elif entry.is_file(follow_symlinks=False):
                if rel in found:
                    raise ProvenanceError(f"Two files normalize to the same path: {rel!r}")
                found[rel] = (Path(entry.path), entry.stat(follow_symlinks=False).st_size)
            else:
                raise UnsupportedFileTypeError(
                    f"Only regular files and directories can be hashed: {rel!r}"
                )

    walk(root, "")
    return [(rel, found[rel][0], found[rel][1]) for rel in sorted(found, key=lambda r: r.encode())]


def sha256_directory_detailed(path: str | Path) -> dict[str, Any]:
    """Deterministic content hash of a directory tree.

    Algorithm ``sha256-directory-v1``::

        digest = SHA-256( b"avfuse-directory-hash-v1\\n"
                          + for each file, sorted by UTF-8 bytes of its relative path:
                              len(path_bytes) as 8-byte big-endian
                              + path_bytes
                              + file size as 8-byte big-endian
                              + raw 32-byte SHA-256 of the file content )

    * Path = path relative to ``path``, ``/``-separated, Unicode NFC. Two names that
      normalize identically, and names containing a backslash, are rejected.
    * Only regular files contribute. Empty directories do **not** contribute (an empty
      tree hashes to the SHA-256 of the magic header alone).
    * Modification times, permissions, owners and absolute paths never contribute.
    * Symbolic links (including the root) raise :class:`SymlinkNotAllowedError`; other
      special files raise :class:`UnsupportedFileTypeError`. Nothing is skipped silently.
    * The file list is scanned twice (before and after hashing); a difference, or a file
      changing while hashed, raises :class:`FileChangedDuringHashError`.

    Returns ``{"sha256", "algorithm", "file_count", "total_bytes"}``.
    """
    root = Path(path)
    if not root.exists() and not root.is_symlink():
        raise MissingInputError(f"Directory to hash does not exist: {root.name!r}")
    if root.is_symlink():
        raise SymlinkNotAllowedError(f"Hashed directory root is a symbolic link: {root.name!r}")
    if not root.is_dir():
        raise NotADirectoryError(f"Expected a directory: {root.name!r}")

    first = _scan_directory(root)
    outer = hashlib.sha256()
    outer.update(_DIRECTORY_MAGIC)
    total = 0
    for rel, file_path, size in first:
        digest = sha256_file(file_path)
        raw = rel.encode("utf-8")
        outer.update(len(raw).to_bytes(8, "big"))
        outer.update(raw)
        outer.update(size.to_bytes(8, "big"))
        outer.update(bytes.fromhex(digest))
        total += size
    second = _scan_directory(root)
    if [(r, s) for r, _, s in first] != [(r, s) for r, _, s in second]:
        raise FileChangedDuringHashError(
            f"Directory contents changed while being hashed: {root.name!r}"
        )
    return {
        "sha256": outer.hexdigest(),
        "algorithm": ALGORITHM_DIRECTORY,
        "file_count": len(first),
        "total_bytes": total,
    }


def sha256_directory(path: str | Path) -> str:
    """Hex digest of :func:`sha256_directory_detailed`."""
    return sha256_directory_detailed(path)["sha256"]


# ------------------------------------------------------------------- canonical JSON hashing
def _check_json_value(value: Any, seen: set[int], where: str) -> None:
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalJsonError(f"Non-finite number ({value!r}) at {where}")
        return
    if isinstance(value, (list, tuple, dict)):
        if id(value) in seen:
            raise CanonicalJsonError(f"Circular reference at {where}")
        seen.add(id(value))
        if isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(key, str):
                    raise CanonicalJsonError(
                        f"Object keys must be strings, got {type(key).__name__} at {where}"
                    )
                _check_json_value(item, seen, f"{where}.{key}")
        else:
            for index, item in enumerate(value):
                _check_json_value(item, seen, f"{where}[{index}]")
        seen.discard(id(value))
        return
    raise CanonicalJsonError(f"Unsupported type {type(value).__name__} at {where}")


def canonical_json_bytes(value: Any) -> bytes:
    """Sorted-key canonical serialization used for content hashing.

    Object keys sorted, separators ``(",", ":")``, no insignificant whitespace,
    ``ensure_ascii=False`` encoded as UTF-8, NaN/Infinity rejected, string keys only.
    Ordinary JSON indentation and dict insertion order never change the output.

    This is a project-defined method, **not** RFC 8785 (JCS): number formatting follows
    Python's ``repr`` rules, and ``1`` and ``1.0`` serialize differently.
    """
    _check_json_value(value, set(), "$")
    try:
        text = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
        return text.encode("utf-8")
    except (ValueError, UnicodeEncodeError) as exc:
        raise CanonicalJsonError(f"Cannot canonicalize value: {exc}") from exc


def sha256_json(value: Any) -> str:
    """SHA-256 of :func:`canonical_json_bytes` (algorithm ``sha256-json-sorted-v1``)."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


# -------------------------------------------------------------------------- hash records
def make_hash_record(
    kind: str, algorithm: str, sha256: str, verified_scope: str, *, name: str | None = None,
    **details: Any,
) -> dict[str, Any]:
    """Describe exactly what was hashed. Never includes an absolute path."""
    if kind not in HASH_KINDS:
        raise ValueError(f"Unknown hash kind {kind!r}; use one of {HASH_KINDS}")
    record: dict[str, Any] = {
        "kind": kind, "algorithm": algorithm, "sha256": sha256,
        "verified_scope": verified_scope, "name": name,
    }
    record.update(details)
    return record


def _require_path(path: str | Path | None, what: str) -> Path:
    if path is None:
        raise MissingInputError(f"{what} hashing was requested but no path was provided")
    p = Path(path)
    if not p.exists():
        raise MissingInputError(f"{what} path does not exist: {p.name!r}")
    return p


def hash_config(config: Any) -> dict[str, Any]:
    """Hash of the resolved configuration (canonical JSON, so formatting-independent)."""
    return make_hash_record(
        KIND_CONFIG, ALGORITHM_JSON, sha256_json(config), SCOPE_CANONICAL_JSON
    )


def hash_checkpoint(path: str | Path | None) -> dict[str, Any]:
    """Hash a checkpoint file (or a checkpoint directory, by directory content)."""
    p = _require_path(path, "Checkpoint")
    if p.is_dir():
        detail = sha256_directory_detailed(p)
        return make_hash_record(
            KIND_CHECKPOINT, ALGORITHM_DIRECTORY, detail["sha256"], SCOPE_DIRECTORY_CONTENTS,
            name=p.name, file_count=detail["file_count"], total_bytes=detail["total_bytes"],
        )
    return make_hash_record(
        KIND_CHECKPOINT, ALGORITHM_FILE, sha256_file(p), SCOPE_FILE_BYTES, name=p.name,
        size_bytes=p.stat().st_size,
    )


def hash_cache_manifest(path: str | Path | None) -> dict[str, Any]:
    """Hash a cache manifest *file only*. This does not verify any feature file."""
    p = _require_path(path, "Cache manifest")
    if not p.is_file():
        raise IsADirectoryError(f"Cache manifest must be a file: {p.name!r}")
    return make_hash_record(
        KIND_CACHE_MANIFEST, ALGORITHM_FILE, sha256_file(p), SCOPE_MANIFEST_ONLY, name=p.name,
        size_bytes=p.stat().st_size,
        note="Only the manifest file was hashed; feature-cache file contents were NOT verified.",
    )


def hash_feature_cache(path: str | Path | None) -> dict[str, Any]:
    """Hash the actual contents of a feature-cache directory (``sha256-directory-v1``)."""
    p = _require_path(path, "Feature cache")
    if not p.is_dir():
        raise NotADirectoryError(
            f"Feature-cache content hashing needs a directory: {p.name!r}. "
            "To hash only a manifest file use hash_cache_manifest()."
        )
    detail = sha256_directory_detailed(p)
    return make_hash_record(
        KIND_CACHE_CONTENT, ALGORITHM_DIRECTORY, detail["sha256"], SCOPE_DIRECTORY_CONTENTS,
        name=p.name, file_count=detail["file_count"], total_bytes=detail["total_bytes"],
    )
