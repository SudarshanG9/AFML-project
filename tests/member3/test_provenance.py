"""Hashing and provenance tests (Segment D). Tiny synthetic temporary files only."""
import hashlib
import json
import math
import os
from pathlib import Path

import pytest

from avfuse.member3_runtime import provenance as prov

ABC = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
EMPTY = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def _tree(root: Path, files: dict[str, bytes]) -> Path:
    for rel, data in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return root


def _symlink(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported here")


class _ReadSpy:
    """Wrap a file object; record read sizes and optionally run a hook after the first read."""

    def __init__(self, fh, sizes, after_first=None):
        self._fh, self._sizes, self._after = fh, sizes, after_first

    def __enter__(self):
        self._fh.__enter__()
        return self

    def __exit__(self, *exc):
        return self._fh.__exit__(*exc)

    def fileno(self):
        return self._fh.fileno()

    def read(self, n=-1):
        data = self._fh.read(n)
        self._sizes.append(n)
        if self._after and len(self._sizes) == 1:
            self._after()
        return data


# ------------------------------------------------------------------------------ file hashing
def test_known_digests(tmp_path):
    (tmp_path / "abc").write_bytes(b"abc")
    (tmp_path / "empty").write_bytes(b"")
    assert prov.sha256_file(tmp_path / "abc") == ABC
    assert prov.sha256_file(str(tmp_path / "empty")) == EMPTY


def test_large_file_matches_hashlib(tmp_path):
    data = (b"0123456789abcdef" * 1024) * 200 + b"tail"  # ~3.2 MiB
    path = tmp_path / "big.bin"
    path.write_bytes(data)
    assert prov.sha256_file(path) == hashlib.sha256(data).hexdigest()


def test_reads_are_chunked_never_whole_file(tmp_path, monkeypatch):
    path = tmp_path / "f.bin"
    path.write_bytes(b"x" * 100)
    monkeypatch.setattr(prov, "CHUNK_BYTES", 16)
    sizes: list[int] = []
    real_fdopen = os.fdopen

    def spy(fd, *a, **k):
        fh = real_fdopen(fd, *a, **k)
        return _ReadSpy(fh, sizes) if os.fstat(fd).st_ino == path.stat().st_ino else fh

    monkeypatch.setattr(prov.os, "fdopen", spy)
    assert prov.sha256_file(path) == hashlib.sha256(b"x" * 100).hexdigest()
    assert sizes and set(sizes) == {16}
    assert len(sizes) == 8  # 7 full chunks + remainder + terminating empty read


def test_missing_and_directory_inputs(tmp_path):
    with pytest.raises(prov.MissingInputError, match="does not exist"):
        prov.sha256_file(tmp_path / "nope.bin")
    with pytest.raises(FileNotFoundError):  # still a FileNotFoundError for existing callers
        prov.sha256_file(tmp_path / "nope.bin")
    with pytest.raises(IsADirectoryError):
        prov.sha256_file(tmp_path)


@pytest.mark.skipif(os.name != "posix", reason="POSIX FIFO test")
def test_sha256_file_rejects_fifo_before_opening(tmp_path):
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    with pytest.raises(prov.UnsupportedFileTypeError, match="regular files"):
        prov.sha256_file(fifo)


def test_sha256_file_rejects_symlink(tmp_path):
    target = tmp_path / "target.bin"
    target.write_bytes(b"content")
    link = tmp_path / "link.bin"
    _symlink(link, target)
    with pytest.raises(prov.SymlinkNotAllowedError):
        prov.sha256_file(link)


def test_unreadable_file_raises_and_leaks_no_content(tmp_path, monkeypatch):
    path = tmp_path / "secret.bin"
    path.write_bytes(b"TOP-SECRET-CONTENT")
    real_os_open = os.open

    def deny(target, flags, *a, **k):
        if Path(target) == path:
            raise PermissionError(13, "Permission denied")
        return real_os_open(target, flags, *a, **k)

    monkeypatch.setattr(prov.os, "open", deny)
    with pytest.raises(PermissionError) as info:
        prov.sha256_file(path)
    assert "TOP-SECRET" not in str(info.value)


def test_file_mutation_during_hashing_is_detected(tmp_path, monkeypatch):
    path = tmp_path / "grow.bin"
    path.write_bytes(b"a" * 64)
    monkeypatch.setattr(prov, "CHUNK_BYTES", 16)
    real_open = Path.open
    real_fdopen = os.fdopen

    def append():
        with real_open(path, "ab") as fh:
            fh.write(b"MUTATED-CONTENT")

    def spy(fd, *a, **k):
        fh = real_fdopen(fd, *a, **k)
        same_file = os.fstat(fd).st_ino == path.stat().st_ino
        return _ReadSpy(fh, [], after_first=append) if same_file else fh

    monkeypatch.setattr(prov.os, "fdopen", spy)
    with pytest.raises(prov.FileChangedDuringHashError) as info:
        prov.sha256_file(path)
    assert "MUTATED" not in str(info.value)


def test_environment_info_still_available():
    info = prov.environment_info()
    assert {"python", "platform", "torch", "cuda_runtime", "mps_available",
            "cuda_available"} <= set(info)


# ------------------------------------------------------------------------- directory hashing
def _expected_directory_digest(files: dict[str, bytes]) -> str:
    outer = hashlib.sha256(b"avfuse-directory-hash-v1\n")
    for rel in sorted(files, key=lambda r: r.encode()):
        raw = rel.encode()
        outer.update(len(raw).to_bytes(8, "big") + raw + len(files[rel]).to_bytes(8, "big"))
        outer.update(hashlib.sha256(files[rel]).digest())
    return outer.hexdigest()


def test_directory_hash_matches_documented_algorithm(tmp_path):
    files = {"b.txt": b"two", "a/c.txt": b"one", "a/b.txt": b""}
    root = _tree(tmp_path / "d", files)
    detail = prov.sha256_directory_detailed(root)
    assert detail["sha256"] == _expected_directory_digest(files) == prov.sha256_directory(root)
    assert detail["algorithm"] == "sha256-directory-v1"
    assert (detail["file_count"], detail["total_bytes"]) == (3, 6)


def test_directory_hash_independent_of_creation_order_location_and_mtime(tmp_path):
    files = {"x/1.bin": b"11", "x/2.bin": b"22", "y.bin": b"3"}
    one = _tree(tmp_path / "one", files)
    two = tmp_path / "elsewhere" / "deeper" / "two"
    _tree(two, dict(reversed(list(files.items()))))
    os.utime(two / "y.bin", (1_000_000, 1_000_000))
    assert prov.sha256_directory(one) == prov.sha256_directory(two)


def test_directory_hash_sensitive_to_content_and_names(tmp_path):
    base = prov.sha256_directory(_tree(tmp_path / "a", {"f.txt": b"data"}))
    changed = prov.sha256_directory(_tree(tmp_path / "b", {"f.txt": b"datA"}))
    renamed = prov.sha256_directory(_tree(tmp_path / "c", {"g.txt": b"data"}))
    moved = prov.sha256_directory(_tree(tmp_path / "d", {"sub/f.txt": b"data"}))
    assert len({base, changed, renamed, moved}) == 4


def test_empty_directories_do_not_contribute(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    header_only = hashlib.sha256(b"avfuse-directory-hash-v1\n").hexdigest()
    assert prov.sha256_directory(empty) == header_only
    with_empty_sub = _tree(tmp_path / "w", {"f": b"1"})
    (with_empty_sub / "emptysub").mkdir()
    assert prov.sha256_directory(with_empty_sub) == prov.sha256_directory(
        _tree(tmp_path / "w2", {"f": b"1"}))


def test_directory_symlinks_are_rejected(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"outside")
    root = _tree(tmp_path / "d", {"f": b"1"})
    _symlink(root / "link.txt", outside)
    with pytest.raises(prov.SymlinkNotAllowedError):
        prov.sha256_directory(root)


def test_directory_symlink_to_directory_and_root_symlink_rejected(tmp_path):
    real = _tree(tmp_path / "real", {"f": b"1"})
    root = _tree(tmp_path / "d", {"g": b"2"})
    _symlink(root / "dirlink", real)
    with pytest.raises(prov.SymlinkNotAllowedError):
        prov.sha256_directory(root)
    _symlink(tmp_path / "rootlink", real)
    with pytest.raises(prov.SymlinkNotAllowedError):
        prov.sha256_directory(tmp_path / "rootlink")


def test_directory_missing_and_not_a_directory(tmp_path):
    with pytest.raises(prov.MissingInputError):
        prov.sha256_directory(tmp_path / "missing")
    (tmp_path / "f").write_bytes(b"x")
    with pytest.raises(NotADirectoryError):
        prov.sha256_directory(tmp_path / "f")


def test_directory_unreadable_file_is_not_skipped(tmp_path, monkeypatch):
    root = _tree(tmp_path / "d", {"ok": b"1", "locked": b"2"})
    real_os_open = os.open

    def deny(path, flags, *a, **k):
        if Path(path).name == "locked":
            raise PermissionError(13, "denied")
        return real_os_open(path, flags, *a, **k)

    monkeypatch.setattr(prov.os, "open", deny)
    with pytest.raises(PermissionError):
        prov.sha256_directory(root)


def test_directory_change_during_hashing_is_detected(tmp_path, monkeypatch):
    root = _tree(tmp_path / "d", {"a": b"1", "b": b"2"})
    real = prov.sha256_file
    calls = {"n": 0}

    def hashing_then_add(path):
        calls["n"] += 1
        if calls["n"] == 1:
            (root / "late.bin").write_bytes(b"added during hashing")
        return real(path)

    monkeypatch.setattr(prov, "sha256_file", hashing_then_add)
    with pytest.raises(prov.FileChangedDuringHashError):
        prov.sha256_directory(root)


def test_directory_names_with_backslash_or_unicode_collisions(tmp_path):
    root = tmp_path / "d"
    root.mkdir()
    try:
        (root / "a\\b").write_bytes(b"x")
    except OSError:
        pytest.skip("backslash file names unsupported")
    with pytest.raises(prov.ProvenanceError, match="backslash"):
        prov.sha256_directory(root)

    composed, decomposed = "\u00e9.txt", "e\u0301.txt"
    one, two = tmp_path / "nfc", tmp_path / "nfd"
    _tree(one, {composed: b"same"})
    _tree(two, {decomposed: b"same"})
    assert prov.sha256_directory(one) == prov.sha256_directory(two)  # NFC-normalized
    both = _tree(tmp_path / "both", {composed: b"1"})
    (both / decomposed).write_bytes(b"2")
    if len(list(both.iterdir())) == 2:  # filesystem kept them distinct
        with pytest.raises(prov.ProvenanceError, match="normalize"):
            prov.sha256_directory(both)


# -------------------------------------------------------------------------- canonical JSON
def test_canonical_json_known_value_and_order_independence():
    expected = hashlib.sha256(b'{"a":1,"b":[true,null]}').hexdigest()
    assert prov.sha256_json({"b": [True, None], "a": 1}) == expected
    assert prov.canonical_json_bytes({"b": 1, "a": 2}) == b'{"a":2,"b":1}'


def test_canonical_json_independent_of_indentation(tmp_path):
    value = {"z": {"y": [1, 2.5, "s"], "x": None}, "a": "\u00e9"}
    pretty = json.dumps(value, indent=4)
    compact = json.dumps(value, separators=(",", ":"), sort_keys=True)
    assert (prov.sha256_json(json.loads(pretty)) == prov.sha256_json(json.loads(compact))
            == prov.sha256_json(value))


def test_canonical_json_is_utf8_not_ascii_escaped():
    assert prov.canonical_json_bytes({"k": "\u00e9"}) == '{"k":"\u00e9"}'.encode("utf-8")


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_canonical_json_rejects_non_finite(bad):
    with pytest.raises(prov.CanonicalJsonError):
        prov.sha256_json({"x": [bad]})
    with pytest.raises(ValueError):  # also a ValueError for ordinary handlers
        prov.sha256_json(bad)


def test_canonical_json_rejects_non_string_keys_unknown_types_and_cycles():
    with pytest.raises(prov.CanonicalJsonError, match="keys"):
        prov.sha256_json({1: "a"})
    with pytest.raises(prov.CanonicalJsonError, match="Unsupported"):
        prov.sha256_json({"p": Path("x")})
    cyclic: list = []
    cyclic.append(cyclic)
    with pytest.raises(prov.CanonicalJsonError, match="Circular"):
        prov.sha256_json(cyclic)
    with pytest.raises(prov.CanonicalJsonError):
        prov.sha256_json("\ud800")  # lone surrogate cannot be UTF-8 encoded


def test_canonical_json_distinguishes_int_and_float_as_documented():
    assert prov.sha256_json({"v": 1}) != prov.sha256_json({"v": 1.0})


# ---------------------------------------------------------------------------- hash records
def test_checkpoint_file_and_directory_records(tmp_path):
    ckpt = tmp_path / "model.bin"
    ckpt.write_bytes(b"abc")
    rec = prov.hash_checkpoint(ckpt)
    assert rec["kind"] == "checkpoint" and rec["sha256"] == ABC
    assert rec["algorithm"] == "sha256-file" and rec["name"] == "model.bin"
    assert str(tmp_path) not in json.dumps(rec)  # never an absolute path

    shard_dir = _tree(tmp_path / "ckpt_dir", {"a": b"1", "b": b"2"})
    drec = prov.hash_checkpoint(shard_dir)
    assert drec["algorithm"] == "sha256-directory-v1" and drec["file_count"] == 2


def test_hash_requested_but_input_missing_fails_clearly(tmp_path):
    with pytest.raises(prov.MissingInputError, match="no path was provided"):
        prov.hash_checkpoint(None)
    with pytest.raises(prov.MissingInputError, match="does not exist"):
        prov.hash_checkpoint(tmp_path / "gone.pt")
    with pytest.raises(prov.MissingInputError):
        prov.hash_cache_manifest(tmp_path / "gone.csv")
    with pytest.raises(prov.MissingInputError):
        prov.hash_feature_cache(tmp_path / "gone_dir")


def test_manifest_hash_is_never_labelled_as_cache_content(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_bytes(b"abc")
    rec = prov.hash_cache_manifest(manifest)
    assert rec["kind"] == "cache_manifest" and rec["kind"] != "feature_cache_content"
    assert rec["verified_scope"] == "manifest_file_only"
    assert "NOT verified" in rec["note"]


def test_feature_cache_hash_covers_file_contents(tmp_path):
    cache = _tree(tmp_path / "cache", {"clip1.bin": b"1", "clip2.bin": b"2"})
    rec = prov.hash_feature_cache(cache)
    assert rec["kind"] == "feature_cache_content"
    assert rec["verified_scope"] == "directory_contents" and rec["file_count"] == 2
    before = rec["sha256"]
    (cache / "clip2.bin").write_bytes(b"changed")
    assert prov.hash_feature_cache(cache)["sha256"] != before
    with pytest.raises(NotADirectoryError, match="hash_cache_manifest"):
        prov.hash_feature_cache(cache / "clip1.bin")


def test_config_hash_record_and_kind_validation():
    one = prov.hash_config({"a": 1, "b": 2})
    two = prov.hash_config({"b": 2, "a": 1})
    assert one == two and one["kind"] == "config" and one["algorithm"] == "sha256-json-sorted-v1"
    with pytest.raises(ValueError, match="Unknown hash kind"):
        prov.make_hash_record("weights", "sha256-file", "0" * 64, "file_bytes")
    assert len(set(prov.HASH_KINDS)) == 6  # six separate identity kinds
