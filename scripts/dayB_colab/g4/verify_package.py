"""Verify packaged bytes only; never opens a feature cache."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]

def verify():
    manifest = json.loads((ROOT / "PACKAGE_MANIFEST.json").read_text())
    expected = {entry["path"] for entry in manifest["files"]}
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*")
              if p.is_file() and "__pycache__" not in p.parts
              and p.name != "PACKAGE_MANIFEST.json"}
    if actual != expected:
        raise RuntimeError(f"Package file-set mismatch: missing={expected-actual}; extra={actual-expected}")
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise RuntimeError(f"Package checksum mismatch: {entry['path']}")
    return manifest

if __name__ == "__main__":
    result = verify()
    print(f"PACKAGE INTEGRITY PASS: {len(result['files'])} files")
