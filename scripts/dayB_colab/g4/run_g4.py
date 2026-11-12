"""User-executed G4 only. No Drive mounting, inference or test-feature access."""
import argparse
from collections import Counter
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from verify_package import verify

EXPECTED_CHECKPOINT = "e1e59006d1bb6ad36a3859af5486a6649ed41bba23258b7c41e24f873be7948e"
EXPECTED_VAL = "22d1d9e02403cfdbb05f2116da16bc4015f540547a4ac00bf6b7127c119e359c"
EXPECTED_WHOLE = "8339b2476a71e62c6c6fca7d937ba4cc7a704b09d26f08506c20d1f10de8f3dd"
COUNTS = {"real": 300, "audio_fake": 300, "video_fake": 300, "both_fake": 300}

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    package = verify()
    from avfuse.member2_data.manifest import read_manifest
    from avfuse.member2_data.splits import split_audit
    val_path = ROOT / "data/manifests/manifest_whole_val.jsonl"
    whole_path = ROOT / "data/manifests/manifest_whole.jsonl"
    assert digest(val_path) == EXPECTED_VAL
    assert digest(whole_path) == EXPECTED_WHOLE
    val, whole = read_manifest(val_path), read_manifest(whole_path)
    assert len(val) == 1200 and all(row["split"] == "val" for row in val)
    assert Counter(row["manipulation_type"] for row in val) == COUNTS
    assert len(whole) == 3200 and Counter(row["split"] for row in whole) == {"val": 1200, "test": 2000}
    assert val == [row for row in whole if row["split"] == "val"]
    assert split_audit(whole)["passed"]
    assert args.checkpoint.is_file() and digest(args.checkpoint) == EXPECTED_CHECKPOINT
    cache_root = args.cache_root.resolve(strict=True)
    val_root = cache_root / "val"
    if not val_root.is_dir() or val_root.is_symlink():
        raise RuntimeError("Expected a real val/ directory, not a symlink")
    # Validate resolved paths BEFORE the existing preflight can load arrays.
    # This never walks or opens the sealed test/ subtree.
    for row in val:
        for filename in ("features.npz", "meta.json"):
            path = val_root / row["clip_id"] / filename
            if not path.resolve().is_relative_to(val_root):
                raise RuntimeError(f"Validation cache path escapes val/: {row['clip_id']}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    run_id = "G4_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    run = args.output_root / run_id
    run.mkdir(exist_ok=False)
    versions = {}
    for name in ("numpy", "jsonschema", "torch", "setuptools", "wheel"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    metadata = {"gate": "G4", "run_id": run_id, "python": sys.version,
                "platform": platform.platform(), "package_versions": versions,
                "source_snapshot": package["source_snapshot"],
                "package_manifest_sha256": digest(ROOT / "PACKAGE_MANIFEST.json"),
                "cache_root": str(cache_root), "validation_manifest_sha256": EXPECTED_VAL,
                "canonical_manifest_sha256": EXPECTED_WHOLE,
                "checkpoint_sha256": digest(args.checkpoint),
                "test_feature_access": False, "inference_performed": False,
                "device_scope": "CPU NumPy cache-integrity preflight; no GPU/model benchmark"}
    (run / "execution_environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    command = [sys.executable, "-B", str(ROOT / "scripts/day6_preflight.py"),
               "--manifest", str(val_path), "--cache-root", str(cache_root),
               "--checkpoint", str(args.checkpoint.resolve()),
               "--expected-count", "1200", "--expected-per-type", "300",
               "--out-manifest", str(run / "dayB_val_ready.jsonl"),
               "--out-report", str(run / "preflight_report.json")]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                            env=dict(os.environ, PYTHONPATH=str(ROOT / "src"),
                                     PYTHONDONTWRITEBYTECODE="1", GIT_OPTIONAL_LOCKS="0"))
    (run / "command.json").write_text(json.dumps({"argv": command, "exit_status": result.returncode,
              "stdout": result.stdout, "stderr": result.stderr}, indent=2) + "\n")
    (run / "preflight.log").write_text(result.stdout + result.stderr)
    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)
    print("G4 output directory:", run)
    if result.returncode:
        raise SystemExit(result.returncode)
    report = json.loads((run / "preflight_report.json").read_text())
    ready = read_manifest(run / "dayB_val_ready.jsonl")
    assert report["status"] == "PASS" and report["errors_count"] == 0
    assert report["n_expected"] == report["n_manifest"] == report["n_verified"] == len(ready) == 1200
    assert report["counts_by_type"] == COUNTS and report["checkpoint_sha256"] == EXPECTED_CHECKPOINT
    assert report["source_manifest_sha256"] == EXPECTED_VAL
    assert report["validated_manifest_sha256"] == digest(run / "dayB_val_ready.jsonl")
    originals = {row["clip_id"]: row for row in val}
    assert {row["clip_id"] for row in ready} == set(originals)
    for row in ready:
        assert row["split"] == "val" and Path(row["feature_path"]).is_file()
        assert Path(row["feature_path"]).resolve().is_relative_to(val_root)
        assert {k: v for k, v in row.items() if k != "feature_path"} == {
            k: v for k, v in originals[row["clip_id"]].items() if k != "feature_path"}
    summary = {"gate": "G4", "status": "PASS", "n_verified": 1200,
               "counts_by_type": COUNTS, "original_manifest_sha256": EXPECTED_VAL,
               "derived_manifest_sha256": digest(run / "dayB_val_ready.jsonl"),
               "checkpoint_sha256": EXPECTED_CHECKPOINT, "test_feature_access": False,
               "next_action": "STOP; return reports for human G4 review; G5 not authorized"}
    (run / "verification_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    hashes = {p.name: digest(p) for p in run.iterdir() if p.is_file()}
    (run / "output_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print("DAY B FULL VALIDATION PREFLIGHT: PASS (1200/1200)")
    print("STOP AT G4. Return this output directory for review. Do not run inference/G5.")

if __name__ == "__main__":
    main()
