
"""Day 6: verify Member 2's whole-video validation handoff.

Consumes ONLY a validation-only manifest. Never reads held-out
test data, performs inference, or modifies Member 2's source manifest.

Expected layout:
    <cache_root>/val/<clip_id>/features.npz
    <cache_root>/val/<clip_id>/meta.json
"""

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from avfuse.member2_data.manifest import (
    read_manifest,
    write_manifest,
)
from avfuse.member2_data.labels import label_of
from avfuse.member2_data.splits import group_clusters


EXPECTED_CHECKPOINT_SHA = (
    "e1e59006d1bb6ad36a3859af5486a6649ed41bba23258b7c41e24f873be7948e"
)

TYPES = (
    "real",
    "audio_fake",
    "video_fake",
    "both_fake",
)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)
    return h.hexdigest()


def git_value(*args):
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def inspect_npz(path, metadata_path, clip_id):
    """Validate one frozen encoder cache and its extractor metadata."""

    if not path.is_file():
        raise FileNotFoundError(f"Missing NPZ: {path}")

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing extraction metadata: {metadata_path}"
        )

    meta = json.loads(metadata_path.read_text())

    if meta.get("status") != "ok":
        raise ValueError("Extraction status is not ok")

    if meta.get("video_id") != clip_id:
        raise ValueError("Cache video_id does not match manifest")

    if meta.get("split") != "val":
        raise ValueError("Cache belongs to a different split")

    with np.load(path, allow_pickle=False) as data:
        required = {"video_features", "audio_features"}

        if not required.issubset(set(data.files)):
            raise ValueError("NPZ is missing audio/video features")

        video = data["video_features"]
        audio = data["audio_features"]

        for name, array in (
            ("video", video),
            ("audio", audio),
        ):
            if array.ndim != 2:
                raise ValueError(f"{name}: expected 2D array")

            if array.shape[1] != 768:
                raise ValueError(f"{name}: expected 768 features")

            if array.shape[0] == 0:
                raise ValueError(f"{name}: empty sequence")

            if array.dtype != np.float32:
                raise ValueError(f"{name}: expected float32")

            if not np.isfinite(array).all():
                raise ValueError(f"{name}: non-finite values")

        tv, ta = video.shape[0], audio.shape[0]

    expected_metadata = {
        "T_v": tv,
        "T_a": ta,
        "D_v": 768,
        "D_a": 768,
        "dtype": "float32",
    }

    for field, expected in expected_metadata.items():
        if meta.get(field) != expected:
            raise ValueError(
                f"Metadata mismatch for {field}: "
                f"expected {expected}, found {meta.get(field)}"
            )

    digest = sha256_file(path)

    if (
        meta.get("feature_sha256") is not None
        and meta["feature_sha256"] != digest
    ):
        raise ValueError("NPZ checksum differs from metadata")

    return {
        "clip_id": clip_id,
        "feature_path": str(path.resolve()),
        "feature_sha256": digest,
        "metadata_sha256": sha256_file(metadata_path),
        "video_tokens": tv,
        "audio_tokens": ta,
        "valid_tokens": min(tv, ta, 600),
        "truncated": min(tv, ta) > 600,
    }


def run_preflight(
    manifest_path,
    cache_root,
    checkpoint_path,
    out_manifest,
    out_report,
    *,
    expected_count=1200,
    expected_per_type=300,
    expected_checkpoint_sha=EXPECTED_CHECKPOINT_SHA,
):
    manifest_path = Path(manifest_path)
    cache_root = Path(cache_root).resolve()
    checkpoint_path = Path(checkpoint_path)
    out_manifest = Path(out_manifest)
    out_report = Path(out_report)

    if not cache_root.is_dir():
        raise FileNotFoundError(
            f"Cache root does not exist: {cache_root}"
        )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint missing: {checkpoint_path}"
        )

    if out_manifest.exists():
        raise FileExistsError(
            f"Refusing to overwrite frozen manifest: {out_manifest}"
        )

    checkpoint_sha = sha256_file(checkpoint_path)

    if checkpoint_sha != expected_checkpoint_sha:
        raise RuntimeError(
            "Checkpoint SHA256 does not match the approved "
            "pretrained LAV-DF detector"
        )

    rows = read_manifest(manifest_path)

    # Prevent Member 1's preflight from opening test caches.
    if any(row["split"] != "val" for row in rows):
        raise RuntimeError(
            "Supply a validation-only manifest, not a "
            "combined validation/test manifest"
        )

    counts = Counter(
        row["manipulation_type"] for row in rows
    )

    errors = []

    if len(rows) != expected_count:
        errors.append(
            f"Expected {expected_count} validation rows, "
            f"found {len(rows)}"
        )

    for category in TYPES:
        if counts[category] != expected_per_type:
            errors.append(
                f"{category}: expected {expected_per_type}, "
                f"found {counts[category]}"
            )

    validated_rows = []
    inventory = []
    seen_paths = set()
    seen_hashes = set()

    for row in sorted(rows, key=lambda r: r["clip_id"]):
        clip_id = row["clip_id"]

        try:
            # Whole-video IDs generated by Member 2 are
            # 16-character hexadecimal digests.
            if not re.fullmatch(r"[0-9a-f]{16}", clip_id):
                raise ValueError("Invalid whole-video clip ID")

            if row["label"] != label_of(
                row["manipulation_type"]
            ):
                raise ValueError("Label/category mismatch")

            directory = cache_root / "val" / clip_id

            feature_path = (
                directory / "features.npz"
            ).resolve()

            metadata_path = (
                directory / "meta.json"
            ).resolve()

            # Reject unexpected paths or symlink escapes.
            if (
                not feature_path.is_relative_to(cache_root)
                or not metadata_path.is_relative_to(cache_root)
            ):
                raise ValueError("Cache path escapes cache root")

            if feature_path in seen_paths:
                raise ValueError("Duplicate cache path")

            record = inspect_npz(
                feature_path,
                metadata_path,
                clip_id,
            )

            if record["feature_sha256"] in seen_hashes:
                raise ValueError(
                    "Duplicate NPZ archive hash"
                )

            seen_paths.add(feature_path)
            seen_hashes.add(record["feature_sha256"])

            # Preserve labels, IDs, groups and all other
            # scientific metadata. Correct only the
            # cache location in this derived manifest.
            corrected = dict(row)
            corrected["feature_path"] = str(feature_path)

            validated_rows.append(corrected)

            inventory.append({
                **record,
                "source_id": row["source_id"],
                "manipulation_type": row["manipulation_type"],
                "original_feature_path": row["feature_path"],
            })

        except (
            ValueError,
            FileNotFoundError,
            OSError,
            KeyError,
        ) as exc:
            errors.append(
                f"{clip_id}: {type(exc).__name__}: {exc}"
            )

    clusters = group_clusters(rows)

    report = {
        "status": "FAIL" if errors else "PASS",
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": sha256_file(
            manifest_path
        ),
        "cache_root": str(cache_root),
        "checkpoint_sha256": checkpoint_sha,
        "git_revision": git_value("rev-parse", "HEAD"),
        "git_worktree_status": git_value(
            "status", "--porcelain"
        ),
        "n_expected": expected_count,
        "n_manifest": len(rows),
        "n_verified": len(inventory),
        "counts_by_type": dict(counts),
        "n_source_identity_clusters": len(set(
            clusters.values()
        )),
        "errors_count": len(errors),
        "errors": errors[:100],
        "cache_inventory": inventory,
    }

    out_report.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if errors:
        out_report.write_text(
            json.dumps(report, indent=2)
        )
        raise RuntimeError(
            f"Preflight failed: {len(errors)} problem(s). "
            f"See {out_report}"
        )

    out_manifest.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Uses Member 2's existing manifest schema validator.
    write_manifest(validated_rows, out_manifest)

    report["validated_manifest"] = str(
        out_manifest.resolve()
    )
    report["validated_manifest_sha256"] = sha256_file(
        out_manifest
    )

    out_report.write_text(
        json.dumps(report, indent=2)
    )

    print("DAY 6 PREFLIGHT: PASS")
    print("Verified validation clips:", len(inventory))
    print("Category counts:", dict(counts))
    print("Source/identity clusters:", len(set(
        clusters.values()
    )))
    print("Checkpoint SHA256:", checkpoint_sha)
    print("Validated manifest:", out_manifest)
    print("Inventory report:", out_report)

    return report


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-manifest", required=True)
    parser.add_argument("--out-report", required=True)

    parser.add_argument(
        "--expected-count", type=int, default=1200
    )
    parser.add_argument(
        "--expected-per-type", type=int, default=300
    )

    args = parser.parse_args()

    run_preflight(
        manifest_path=args.manifest,
        cache_root=args.cache_root,
        checkpoint_path=args.checkpoint,
        out_manifest=args.out_manifest,
        out_report=args.out_report,
        expected_count=args.expected_count,
        expected_per_type=args.expected_per_type,
    )


if __name__ == "__main__":
    main()
