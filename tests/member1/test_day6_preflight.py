
import hashlib
import json

import numpy as np
import pytest

from scripts.day6_preflight import run_preflight
from avfuse.member2_data.manifest import read_manifest


CATEGORIES = (
    ("real", 0),
    ("audio_fake", 1),
    ("video_fake", 1),
    ("both_fake", 1),
)


def build_fixture(tmp_path):
    cache_root = tmp_path / "cache"
    rows = []

    for i, (category, label) in enumerate(CATEGORIES):
        clip_id = f"{i + 1:016x}"

        folder = cache_root / "val" / clip_id
        folder.mkdir(parents=True)

        video = np.full(
            (5, 768),
            0.1 + i,
            dtype=np.float32,
        )

        audio = np.full(
            (6, 768),
            0.2 + i,
            dtype=np.float32,
        )

        np.savez(
            folder / "features.npz",
            video_features=video,
            audio_features=audio,
        )

        metadata = {
            "status": "ok",
            "video_id": clip_id,
            "split": "val",
            "T_v": 5,
            "T_a": 6,
            "D_v": 768,
            "D_a": 768,
            "dtype": "float32",
        }

        (folder / "meta.json").write_text(
            json.dumps(metadata)
        )

        rows.append({
            "clip_id": clip_id,
            "dataset": "lavdf",
            "source_id": f"source_{i}",
            "identity_id": f"source_{i}",
            "split": "val",
            "label": label,
            "manipulation_type": category,
            "feature_path": f"data/cache/{clip_id}.pt",
        })

    manifest = tmp_path / "manifest.jsonl"

    with manifest.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"test-checkpoint")

    expected_sha = hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()

    return manifest, cache_root, checkpoint, expected_sha


def execute_preflight(tmp_path):
    manifest, cache, checkpoint, sha = (
        build_fixture(tmp_path)
    )

    out_manifest = tmp_path / "ready.jsonl"
    out_report = tmp_path / "report.json"

    report = run_preflight(
        manifest_path=manifest,
        cache_root=cache,
        checkpoint_path=checkpoint,
        out_manifest=out_manifest,
        out_report=out_report,
        expected_count=4,
        expected_per_type=1,
        expected_checkpoint_sha=sha,
    )

    return report, out_manifest, out_report


def test_preflight_accepts_valid_cache(tmp_path):
    report, out_manifest, _ = execute_preflight(
        tmp_path
    )

    assert report["status"] == "PASS"
    assert report["n_verified"] == 4

    rows = read_manifest(out_manifest)

    assert len(rows) == 4

    for row in rows:
        assert row["feature_path"].endswith(
            "features.npz"
        )


def test_missing_cache_is_rejected(tmp_path):
    manifest, cache, checkpoint, sha = (
        build_fixture(tmp_path)
    )

    missing = (
        cache / "val" / "0000000000000001" / "features.npz"
    )
    missing.unlink()

    out_manifest = tmp_path / "ready.jsonl"
    out_report = tmp_path / "report.json"

    with pytest.raises(RuntimeError):
        run_preflight(
            manifest,
            cache,
            checkpoint,
            out_manifest,
            out_report,
            expected_count=4,
            expected_per_type=1,
            expected_checkpoint_sha=sha,
        )

    assert not out_manifest.exists()

    report = json.loads(out_report.read_text())
    assert report["status"] == "FAIL"


def test_nonfinite_features_are_rejected(tmp_path):
    manifest, cache, checkpoint, sha = (
        build_fixture(tmp_path)
    )

    path = (
        cache / "val" / "0000000000000001" / "features.npz"
    )

    video = np.zeros((5, 768), dtype=np.float32)
    video[0, 0] = np.nan

    audio = np.zeros((6, 768), dtype=np.float32)

    np.savez(
        path,
        video_features=video,
        audio_features=audio,
    )

    with pytest.raises(RuntimeError):
        run_preflight(
            manifest,
            cache,
            checkpoint,
            tmp_path / "ready.jsonl",
            tmp_path / "report.json",
            expected_count=4,
            expected_per_type=1,
            expected_checkpoint_sha=sha,
        )
