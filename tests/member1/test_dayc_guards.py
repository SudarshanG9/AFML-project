"""Synthetic safety fixtures only; never use real held-out assets or approvals."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import pytest

from avfuse.member1_fusion import dayc_guards as guards


def test_containment(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    assert guards.ensure_inside_repo(root, Path("new/output.json")) == root / "new/output.json"
    assert guards.ensure_inside_repo(root, root) == root
    with pytest.raises(FileNotFoundError):
        guards.ensure_inside_repo(root, Path("absent"), must_exist=True)
    for path in [Path("../escape"), tmp_path / "repo-sibling/file"]:
        with pytest.raises(PermissionError):
            guards.ensure_inside_repo(root, path)


def test_symlink_escape_and_internal_link(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "outside").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(PermissionError):
        guards.ensure_inside_repo(root, Path("outside/escape"))
    (root / "inside").mkdir()
    (root / "alias").symlink_to(root / "inside", target_is_directory=True)
    assert guards.ensure_inside_repo(root, Path("alias/new")) == root / "inside/new"


def test_hashes_and_absent_checkpoint(tmp_path):
    path = tmp_path / "checkpoint.pth"
    with pytest.raises(FileNotFoundError):
        guards.require_sha256(path, "0" * 64)
    path.write_bytes(b"TEST_FIXTURE; not a model")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    guards.require_sha256(path, digest)
    with pytest.raises(ValueError, match="mismatch"):
        guards.require_sha256(path, "0" * 64)
    with pytest.raises(ValueError):
        guards.require_sha256(path, "not-a-digest")
    alias = tmp_path / "alias"
    alias.symlink_to(path)
    with pytest.raises(ValueError):
        guards.sha256_file(alias)
    with pytest.raises(ValueError):
        guards.sha256_file(tmp_path)


@pytest.mark.parametrize("kind", ["file", "directory", "dangling_symlink"])
def test_existing_destination_rejected(tmp_path, kind):
    path = tmp_path / "existing"
    if kind == "file":
        path.write_bytes(b"preserve")
    elif kind == "directory":
        path.mkdir()
    else:
        path.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        guards.require_new_file(path)
    with pytest.raises(FileExistsError):
        guards.write_new_json(path, {"new": 1})
    if kind == "file":
        assert path.read_bytes() == b"preserve"


def test_deterministic_atomic_outputs(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    digest = guards.write_new_json(a, {"z": 3, "a": {"y": 2, "x": 1}})
    assert guards.write_new_json(b, {"a": {"x": 1, "y": 2}, "z": 3}) == digest
    assert guards.sha256_file(a) == digest
    rows = [{"score_fake": 0.5}, {"score_fake": 1.0}]
    out = tmp_path / "scores.jsonl"
    assert guards.write_new_jsonl(out, rows) == guards.sha256_file(out)
    assert [json.loads(line) for line in out.read_text().splitlines()] == rows
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_rejected_before_output(tmp_path, value):
    for name, writer, payload in [("x.json", guards.write_new_json, {"nested": [value]}),
                                  ("x.jsonl", guards.write_new_jsonl,
                                   [{"score_fake": 0.5}, {"score_fake": value}])]:
        path = tmp_path / name
        with pytest.raises(ValueError):
            writer(path, payload)
        assert not path.exists()
    assert not list(tmp_path.iterdir())


def test_exclusive_commit_race_preserves_other_writer(tmp_path, monkeypatch):
    target = tmp_path / "output.json"
    real_link = guards.os.link
    def racing_link(source, destination):
        Path(destination).write_bytes(b"other writer")
        return real_link(source, destination)
    monkeypatch.setattr(guards.os, "link", racing_link)
    with pytest.raises(FileExistsError):
        guards.write_new_json(target, {"ours": True})
    assert target.read_bytes() == b"other writer"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["output.json"]


def approval_fixture(tmp_path):
    # TEST_FIXTURE: this declaration is not research approval evidence.
    path = tmp_path / "TEST_FIXTURE_approval.json"
    record = {"gate": "C0", "ack_text": guards.GATE_ACKS["C0"],
              "context_sha256": "a" * 64, "user_supplied": True,
              "approved_at": (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()}
    return path, record


def test_valid_approval_is_read_only(tmp_path):
    path, record = approval_fixture(tmp_path)
    path.write_text(json.dumps(record))
    before = path.read_bytes()
    assert guards.require_human_approval(path, "C0", context_sha256="a" * 64) == record
    assert path.read_bytes() == before


@pytest.mark.parametrize("key,value", [("gate", "C8A"), ("ack_text", "PASS"),
    ("context_sha256", "b" * 64), ("user_supplied", False), ("user_supplied", 1),
    ("approved_at", "2026-01-01T00:00:00"), ("approved_at", "not-a-date"),
    ("approved_at", "2999-01-01T00:00:00Z")])
def test_invalid_or_stale_approval(tmp_path, key, value):
    path, record = approval_fixture(tmp_path)
    record[key] = value
    path.write_text(json.dumps(record))
    with pytest.raises(PermissionError):
        guards.require_human_approval(path, "C0", context_sha256="a" * 64)


def test_missing_unknown_or_ambiguous_approval(tmp_path):
    path, record = approval_fixture(tmp_path)
    with pytest.raises(FileNotFoundError):
        guards.require_human_approval(path, "C0", context_sha256="a" * 64)
    with pytest.raises(ValueError):
        guards.require_human_approval(path, "UNKNOWN", context_sha256="a" * 64)
    for text in ['{"gate":"C0","gate":"C8A"}', '{"x":NaN}', '{"x":1e999}']:
        path.write_text(text)
        with pytest.raises(ValueError):
            guards.read_strict_json(path)


@pytest.mark.parametrize("phase", ["baseline", "day8", "day9", "day10-readonly"])
@pytest.mark.parametrize("approved", [False, True])
def test_no_test_access_before_heldout_execution(phase, approved):
    with pytest.raises(PermissionError):
        guards.require_allowed_split("test", phase=phase, test_approved=approved)
    guards.require_allowed_split("val", phase=phase, test_approved=approved)


@pytest.mark.parametrize("phase", ["baseline", "day8", "day9", "day10", "unknown"])
def test_training_always_rejected(phase):
    with pytest.raises(PermissionError):
        guards.require_allowed_split("train", phase=phase, test_approved=True)


def test_heldout_requires_explicit_boolean_and_correct_phase():
    guards.require_allowed_split("test", phase="day10", test_approved=True)
    for flag in [False, 1, "true", None]:
        with pytest.raises(PermissionError):
            guards.require_allowed_split("test", phase="day10", test_approved=flag)
    with pytest.raises(PermissionError):
        guards.require_allowed_split("val", phase="unknown", test_approved=True)


def test_frozen_inputs_exact_and_unchanged(tmp_path):
    path = tmp_path / "input"
    path.write_bytes(b"frozen")
    digest = guards.sha256_file(path)
    guards.require_frozen_inputs({"input": digest}, {"input": path})
    for expected, paths in [({}, {}), ({"input": digest}, {}),
                             ({"input": digest}, {"input": path, "extra": path})]:
        with pytest.raises(ValueError):
            guards.require_frozen_inputs(expected, paths)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError):
        guards.require_frozen_inputs({"input": digest}, {"input": path})


# C0 preflight uses the existing Member 2 contracts on synthetic validation rows.
from scripts.dayC_member1 import preflight


@pytest.fixture
def synthetic_val_rows():
    categories = list(preflight.CATEGORIES)
    return [{"clip_id": f"{i:016x}", "dataset": "TEST_FIXTURE", "source_id": str(i),
             "identity_id": str(i), "split": "val", "label": int(i // 300 != 0),
             "manipulation_type": categories[i // 300],
             "feature_path": f"cache/val/{i:016x}/features.npz"} for i in range(1200)]


def write_fixture_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_preflight_complete_validation_metadata(tmp_path, synthetic_val_rows):
    path = tmp_path / "TEST_FIXTURE_val.jsonl"
    write_fixture_jsonl(path, synthetic_val_rows)
    before = path.read_bytes()
    rows = preflight._read_val_manifest(path)
    assert len(preflight._reconcile_val_manifests(rows, rows)) == 1200
    assert path.read_bytes() == before


@pytest.mark.parametrize("mutation", ["short", "duplicate", "wrong_category", "test", "nonfinite"])
def test_preflight_rejects_bad_validation_metadata(tmp_path, synthetic_val_rows, mutation):
    rows = synthetic_val_rows
    if mutation == "short":
        rows = rows[:-1]
    elif mutation == "duplicate":
        rows[-1] = dict(rows[0])
    elif mutation == "wrong_category":
        rows[-1]["manipulation_type"] = "audio_fake"
    elif mutation == "test":
        rows[0]["split"] = "test"  # Synthetic TEST_FIXTURE, no held-out data.
    else:
        rows[0]["duration_s"] = float("nan")
    path = tmp_path / "TEST_FIXTURE_val.jsonl"
    write_fixture_jsonl(path, rows)
    with pytest.raises((ValueError, PermissionError)):
        preflight._read_val_manifest(path)


def test_preflight_rejects_changed_ready_labels(synthetic_val_rows):
    changed = [dict(row) for row in synthetic_val_rows]
    changed[0]["source_id"] = "different-source"
    with pytest.raises(ValueError):
        preflight._reconcile_val_manifests(synthetic_val_rows, changed)


@pytest.mark.parametrize("mutation", [None, "duplicate_across_runs", "wrong_checkpoint", "test", "labels"])
def test_preflight_prediction_contract(tmp_path, synthetic_val_rows, mutation):
    from avfuse.member2_data.predictions import ContractError
    rows = [{"run_id": "TEST_FIXTURE", "clip_id": row["clip_id"], "split": "val",
             "config": "reference", "score_fake": 0.5,
             "checkpoint_sha256": preflight.CHECKPOINT} for row in synthetic_val_rows]
    if mutation == "duplicate_across_runs":
        rows[-1]["clip_id"] = rows[0]["clip_id"]
        rows[-1]["run_id"] = "TEST_FIXTURE_second_run"
    elif mutation == "wrong_checkpoint":
        for row in rows:
            row["checkpoint_sha256"] = "0" * 64
    elif mutation == "test":
        rows[0]["split"] = "test"
    elif mutation == "labels":
        rows[0]["label"] = 0
    path = tmp_path / "TEST_FIXTURE_scores.jsonl"
    write_fixture_jsonl(path, rows)
    operation = lambda: preflight._check_predictions(
        path, "reference", {row["clip_id"] for row in synthetic_val_rows}, preflight.CHECKPOINT)
    if mutation is None:
        assert operation()["rows"] == 1200
    else:
        with pytest.raises((ValueError, PermissionError, ContractError)):
            operation()


def test_preflight_cache_metadata_only_and_no_external_read(tmp_path, monkeypatch):
    path = tmp_path / "val/clip/features.npz"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"TEST_FIXTURE; must never be loaded")
    original_open = Path.open
    def no_feature_reads(self, *args, **kwargs):
        if self.suffix == ".npz":
            pytest.fail("preflight tried to read feature tensors")
        return original_open(self, *args, **kwargs)
    monkeypatch.setattr(Path, "open", no_feature_reads)
    report = preflight._check_feature_paths(tmp_path, [
        {"clip_id": "clip", "split": "val", "feature_path": str(path)},
        {"clip_id": "missing", "split": "val", "feature_path": "val/missing/features.npz"},
        {"clip_id": "external", "split": "val", "feature_path": "/content/val/external/features.npz"}])
    assert report["existing_local_paths"] == 1
    assert len(report["unresolved_paths"]) == 2
    assert not report["feature_file_contents_read"]
    assert not report["external_paths_opened"]


@pytest.mark.parametrize("phase", ["day8", "day9", "day10-readonly"])
def test_unreleased_preflight_phases_open_no_data(tmp_path, monkeypatch, phase):
    def forbidden(*args, **kwargs):
        pytest.fail("unreleased phase opened a file")
    monkeypatch.setattr(Path, "open", forbidden)
    report = preflight.build_baseline_report(tmp_path, Path("unavailable.json"), phase=phase)
    assert report["status"] == "BLOCKED"
    assert not report["test_feature_access"]
    assert not report["canonical_manifest_rows_inspected"]


def test_missing_spec_becomes_blocked_report(tmp_path):
    report = preflight.build_baseline_report(tmp_path, Path("missing.json"))
    assert report["status"] == "BLOCKED"
    assert "spec_path" in report["blockers"]



def test_complete_preflight_report_is_json_serializable(tmp_path, monkeypatch):
    # Metadata-only TEST_FIXTURE root, with no model/cache or held-out rows.
    for name in [".git", "src", "scripts", "tests/member1"]:
        (tmp_path / name).mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text("# TEST_FIXTURE\n")
    actual_root = Path(preflight.__file__).resolve().parents[2]
    spec = json.loads((actual_root / "configs/dayC/member1/experiment_spec.json").read_text())
    fixture_spec = tmp_path / "TEST_FIXTURE_spec.json"
    fixture_spec.write_text(json.dumps(spec))
    def fixture_git(root, *args):
        return {("branch", "--show-current"): "member1",
                ("rev-parse", "HEAD"): spec["expected_baseline"]["head"],
                ("status", "--short"): ""}[args]
    monkeypatch.setattr(preflight, "_git", fixture_git)
    monkeypatch.setattr(preflight, "_environment", lambda: {})
    report = preflight.build_baseline_report(tmp_path, fixture_spec)
    assert report["status"] == "BLOCKED"
    assert "input_checkpoint" in report["blockers"]
    guards.write_new_json(tmp_path / "TEST_FIXTURE_report.json", report)
    assert json.loads((tmp_path / "TEST_FIXTURE_report.json").read_text())["status"] == "BLOCKED"



def test_preflight_output_dangling_symlink_is_not_a_new_name(tmp_path):
    parent = tmp_path / "artifacts/dayC/member1/preflight"
    parent.mkdir(parents=True)
    target = parent / "not_created.json"
    link = parent / "existing_link.json"
    link.symlink_to(target)
    with pytest.raises(FileExistsError):
        preflight._new_report_path(tmp_path, link)
    assert link.is_symlink()
    assert not target.exists()
    assert preflight._new_report_path(tmp_path, parent / "new.json") == parent / "new.json"
    with pytest.raises(PermissionError):
        preflight._new_report_path(tmp_path, Path("src/should_not_write.json"))
