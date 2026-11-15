"""Segment E prediction generation. Mock adapters + synthetic temp files; no real results."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from avfuse.member2_data.predictions import read_predictions, validate_predictions
from avfuse.member2_data.seal import accesses
from avfuse.member3_runtime import predictions as P
from avfuse.member3_runtime.feature_loader import ClipRef, make_feature_provider
from avfuse.member3_runtime.runner import ModeFailedError
from avfuse.shared.contracts import PRECISION_MODES

SHA = "ab" * 32


class Mock:
    def __init__(self, supported=PRECISION_MODES, fail_on_call=None):
        self.supported, self.n, self.fail_on_call = set(supported), 0, fail_on_call
        self.seen_clip_features = []

    def predict_from_cached_features(self, features, mode):
        self.n += 1
        if self.fail_on_call == self.n:
            raise RuntimeError("boom")
        if mode not in self.supported:
            raise NotImplementedError(f"{mode} pending Member 1")
        self.seen_clip_features.append(tuple(features["video_tokens"].shape))
        base = {"reference": 0.1, "low_precision": 0.2, "restore_value_output": 0.3,
                "restore_query": 0.4}[mode]
        return torch.tensor([base])


def make_features(root, split, ids):
    for i, cid in enumerate(ids):
        d = root / split / cid
        d.mkdir(parents=True)
        np.savez(d / "features.npz", video_features=np.ones((4 + i, 768), np.float32),
                 audio_features=np.ones((3 + i, 768), np.float32))


def make_manifest(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.fixture
def world(tmp_path):
    root = tmp_path / "feats"
    ids_val, ids_test = ["v1", "v2", "v3"], ["t1", "t2"]
    make_features(root, "val", ids_val)
    make_features(root, "test", ids_test)
    rows = [{"clip_id": c, "split": "val", "label": 1, "manipulation_type": "both_fake",
             "source_id": "s", "feature_path": f"data/cache/{c}.pt"} for c in ids_val]
    rows += [{"clip_id": c, "split": "test", "label": 0, "manipulation_type": "real",
              "source_id": "s", "feature_path": f"data/cache/{c}.pt"} for c in ids_test]
    manifest = tmp_path / "manifest_whole.jsonl"
    make_manifest(manifest, rows)
    return tmp_path, root, manifest


def run(world, model=None, **kw):
    tmp, root, manifest = world
    split = kw.get("split", "val")
    kw.setdefault("output_path", tmp / "out" / "pred.jsonl")
    return P.generate_predictions(
        model or Mock(), P.load_clip_refs(manifest, split), make_feature_provider("dimodif_npz", root),
        run_id="run1", checkpoint_sha256=SHA, **kw)


def test_four_mode_rows_are_schema_valid_label_free(world):
    rep = run(world)
    assert rep.status == "complete" and rep.four_mode_complete and rep.published
    assert rep.n_clips == 3 and rep.n_rows == 12 and rep.full_split
    rows = read_predictions(rep.output_path)  # Member 2's public validator
    validate_predictions(rows)
    assert {(r["clip_id"], r["config"]) for r in rows} == {
        (c, m) for c in ("v1", "v2", "v3") for m in PRECISION_MODES}
    for r in rows:
        assert set(r) == set(P.PREDICTION_FIELDS)
        assert r["split"] == "val" and r["run_id"] == "run1" and r["checkpoint_sha256"] == SHA
        assert 0 <= r["score_fake"] <= 1
    text = rep.output_path.read_text()
    for forbidden in ("label", "ground_truth", "manipulation_type", "source_id"):
        assert forbidden not in text


def test_default_split_is_val_and_test_features_not_read(world, monkeypatch):
    read = []
    real = np.load
    monkeypatch.setattr("avfuse.member3_runtime.feature_loader.np.load",
                        lambda p, **k: read.append(str(p)) or real(p, **k))
    rep = run(world)
    assert rep.split == "val"
    assert read and all(Path(p).parent.parent.name == "val" for p in read)


def test_test_split_rejected_without_authorization_before_reading_features(world, monkeypatch):
    tmp = world[0]
    read = []
    real = np.load
    monkeypatch.setattr("avfuse.member3_runtime.feature_loader.np.load",
                        lambda p, **k: read.append(p) or real(p, **k))
    model = Mock()
    protocol = tmp / "protocol.json"
    protocol.write_text("{}")
    cases = [
        {},                                                                 # nothing
        {"protocol_path": protocol},                                        # no unseal
        {"protocol_path": protocol, "unseal": True},                        # no purpose
        {"protocol_path": protocol, "unseal": True, "purpose": "  "},       # blank purpose
        {"protocol_path": protocol, "unseal": "yes", "purpose": "final"},   # not literally True
        {"protocol_path": tmp / "missing.json", "unseal": True, "purpose": "final"},  # no file
        {"unseal": True, "purpose": "final"},                               # no protocol
    ]
    for kw in cases:
        with pytest.raises(P.TestAccessError):
            run(world, model, split="test", output_path=tmp / "t.jsonl", **kw)
    assert read == [] and model.n == 0 and not (tmp / "t.jsonl").exists()
    assert accesses(protocol) == []


def test_authorized_test_run_logs_exactly_once_with_all_configs(world):
    tmp, _, _ = world
    protocol = tmp / "protocol.json"
    protocol.write_text("{}")
    rep = run(world, split="test", output_path=tmp / "t.jsonl", protocol_path=protocol,
              unseal=True, purpose="final held-out run")
    assert rep.test_access_logged and rep.n_clips == 2 and rep.n_rows == 8
    log = accesses(protocol)
    assert len(log) == 1
    assert log[0]["purpose"] == "final held-out run"
    assert log[0]["configs"] == sorted(PRECISION_MODES)
    assert {r["split"] for r in read_predictions(rep.output_path)} == {"test"}


def test_val_run_never_logs_test_access(world):
    tmp, _, _ = world
    protocol = tmp / "protocol.json"
    protocol.write_text("{}")
    rep = run(world, protocol_path=protocol, unseal=True, purpose="irrelevant")
    assert not rep.test_access_logged and accesses(protocol) == []


def test_missing_clip_feature_fails_and_publishes_nothing(world):
    tmp, root, _ = world
    (root / "val" / "v2" / "features.npz").unlink()
    with pytest.raises(FileNotFoundError):
        run(world)
    assert not (tmp / "out" / "pred.jsonl").exists()
    assert not list((tmp / "out").glob("*")) if (tmp / "out").exists() else True


def test_mid_run_failure_publishes_no_completed_file(world):
    tmp, _, _ = world
    with pytest.raises(ModeFailedError):
        run(world, Mock(fail_on_call=7))  # fails during the 2nd clip
    assert not (tmp / "out" / "pred.jsonl").exists()
    assert not list((tmp / "out").glob(".*tmp")) if (tmp / "out").exists() else True


def test_existing_output_preserved_when_overwrite_disabled(world):
    tmp, _, _ = world
    out = tmp / "out" / "pred.jsonl"
    out.parent.mkdir()
    out.write_text("KEEP\n")
    with pytest.raises(FileExistsError):
        run(world)
    assert out.read_text() == "KEEP\n"
    run(world, overwrite=True)
    assert out.read_text() != "KEEP\n"


def test_strict_four_mode_with_reference_only_model_is_blocked(world):
    tmp, _, _ = world
    model = Mock(supported=("reference",))
    rep = run(world, model)
    assert rep.status == "blocked" and not rep.published and rep.output_path is None
    assert not rep.four_mode_complete and rep.n_rows == 0
    assert set(rep.unsupported_modes) == {"low_precision", "restore_value_output", "restore_query"}
    assert rep.n_clips == 1 and not rep.full_split  # stopped at the first clip
    assert not (tmp / "out").exists()
    assert len(model.seen_clip_features) == 1


def test_reference_only_request_publishes_only_reference_rows(world):
    rep = run(world, Mock(supported=("reference",)), modes=("reference",))
    assert rep.status == "complete" and not rep.four_mode_complete
    rows = read_predictions(rep.output_path)
    assert {r["config"] for r in rows} == {"reference"} and len(rows) == 3
    assert any("Not a four-mode result" in n for n in rep.notes)


def test_partial_mode_is_labelled_and_skips_known_unsupported(world):
    model = Mock(supported=("reference",))
    rep = run(world, model, allow_partial=True)
    assert rep.status == "partial" and rep.published and not rep.four_mode_complete
    assert rep.completed_modes == ("reference",) and len(rep.unsupported_modes) == 3
    assert {r["config"] for r in read_predictions(rep.output_path)} == {"reference"}
    # unsupported modes are probed on the first clip only: 3 clips * 1 + 3 probes
    assert model.n == 3 + 3
    assert any("PARTIAL" in n for n in rep.notes)


def test_checkpoint_hash_checks(world, monkeypatch):
    tmp, _, _ = world
    with pytest.raises(P.CheckpointMismatchError, match="mismatch"):
        run(world, expected_checkpoint_sha256="cd" * 32)
    with pytest.raises(P.CheckpointMismatchError, match="64 lowercase"):
        P.check_checkpoint_hash("XYZ")
    with pytest.raises(P.CheckpointMismatchError):
        run(world, require_lavdf_hash=True)  # SHA is not the agreed LAV-DF hash
    from avfuse.member2_data.readiness import DIMODIF_LAVDF_SHA256
    P.check_checkpoint_hash(DIMODIF_LAVDF_SHA256, require_lavdf=True)
    assert not (tmp / "out").exists()
    with pytest.raises(ValueError):
        P.check_checkpoint_hash(SHA, "cd" * 32, require_lavdf=True)


def test_one_checkpoint_hash_and_no_duplicates_in_output(world):
    rows = read_predictions(run(world).output_path)
    assert {r["checkpoint_sha256"] for r in rows} == {SHA}
    keys = [(r["run_id"], r["config"], r["clip_id"]) for r in rows]
    assert len(keys) == len(set(keys))


def test_load_clip_refs_is_label_free_and_split_filtered(world):
    _, _, manifest = world
    refs = P.load_clip_refs(manifest, "val")
    assert [r.clip_id for r in refs] == ["v1", "v2", "v3"]
    assert all(not hasattr(r, "label") and r.split == "val" for r in refs)
    with pytest.raises(ValueError):
        P.load_clip_refs(manifest, "train")


def test_duplicate_manifest_ids_rejected(tmp_path):
    m = tmp_path / "m.jsonl"
    make_manifest(m, [{"clip_id": "a", "split": "val"}, {"clip_id": "a", "split": "val"}])
    with pytest.raises(ValueError, match="duplicate"):
        P.load_clip_refs(m, "val")


def test_max_clips_marks_subset_run(world):
    rep = run(world, max_clips=2)
    assert rep.n_clips == 2 and rep.n_clips_in_split == 3 and not rep.full_split
    assert rep.n_rows == 8


def test_clips_of_other_split_are_ignored_not_read():
    seen = []
    clips = [ClipRef("a", "val"), ClipRef("b", "test")]

    def provider(c):
        seen.append(c.clip_id)
        return {"video_tokens": torch.zeros(2, 768), "audio_tokens": torch.zeros(2, 768)}

    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        P.generate_predictions(Mock(), clips, provider, run_id="r", checkpoint_sha256=SHA,
                               output_path=Path(d) / "p.jsonl")
    assert seen == ["a"]
