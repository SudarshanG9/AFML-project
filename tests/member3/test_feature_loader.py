"""Segment E feature loader. Synthetic temporary NPZ/PT files only; no real data."""
import hashlib

import numpy as np
import pytest
import torch

from avfuse.member2_data.cache import save_feature_cache
from avfuse.member3_runtime import feature_loader as fl
from avfuse.member3_runtime.feature_loader import (
    ClipRef,
    FeatureLoadError,
    dimodif_npz_path,
    load_clip_features,
    load_dimodif_features,
    load_shared_pt_features,
    make_feature_provider,
)


def write_npz(path, tv=7, ta=5, dim=768, dtype=np.float32, **over):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    arrays = {"video_features": rng.standard_normal((tv, dim)).astype(dtype),
              "audio_features": rng.standard_normal((ta, dim)).astype(dtype)}
    arrays.update(over)
    arrays = {k: v for k, v in arrays.items() if v is not None}
    np.savez(path, **arrays)
    return path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_valid_npz_maps_keys_and_keeps_different_lengths(tmp_path):
    p = write_npz(tmp_path / "f.npz", tv=9, ta=4)
    out = load_dimodif_features(p)
    assert set(out) == {"video_tokens", "audio_tokens"}
    assert out["video_tokens"].shape == (9, 768) and out["audio_tokens"].shape == (4, 768)
    assert out["video_tokens"].dtype == torch.float32 == out["audio_tokens"].dtype
    with np.load(p) as ref:
        np.testing.assert_array_equal(out["video_tokens"].numpy(), ref["video_features"])
        np.testing.assert_array_equal(out["audio_tokens"].numpy(), ref["audio_features"])


def test_float64_is_converted_to_fp32_and_file_not_modified(tmp_path):
    p = write_npz(tmp_path / "f.npz", dtype=np.float64)
    before = sha(p)
    out = load_dimodif_features(p)
    assert out["video_tokens"].dtype == torch.float32
    assert sha(p) == before
    assert sorted(x.name for x in tmp_path.iterdir()) == ["f.npz"]  # no .pt copy


@pytest.mark.parametrize("over,match", [
    ({"video_features": np.zeros((0, 768), np.float32)}, "empty"),
    ({"audio_features": np.zeros((3, 512), np.float32)}, "dimension"),
    ({"video_features": np.zeros((2, 3, 768), np.float32)}, "rank 2"),
    ({"audio_features": np.zeros((3, 768), np.int32)}, "floating"),
    ({"video_features": np.full((3, 768), np.nan, np.float32)}, "non-finite"),
    ({"audio_features": np.full((3, 768), np.inf, np.float32)}, "non-finite"),
])
def test_invalid_arrays_rejected(tmp_path, over, match):
    p = write_npz(tmp_path / "f.npz", **over)
    with pytest.raises(FeatureLoadError, match=match):
        load_dimodif_features(p)


def test_missing_key_rejected(tmp_path):
    p = write_npz(tmp_path / "f.npz", audio_features=None)
    with pytest.raises(FeatureLoadError, match="audio_features"):
        load_dimodif_features(p)


def test_missing_file_is_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing DiMoDif feature file"):
        load_dimodif_features(tmp_path / "nope.npz")


def test_pickled_object_arrays_rejected(tmp_path):
    p = tmp_path / "f.npz"
    np.savez(p, video_features=np.array([{"a": 1}], dtype=object),
             audio_features=np.zeros((2, 768), np.float32))
    with pytest.raises(FeatureLoadError, match="pickle-free"):
        load_dimodif_features(p)


def test_allow_pickle_false_is_used(tmp_path, monkeypatch):
    p = write_npz(tmp_path / "f.npz")
    seen = {}
    real = np.load

    def spy(*a, **k):
        seen.update(k)
        return real(*a, **k)

    monkeypatch.setattr(fl.np, "load", spy)
    load_dimodif_features(p)
    assert seen["allow_pickle"] is False


def test_whole_video_path_uses_feature_root_not_manifest_pt_path(tmp_path):
    assert dimodif_npz_path(tmp_path, "val", "abc123") == tmp_path / "val" / "abc123" / "features.npz"
    write_npz(tmp_path / "val" / "abc123" / "features.npz")
    clip = ClipRef("abc123", "val", feature_path="data/cache/abc123.pt")  # planned .pt path
    out = load_clip_features(clip, feature_format="dimodif_npz", feature_root=tmp_path)
    assert out["video_tokens"].shape[1] == 768
    with pytest.raises(ValueError, match="explicit feature_root"):
        load_clip_features(clip, feature_format="dimodif_npz")


@pytest.mark.parametrize("split,cid", [("train", "a"), ("val", "../x"), ("val", "a/b"), ("val", "")])
def test_unsafe_path_parts_rejected(tmp_path, split, cid):
    with pytest.raises(ValueError):
        dimodif_npz_path(tmp_path, split, cid)


def test_unknown_format_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown feature format"):
        load_clip_features(ClipRef("a", "val"), feature_format="auto", feature_root=tmp_path)
    with pytest.raises(ValueError):
        make_feature_provider("auto")


def test_shared_pt_uses_member2_loader(tmp_path, monkeypatch):
    p = tmp_path / "c.pt"
    save_feature_cache(p, {"video_tokens": torch.randn(6, 768), "audio_tokens": torch.randn(3, 768)})
    calls = []
    real = fl.load_feature_cache
    monkeypatch.setattr(fl, "load_feature_cache", lambda path: calls.append(path) or real(path))
    out = load_shared_pt_features(p)
    assert calls == [p]
    assert out["video_tokens"].shape == (6, 768) and out["audio_tokens"].shape == (3, 768)
    clip = ClipRef("c", "val", feature_path="c.pt")
    assert load_clip_features(clip, feature_format="shared_pt", feature_root=tmp_path)


@pytest.mark.parametrize("v,a,match", [
    (torch.randn(4, 512), torch.randn(4, 768), "dimension"),
    (torch.full((4, 768), float("nan")), torch.randn(4, 768), "non-finite"),
    (torch.randn(4, 768), torch.randn(2, 4, 768), "same rank"),
    (torch.zeros(4, 768, dtype=torch.long), torch.randn(4, 768), "floating"),
])
def test_shared_pt_invalid(tmp_path, v, a, match):
    p = tmp_path / "c.pt"
    torch.save({"video_tokens": v, "audio_tokens": a}, p)
    with pytest.raises(FeatureLoadError, match=match):
        load_shared_pt_features(p)


def test_clipref_has_no_label_field():
    assert not hasattr(ClipRef("a", "val"), "label")
    assert {f for f in ClipRef.__dataclass_fields__} == {"clip_id", "split", "feature_path"}


def test_provider_loads_from_root(tmp_path):
    write_npz(tmp_path / "val" / "v1" / "features.npz")
    provider = make_feature_provider("dimodif_npz", tmp_path)
    assert provider(ClipRef("v1", "val"))["audio_tokens"].shape == (5, 768)
