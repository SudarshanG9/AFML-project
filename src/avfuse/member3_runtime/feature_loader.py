"""Read-only loading of cached encoder tokens for the Member 1 model adapter.

Two explicit formats are supported; nothing is guessed from a file extension or from
the manifest's ``feature_path`` column:

``dimodif_npz``
    Member 2's current whole-video DiMoDif output::

        <feature_root>/<split>/<clip_id>/features.npz   (video_features, audio_features)

    The manifest's ``feature_path`` column still carries the *planned* ``.pt`` path for
    these rows and does NOT point at this file. The path is therefore built from an
    explicit ``feature_root`` by :func:`dimodif_npz_path`.

``shared_pt``
    The shared ``.pt`` contract read through Member 2's
    ``avfuse.member2_data.cache.load_feature_cache``.

The loader keeps the actual (possibly different) audio/video lengths. Alignment,
truncation, FP32 conversion and padding are the adapter's job
(``DiMoDifAdapter.prepare_features``), so they are deliberately not repeated here.
The loader never receives or returns labels.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from avfuse.member2_data.cache import load_feature_cache

FORMAT_DIMODIF_NPZ = "dimodif_npz"
FORMAT_SHARED_PT = "shared_pt"
FEATURE_FORMATS = (FORMAT_DIMODIF_NPZ, FORMAT_SHARED_PT)

FEATURE_DIM = 768
NPZ_FILENAME = "features.npz"
NPZ_VIDEO_KEY = "video_features"
NPZ_AUDIO_KEY = "audio_features"
SPLITS = ("val", "test")
_SAFE_CLIP_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class FeatureLoadError(ValueError):
    """A feature file exists but is not a valid DiMoDif/Member 1 input."""


@dataclass(frozen=True)
class ClipRef:
    """The only per-clip information the loader may see. Deliberately has no label."""

    clip_id: str
    split: str
    feature_path: str | None = None


def dimodif_npz_path(feature_root: str | Path, split: str, clip_id: str) -> Path:
    """``<feature_root>/<split>/<clip_id>/features.npz`` (never ``feature_path``)."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    if not isinstance(clip_id, str) or not _SAFE_CLIP_ID.match(clip_id):
        raise ValueError(f"Unsafe clip_id for a feature path: {clip_id!r}")
    return Path(feature_root) / split / clip_id / NPZ_FILENAME


def _validate_sequence(name: str, array: np.ndarray) -> None:
    if not isinstance(array, np.ndarray) or array.dtype.kind != "f":
        raise FeatureLoadError(
            f"{name} must be a floating-point array, got {getattr(array, 'dtype', type(array))}")
    if array.ndim != 2:
        raise FeatureLoadError(f"{name} must have rank 2 [T, {FEATURE_DIM}], got shape "
                               f"{tuple(array.shape)}")
    if array.shape[0] == 0:
        raise FeatureLoadError(f"{name} is an empty sequence")
    if array.shape[1] != FEATURE_DIM:
        raise FeatureLoadError(
            f"{name} feature dimension must be {FEATURE_DIM}, got {array.shape[1]}")
    if not np.isfinite(array).all():
        raise FeatureLoadError(f"{name} contains non-finite values")


def load_dimodif_features(path: str | Path) -> dict[str, Tensor]:
    """Load one video's ``features.npz`` as ``{"video_tokens", "audio_tokens"}`` (FP32).

    ``T_v`` and ``T_a`` are preserved as stored. The file is only read.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Missing DiMoDif feature file: {p}")
    try:
        with np.load(p, allow_pickle=False) as archive:
            missing = [k for k in (NPZ_VIDEO_KEY, NPZ_AUDIO_KEY) if k not in archive.files]
            if missing:
                raise FeatureLoadError(f"{p.name} is missing required key(s) {missing}; "
                                       f"found {sorted(archive.files)}")
            video = archive[NPZ_VIDEO_KEY]
            audio = archive[NPZ_AUDIO_KEY]
    except FeatureLoadError:
        raise
    except (ValueError, OSError, EOFError) as exc:  # includes pickled object arrays, bad zips
        raise FeatureLoadError(f"Cannot read {p.name} as a pickle-free NPZ: {exc}") from exc
    _validate_sequence(NPZ_VIDEO_KEY, video)
    _validate_sequence(NPZ_AUDIO_KEY, audio)
    return {
        "video_tokens": torch.from_numpy(np.array(video, dtype=np.float32, copy=True)),
        "audio_tokens": torch.from_numpy(np.array(audio, dtype=np.float32, copy=True)),
    }


def load_shared_pt_features(path: str | Path) -> dict[str, Tensor]:
    """Load the shared ``.pt`` contract through Member 2's public loader and validate it."""
    obj = load_feature_cache(path)
    out: dict[str, Tensor] = {}
    for key in ("video_tokens", "audio_tokens"):
        t = obj[key]
        if not t.is_floating_point():
            raise FeatureLoadError(f"{key} must be floating point, got {t.dtype}")
        if t.ndim not in (2, 3):
            raise FeatureLoadError(f"{key} must be [T,{FEATURE_DIM}] or [B,T,{FEATURE_DIM}], "
                                   f"got {tuple(t.shape)}")
        if t.shape[-1] != FEATURE_DIM:
            raise FeatureLoadError(f"{key} feature dimension must be {FEATURE_DIM}, "
                                   f"got {t.shape[-1]}")
        if t.shape[-2] == 0:
            raise FeatureLoadError(f"{key} is an empty sequence")
        if not torch.isfinite(t).all():
            raise FeatureLoadError(f"{key} contains non-finite values")
        out[key] = t.detach().to(dtype=torch.float32)
    if out["video_tokens"].ndim != out["audio_tokens"].ndim:
        raise FeatureLoadError("video_tokens and audio_tokens must have the same rank")
    if out["video_tokens"].ndim == 3 and \
            out["video_tokens"].shape[0] != out["audio_tokens"].shape[0]:
        raise FeatureLoadError("video_tokens and audio_tokens batch sizes differ")
    return out


def load_clip_features(
    clip: ClipRef,
    *,
    feature_format: str,
    feature_root: str | Path | None = None,
) -> dict[str, Tensor]:
    """Load one clip with an explicit format.

    ``dimodif_npz`` needs ``feature_root``; ``shared_pt`` needs ``clip.feature_path``
    (resolved against ``feature_root`` when that is given and the path is relative).
    """
    if feature_format not in FEATURE_FORMATS:
        raise ValueError(f"Unknown feature format {feature_format!r}; use one of "
                         f"{FEATURE_FORMATS}")
    if feature_format == FORMAT_DIMODIF_NPZ:
        if feature_root is None:
            raise ValueError("feature_format 'dimodif_npz' requires an explicit feature_root; "
                             "the manifest feature_path is not used for this format")
        return load_dimodif_features(dimodif_npz_path(feature_root, clip.split, clip.clip_id))
    if not clip.feature_path:
        raise ValueError(f"feature_format 'shared_pt' needs a feature_path for {clip.clip_id!r}")
    path = Path(clip.feature_path)
    if not path.is_absolute() and feature_root is not None:
        path = Path(feature_root) / path
    return load_shared_pt_features(path)


def make_feature_provider(feature_format: str, feature_root: str | Path | None = None):
    """Return ``provider(ClipRef) -> features`` bound to an explicit format/root."""
    if feature_format not in FEATURE_FORMATS:
        raise ValueError(f"Unknown feature format {feature_format!r}; use one of "
                         f"{FEATURE_FORMATS}")
    if feature_format == FORMAT_DIMODIF_NPZ and feature_root is None:
        raise ValueError("feature_format 'dimodif_npz' requires an explicit feature_root")

    def provider(clip: ClipRef) -> Mapping[str, Tensor]:
        return load_clip_features(clip, feature_format=feature_format, feature_root=feature_root)

    return provider
