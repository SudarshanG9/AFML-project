import zlib

import numpy as np

from avfuse.member2_data.shortcuts import clip_features, shortcut_audit, silence_profile


def test_silence_profile():
    a = np.r_[np.zeros(8000), np.full(8000, 0.5), np.zeros(1600)].astype(np.float32)
    p = silence_profile(a)
    assert p["leading_silence_ms"] == 500 and p["trailing_silence_ms"] == 100


def _setup(tmp_path, silence_for):
    manifest, feats = [], {}
    for i in range(40):
        fake = i % 2
        a = (np.random.default_rng(zlib.crc32(str(i).encode())).standard_normal(64000) * 0.1).astype(np.float32)
        a[: int(silence_for(i, fake) * 16000)] = 0
        path = tmp_path / f"{i}.npy"
        np.save(path, a.astype(np.float16))
        cid = f"c{i}"
        manifest.append({"clip_id": cid, "split": "val", "label": fake,
                         "manipulation_type": "audio_fake" if fake else "real"})
        feats[cid] = clip_features({"clip_id": cid, "n_frames": 100, "n_samples": 64000, "audio": str(path),
                                    "start_seconds": 0.0, "fake_start": 2.0 if fake else None,
                                    "fake_end": 2.4 if fake else None})
    return shortcut_audit(manifest, feats)


def test_flags_injected_silence_shortcut(tmp_path):
    rep = _setup(tmp_path, lambda i, fake: 0.0 if fake else 0.6)  # reals open with silence
    assert any("/all: leading_silence_ms" in f for f in rep["flags"]) and not rep["passed"]


def test_quiet_without_shortcut(tmp_path):
    rep = _setup(tmp_path, lambda i, fake: 0.3 * ((i // 2) % 2))   # silence unrelated to label
    assert not any("silence" in f for f in rep["flags"])
