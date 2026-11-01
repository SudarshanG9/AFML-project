"""Label taxonomy: dataset-specific flags -> the shared manipulation_type.

Shared values (contracts/schemas/manifest.schema.json):
    real, audio_fake, video_fake, both_fake
A clip is fake (label 1) if either stream is manipulated.
"""
from avfuse.shared.contracts import MANIPULATION_TYPES

FAKE_TYPES = tuple(t for t in MANIPULATION_TYPES if t != "real")

# (video_fake, audio_fake) -> manipulation_type
BY_FLAGS = {
    (0, 0): "real",
    (0, 1): "audio_fake",
    (1, 0): "video_fake",
    (1, 1): "both_fake",
}

# Folder/category names used by FakeAVCeleb and by the earlier Colab pipeline.
BY_CATEGORY = {
    "RealVideo-RealAudio": "real",
    "RealVideo-FakeAudio": "audio_fake",
    "FakeVideo-RealAudio": "video_fake",
    "FakeVideo-FakeAudio": "both_fake",
}


def manipulation_type(video_fake: int, audio_fake: int) -> str:
    return BY_FLAGS[(int(bool(video_fake)), int(bool(audio_fake)))]


def label_of(mtype: str) -> int:
    if mtype not in MANIPULATION_TYPES:
        raise ValueError(f"unknown manipulation_type {mtype!r}")
    return int(mtype != "real")
