"""Stable enum-like constants shared across members."""
from typing import Literal

PrecisionMode = Literal["reference", "low_precision", "restore_value_output", "restore_query"]
PRECISION_MODES: tuple[PrecisionMode, ...] = (
    "reference", "low_precision", "restore_value_output", "restore_query"
)
MANIPULATION_TYPES = ("real", "audio_fake", "video_fake", "both_fake")
