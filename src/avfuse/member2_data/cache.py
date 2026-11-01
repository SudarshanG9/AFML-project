"""Load *frozen encoder tokens*, not raw videos or post-fusion tensors."""
from pathlib import Path
import torch
from torch import Tensor


def load_feature_cache(path: str | Path) -> dict[str, Tensor]:
    # weights_only limits untrusted Python deserialization (still validate source).
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(obj, dict):
        raise TypeError("Cached features must be tensor dictionaries")
    for field in ("audio_tokens", "video_tokens"):
        if field not in obj or not isinstance(obj[field], Tensor):
            raise ValueError(f"Cache missing required tensor: {field}")
    return obj


def save_feature_cache(path: str | Path, features: dict[str, Tensor]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    for name in ("audio_tokens", "video_tokens"):
        if name not in features or not isinstance(features[name], Tensor):
            raise ValueError(f"Expected tensor field {name}")
    torch.save({name: val.detach().cpu() for name, val in features.items()}, target)
