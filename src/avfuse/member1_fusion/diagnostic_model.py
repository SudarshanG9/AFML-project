"""UNTRAINED diagnostic model: synthetic infrastructure smoke tests only."""
import torch
from torch import Tensor, nn
from collections.abc import Mapping
from avfuse.shared.contracts import PrecisionMode
from .attention import fusion_attention
from .model_adapter import ModelAdapter


class DiagnosticModel(ModelAdapter, nn.Module):
    """Never use for reported deepfake accuracy. Deliberately randomly initialized."""

    def __init__(self, dimension: int = 16):
        super().__init__()
        self.head = nn.Linear(dimension, 1)

    def predict_from_cached_features(self, features: Mapping[str, Tensor], mode: PrecisionMode) -> Tensor:
        audio = features["audio_tokens"]  # [B, T_audio, D]
        video = features["video_tokens"]  # [B, T_video, D]
        q = video.unsqueeze(1)
        k = audio.unsqueeze(1)
        v = audio.unsqueeze(1)
        fused = fusion_attention(q, k, v, mode=mode).mean(dim=(1, 2))
        return torch.sigmoid(self.head(fused)).squeeze(-1)
