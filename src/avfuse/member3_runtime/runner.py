"""Stable interface wiring Member 2 features to Member 1 model."""
from collections.abc import Mapping
import torch
from torch import Tensor
from avfuse.shared.contracts import PRECISION_MODES
from avfuse.member1_fusion.model_adapter import ModelAdapter

@torch.inference_mode()
def run_four_modes(model: ModelAdapter, features: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Return scores. Actual predictions JSONL and label custody handled elsewhere."""
    return {mode: model.predict_from_cached_features(features, mode).detach().clone()
            for mode in PRECISION_MODES}
