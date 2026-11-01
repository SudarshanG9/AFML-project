"""Actual detector/checkpoint goes behind this stable interface on Day 1."""
from abc import ABC, abstractmethod
from collections.abc import Mapping
from torch import Tensor
from avfuse.shared.contracts import PrecisionMode


class ModelAdapter(ABC):
    @abstractmethod
    def predict_from_cached_features(self, features: Mapping[str, Tensor], mode: PrecisionMode) -> Tensor:
        """Return fake probability tensor of shape (batch,) for frozen detector.

        Real implementations must preserve the checkpoint's preprocessing and
        model's original fusion computation outside the selected intervention.
        """
        raise NotImplementedError
