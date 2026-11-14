
"""Concrete adapter for the original pretrained DiMoDif detector.

Day 1: reference inference only.
No attention or model weights are replaced.
"""

from collections.abc import Mapping
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Literal
import hashlib
import json

import torch
import torch.nn.functional as F
from torch import Tensor

from avfuse.shared.contracts import (
    PrecisionMode,
    PRECISION_MODES,
)

from .dimodif_intervention import (
    intervene_dimodif_attention,
)
from .model_adapter import ModelAdapter


CheckpointName = Literal["lavdf", "avdeepfake1m"]

CHECKPOINTS = {
    "lavdf": (
        "dfd_lavdf_reduceonplateau_64_4_1_5_True"
    ),
    "avdeepfake1m": (
        "dfd_avdeepfake1m_reduceonplateau_64_4_1_5_True_whole"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_upstream_model_class(upstream_root: Path):
    """Import the original model without copying its implementation."""

    source_path = upstream_root / "src" / "models.py"

    if not source_path.is_file():
        raise FileNotFoundError(
            f"Missing original DiMoDif model: {source_path}"
        )

    spec = spec_from_file_location(
        "_avfuse_dimodif_original_models",
        source_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError("Cannot load DiMoDif model source")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.AVDeepFakeDetector


class DiMoDifAdapter(ModelAdapter):
    """Pretrained, frozen DiMoDif deepfake detector."""

    def __init__(
        self,
        upstream_root: str | Path,
        checkpoint_name: CheckpointName = "lavdf",
        device: str = "cpu",
    ):
        self.upstream_root = Path(upstream_root).resolve()
        self.device = torch.device(device)
        self.checkpoint_name = checkpoint_name

        if checkpoint_name not in CHECKPOINTS:
            raise ValueError(
                f"Unknown checkpoint: {checkpoint_name}"
            )

        stem = CHECKPOINTS[checkpoint_name]

        checkpoint_dir = (
            self.upstream_root / "ckpt" / "dfd"
        )

        self.checkpoint_path = (
            checkpoint_dir / f"{stem}.pth"
        )

        self.config_path = (
            checkpoint_dir / f"{stem}.json"
        )

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Missing checkpoint: {self.checkpoint_path}"
            )

        if not self.config_path.is_file():
            raise FileNotFoundError(
                f"Missing model config: {self.config_path}"
            )

        with self.config_path.open() as stream:
            self.config = json.load(stream)["config"]

        self.checkpoint_sha256 = sha256_file(
            self.checkpoint_path
        )

        ModelClass = load_upstream_model_class(
            self.upstream_root
        )

        config = self.config

        # Reconstruct the original architecture.
        self.model = ModelClass(
            task=config["task"],
            max_length=config["max_length"],
            d_model=config["d_model"],
            nhead=config["nhead"],
            d_hid=config["d_hid"],
            nlayers=config["nlayers"],
            win_size=config["win_size"],
            feature_pyramid=config["feature_pyramid"],
            device=str(self.device),
        )

        # Load trusted original weights. Never partially load.
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

        if "model" not in checkpoint:
            raise KeyError(
                "Checkpoint missing 'model' state dictionary"
            )

        self.model.load_state_dict(
            checkpoint["model"],
            strict=True,
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        # The upstream evaluation script explicitly
        # disables MHA fastpath. Its feature pyramid
        # depends on hooks at intermediate norm layers.
        torch.backends.mha.set_fastpath_enabled(False)

        self.max_length = config["max_length"]

        if config["task"] != "dfd":
            raise ValueError(
                "This adapter supports DFD checkpoints only"
            )

    def prepare_features(
        self,
        features: Mapping[str, Tensor],
    ) -> tuple[Tensor, Tensor, int]:
        """Reproduce upstream alignment, length and padding."""

        required = ("video_tokens", "audio_tokens")

        for key in required:
            if key not in features:
                raise KeyError(f"Missing cache key: {key}")

        video = features["video_tokens"]
        audio = features["audio_tokens"]

        if not isinstance(video, Tensor):
            raise TypeError("video_tokens must be a Tensor")

        if not isinstance(audio, Tensor):
            raise TypeError("audio_tokens must be a Tensor")

        # Accept [T, 768] for one clip.
        # Also accept [B, T, 768] for batch inference.
        if video.ndim == 2:
            video = video.unsqueeze(0)

        if audio.ndim == 2:
            audio = audio.unsqueeze(0)

        if video.ndim != 3 or audio.ndim != 3:
            raise ValueError(
                "Expected [B,T,768] or [T,768] tensors"
            )

        if video.shape[0] != audio.shape[0]:
            raise ValueError(
                "Audio/video batch sizes must match"
            )

        if video.shape[-1] != 768:
            raise ValueError(
                "Video feature dimension must be 768"
            )

        if audio.shape[-1] != 768:
            raise ValueError(
                "Audio feature dimension must be 768"
            )

        if video.shape[1] == 0 or audio.shape[1] == 0:
            raise ValueError(
                "Empty audio/video feature sequence"
            )

        if not torch.isfinite(video).all():
            raise ValueError("Invalid video features")

        if not torch.isfinite(audio).all():
            raise ValueError("Invalid audio features")

        # Original upstream loader aligns both streams
        # using their minimum available sequence length.
        valid_length = min(
            video.shape[1],
            audio.shape[1],
            self.max_length,
        )

        video = video[:, :valid_length, :]
        audio = audio[:, :valid_length, :]

        # Original DiMoDif runs in floating point.
        video = video.to(
            device=self.device,
            dtype=torch.float32,
        )

        audio = audio.to(
            device=self.device,
            dtype=torch.float32,
        )

        pad_length = self.max_length - valid_length

        video = F.pad(
            video,
            (0, 0, 0, pad_length),
        )

        audio = F.pad(
            audio,
            (0, 0, 0, pad_length),
        )

        return video, audio, valid_length

    @torch.inference_mode()
    def forward_reference(
        self,
        features: Mapping[str, Tensor],
    ) -> dict[str, Tensor]:
        """Unmodified original detector forward pass."""

        video, audio, valid_length = (
            self.prepare_features(features)
        )

        logits, hidden = self.model(
            [video, audio]
        )

        if not torch.isfinite(logits).all():
            raise RuntimeError(
                "Model produced non-finite logits"
            )

        # Match original DiMoDif DFD evaluation:
        # sigmoid -> average feature pyramid -> max.
        probabilities = torch.sigmoid(logits)

        if self.config["feature_pyramid"]:
            probabilities = probabilities.mean(dim=1)

        fake_score = probabilities.amax(dim=-1)

        return {
            "logits": logits,
            "hidden": hidden,
            "modality_scores": probabilities,
            "score_fake": fake_score,
            "valid_length": valid_length,
        }

    @torch.inference_mode()
    def forward_with_mode(
        self,
        features: Mapping[str, Tensor],
        mode: PrecisionMode,
        trace: list[dict] | None = None,
    ) -> dict[str, Tensor]:
        """Run frozen DiMoDif with an attention intervention."""

        if mode not in PRECISION_MODES:
            raise ValueError(
                f"Unsupported precision mode: {mode}"
            )

        video, audio, valid_length = (
            self.prepare_features(features)
        )

        # All five existing MHA modules are temporarily
        # instrumented with the selected precision mode.
        with intervene_dimodif_attention(
            self.model,
            mode=mode,
            trace=trace,
        ):
            logits, hidden = self.model([
                video,
                audio,
            ])

        if not torch.isfinite(logits).all():
            raise RuntimeError(
                "Non-finite logits in precision run"
            )

        # Preserve the exact original DiMoDif scoring.
        probabilities = torch.sigmoid(logits)

        if self.config["feature_pyramid"]:
            probabilities = probabilities.mean(dim=1)

        fake_score = probabilities.amax(dim=-1)

        if not torch.isfinite(fake_score).all():
            raise RuntimeError(
                "Non-finite fake probability"
            )

        return {
            "logits": logits,
            "hidden": hidden,
            "modality_scores": probabilities,
            "score_fake": fake_score,
            "valid_length": valid_length,
        }

    
    @torch.inference_mode()
    def predict_from_cached_features(
        self,
        features: Mapping[str, Tensor],
        mode: PrecisionMode,
    ) -> Tensor:
        """Return one fake probability per batch item."""

        if mode not in PRECISION_MODES:
            raise ValueError(
                f"Unsupported precision mode: {mode}"
            )

        if mode == "reference":
            # Keep original model as authoritative baseline.
            result = self.forward_reference(features)

        else:
            result = self.forward_with_mode(
                features,
                mode=mode,
            )

        scores = result["score_fake"]

        if scores.ndim != 1:
            raise RuntimeError(
                "Expected one fake score per batch item"
            )

        return scores

