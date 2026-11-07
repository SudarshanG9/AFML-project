"""Synthetic integration smoke CLI, not a deepfake detection demonstration."""
import argparse

import torch

from avfuse.member1_fusion.diagnostic_model import DiagnosticModel

from .benchmark import benchmark_callable
from .runner import run_four_modes


def main() -> None:
    parser = argparse.ArgumentParser(description="AVFuse-CausalQ diagnostic (no real model)")
    parser.add_argument("--smoke", action="store_true", help="Use synthetic tokens and random head")
    args = parser.parse_args()
    if not args.smoke:
        parser.error("Real model not yet integrated. Use --smoke for diagnostic only.")
    torch.manual_seed(7)
    device = torch.device("cpu")
    model = DiagnosticModel(16).to(device).eval()
    features = {"audio_tokens": torch.randn(1, 12, 16, device=device),
                "video_tokens": torch.randn(1, 8, 16, device=device)}
    print("DIAGNOSTIC ONLY: synthetic tokens + randomly initialized toy model")
    for mode, score in run_four_modes(model, features).items():
        timing = benchmark_callable(
            lambda mode=mode: model.predict_from_cached_features(features, mode),
            device,
            warmup=2,
            repeats=5,
        )
    print(f"{mode:22s} fake-score={float(score.item()):.4f}  p50={timing['p50_ms']:.3f} ms")
    print("No forensic accuracy or INT8 speedup is claimed by this diagnostic.")


if __name__ == "__main__":
    main()
