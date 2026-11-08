
"""Day 2: DiMoDif modality ablation on multiple NPZ files."""

import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from avfuse.member1_fusion.dimodif_adapter import DiMoDifAdapter


def load_features(path):
    with np.load(path, allow_pickle=False) as data:
        video = np.asarray(
            data["video_features"], dtype=np.float32
        )
        audio = np.asarray(
            data["audio_features"], dtype=np.float32
        )

    for name, array in [("video", video), ("audio", audio)]:
        if array.ndim != 2 or array.shape[1] != 768:
            raise ValueError(f"Invalid {name} shape: {array.shape}")
        if array.shape[0] == 0 or not np.isfinite(array).all():
            raise ValueError(f"Invalid {name} features")

    return {
        "video_tokens": torch.from_numpy(video.copy()),
        "audio_tokens": torch.from_numpy(audio.copy()),
    }


def serialize_prediction(result):
    return {
        "fake_score": float(
            result["score_fake"].reshape(-1)[0].item()
        ),
        "video_score": float(
            result["modality_scores"][0, 0].item()
        ),
        "audio_score": float(
            result["modality_scores"][0, 1].item()
        ),
        "raw_logits": (
            result["logits"][0].detach().cpu().tolist()
        ),
    }


@torch.inference_mode()
def evaluate_clip(adapter, features):
    # Original prediction
    ref = adapter.forward_reference(features)

    # Zero audio only
    audio_removed = {
        **features,
        "audio_tokens": torch.zeros_like(
            features["audio_tokens"]
        ),
    }
    za = adapter.forward_reference(audio_removed)

    # Zero video only
    video_removed = {
        **features,
        "video_tokens": torch.zeros_like(
            features["video_tokens"]
        ),
    }
    zv = adapter.forward_reference(video_removed)

    return {
        "reference": serialize_prediction(ref),
        "zero_audio": serialize_prediction(za),
        "zero_video": serialize_prediction(zv),
        "delta_audio_logit": float(
            (za["logits"] - ref["logits"]).abs().max().item()
        ),
        "delta_video_logit": float(
            (zv["logits"] - ref["logits"]).abs().max().item()
        ),
        "delta_audio_output": float(
            (
                za["modality_scores"][0, 1]
                - ref["modality_scores"][0, 1]
            ).abs().item()
        ),
        "delta_video_output": float(
            (
                zv["modality_scores"][0, 0]
                - ref["modality_scores"][0, 0]
            ).abs().item()
        ),
        "valid_length": int(ref["valid_length"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", default="data/pilot/day2_features"
    )
    parser.add_argument(
        "--upstream", default="third_party/dimodif"
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--output",
        default="artifacts/day2/ablation_report.json",
    )
    args = parser.parse_args()

    root = Path(args.data_dir)

    if not root.is_dir():
        # Check common fallback locations (e.g. data/pilot/day2_features or relative to repo root)
        candidates = [
            Path("data/pilot") / root.name,
            Path("data/pilot") / root,
            Path(__file__).resolve().parents[1] / args.data_dir,
            Path(__file__).resolve().parents[1] / "data" / "pilot" / root.name,
            Path(__file__).resolve().parents[1] / "data" / "pilot" / "day2_features",
        ]
        for candidate in candidates:
            if candidate.is_dir():
                root = candidate
                break
        else:
            raise FileNotFoundError(f"Missing data folder: {root}")

    files = sorted(root.rglob("features*.npz"))

    if not files:
        raise RuntimeError("No features*.npz files found")

    print(f"Found {len(files)} feature files")

    adapter = DiMoDifAdapter(
        upstream_root=args.upstream,
        checkpoint_name="lavdf",
        device=args.device,
    )

    results = []

    for index, path in enumerate(files, 1):
        print(f"\n[{index}/{len(files)}] {path.name}")

        features = load_features(path)
        metrics = evaluate_clip(adapter, features)

        metrics["file"] = str(path)
        metrics["clip_id"] = path.relative_to(root).as_posix()

        results.append(metrics)

        print(
            "Reference:",
            metrics["reference"]["fake_score"]
        )
        print(
            "Zero audio:",
            metrics["zero_audio"]["fake_score"]
        )
        print(
            "Zero video:",
            metrics["zero_video"]["fake_score"]
        )
        print(
            "Audio logit change:",
            metrics["delta_audio_logit"]
        )
        print(
            "Video logit change:",
            metrics["delta_video_logit"]
        )

    summary = {
        "total_clips": len(results),
        "mean_audio_logit_change": mean(
            r["delta_audio_logit"] for r in results
        ),
        "mean_video_logit_change": mean(
            r["delta_video_logit"] for r in results
        ),
        "mean_audio_output_change": mean(
            r["delta_audio_output"] for r in results
        ),
        "mean_video_output_change": mean(
            r["delta_video_output"] for r in results
        ),
    }

    report = {
        "checkpoint_sha256": adapter.checkpoint_sha256,
        "device": str(adapter.device),
        "summary": summary,
        "results": results,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n========== DAY 2 SUMMARY ==========")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print(f"\nReport saved to: {output}")


if __name__ == "__main__":
    main()
