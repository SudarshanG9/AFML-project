
"""Day B: frozen DiMoDif precision experiments.

Pilot:
    python scripts/dayB_inference.py \
        --features PATH_TO_FEATURES_NPZ \
        --mode reference --device cpu

Validation:
    python scripts/dayB_inference.py \
        --manifest data/manifests/manifest.jsonl \
        --mode low_precision --device cuda:0 \
        --out artifacts/predictions/low_precision_val.jsonl

No test-set inference is permitted by this Day B runner.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from avfuse.member1_fusion.dimodif_adapter import (
    DiMoDifAdapter,
)

from avfuse.member2_data.cache import (
    load_feature_cache,
)

from avfuse.member2_data.manifest import (
    read_manifest,
)

from avfuse.member2_data.predictions import (
    validate_predictions,
)

from avfuse.member2_data.splits import (
    split_audit,
)

from avfuse.shared.contracts import (
    PRECISION_MODES,
)

from avfuse.shared.jsonl import (
    write_jsonl,
)


def load_features(path: Path):
    """Load existing pretrained encoder outputs."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing cached features: {path}"
        )

    if path.suffix == ".pt":
        return load_feature_cache(path)

    if path.suffix == ".npz":
        with np.load(
            path,
            allow_pickle=False,
        ) as data:

            video = np.array(
                data["video_features"],
                dtype=np.float32,
                copy=True,
            )

            audio = np.array(
                data["audio_features"],
                dtype=np.float32,
                copy=True,
            )

        return {
            "video_tokens": torch.from_numpy(video),
            "audio_tokens": torch.from_numpy(audio),
        }

    raise ValueError(
        "Expected .pt or .npz cached features"
    )


def create_adapter(args):
    adapter = DiMoDifAdapter(
        upstream_root=args.upstream,
        checkpoint_name="lavdf",
        device=args.device,
    )

    expected_sha = (
        "e1e59006d1bb6ad36a3859af5486a6649ed41bba23258b7c41e24f873be7948e"
    )

    if adapter.checkpoint_sha256 != expected_sha:
        raise RuntimeError(
            "Checkpoint mismatch. Refusing to run "
            "a precision experiment on unapproved weights."
        )

    if adapter.model.training:
        raise RuntimeError(
            "Pretrained detector must be in eval mode."
        )

    if any(
        p.requires_grad
        for p in adapter.model.parameters()
    ):
        raise RuntimeError(
            "Pretrained parameters must remain frozen."
        )

    return adapter

@torch.inference_mode()
def run_pilot(adapter, args):
    """Compare instrumented inference with original."""

    features = load_features(
        Path(args.features)
    )

    original = adapter.forward_reference(features)

    trace = []

    intervention = adapter.forward_with_mode(
        features,
        mode=args.mode,
        trace=trace,
    )

    original_score = float(
        original["score_fake"].reshape(-1)[0].item()
    )

    modified_score = float(
        intervention["score_fake"].reshape(-1)[0].item()
    )

    # This must pass before any low-precision experiment
    # is scientifically meaningful.
    if args.mode == "reference":

        torch.testing.assert_close(
            original["logits"],
            intervention["logits"],
            atol=1e-5,
            rtol=1e-4,
        )

        torch.testing.assert_close(
            original["score_fake"],
            intervention["score_fake"],
            atol=1e-6,
            rtol=1e-5,
        )

        print("FULL-MODEL FP32 PARITY: PASS")

    n_layers = len(
        adapter.model.transformer_encoder.layers
    )

    if len(trace) != n_layers * 4:
        raise RuntimeError(
            "Not all attention layers were instrumented"
        )

    print("Checkpoint:", adapter.checkpoint_sha256)
    print("Mode:", args.mode)
    print("Instrumented layers:", n_layers)
    print("Original score:", original_score)
    print("Modified score:", modified_score)
    print(
        "Score difference:",
        modified_score - original_score,
    )

    if args.trace_out:
        output = Path(args.trace_out)
        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        report = {
            "mode": args.mode,
            "checkpoint_sha256":
                adapter.checkpoint_sha256,
            "original_score": original_score,
            "modified_score": modified_score,
            "valid_length":
                intervention["valid_length"],
            "trace": trace,
        }

        output.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

        print("Diagnostic trace:", output)


@torch.inference_mode()
def run_validation(adapter, args):
    """Generate one prediction per validation clip."""

    manifest = read_manifest(
        args.manifest
    )

    audit = split_audit(manifest)

    if not audit["passed"]:
        raise RuntimeError(
            "Manifest has cross-split group leakage"
        )

    rows = sorted(
        (
            row for row in manifest
            if row["split"] == "val"
        ),
        key=lambda row: row["clip_id"],
    )

    if not rows:
        raise RuntimeError(
            "No validation clips found"
        )

    output_rows = []

    for index, row in enumerate(rows, start=1):

        feature_path = Path(
            row["feature_path"]
        )

        if not feature_path.is_absolute():
            feature_path = (
                Path(args.repo_root) / feature_path
            )

        features = load_features(feature_path)

        scores = (
            adapter.predict_from_cached_features(
                features,
                mode=args.mode,
            )
        )

        if scores.numel() != 1:
            raise RuntimeError(
                "Expected one score for one clip"
            )

        score = float(
            scores.reshape(-1)[0].item()
        )

        if not math.isfinite(score):
            raise RuntimeError(
                f"Non-finite score: {row['clip_id']}"
            )

        if not 0.0 <= score <= 1.0:
            raise RuntimeError(
                f"Probability outside [0,1]: {score}"
            )

        output_rows.append({
            "run_id": (
                args.run_id
                or f"dayB-{args.mode}-val"
            ),
            "clip_id": row["clip_id"],
            "split": "val",
            "config": args.mode,
            "score_fake": score,
            "checkpoint_sha256":
                adapter.checkpoint_sha256,
        })

        if index % 25 == 0 or index == len(rows):
            print(
                f"{args.mode}: {index}/{len(rows)}"
            )

    # Enforce the existing Member 2 prediction schema.
    validate_predictions(output_rows)

    predicted_ids = {
        row["clip_id"] for row in output_rows
    }

    expected_ids = {
        row["clip_id"] for row in rows
    }

    if predicted_ids != expected_ids:
        raise RuntimeError(
            "Incomplete validation predictions"
        )

    write_jsonl(
        args.out,
        output_rows,
    )

    print(
        f"Saved {len(output_rows)} predictions "
        f"to {args.out}"
    )


def main():
    parser = argparse.ArgumentParser()

    source = parser.add_mutually_exclusive_group(
        required=True
    )

    source.add_argument(
        "--features",
        help="One cached .npz or .pt feature file",
    )

    source.add_argument(
        "--manifest",
        help="Audited manifest (validation only)",
    )

    parser.add_argument(
        "--mode",
        choices=PRECISION_MODES,
        required=True,
    )

    parser.add_argument(
        "--upstream",
        default="third_party/dimodif",
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--repo-root",
        default=".",
    )

    parser.add_argument("--out")
    parser.add_argument("--trace-out")
    parser.add_argument("--run-id")

    args = parser.parse_args()
    if args.manifest and Path(args.out).exists():
        raise FileExistsError(
            f"Prediction output already exists: {args.out}. "
            "Use a new output name for a new run."
        )
    if args.manifest and not args.out:
        parser.error(
            "--out is required for manifest inference"
        )

    if args.manifest and args.trace_out:
        parser.error(
            "Use --trace-out only for pilot inference"
        )

    torch.manual_seed(42)

    adapter = create_adapter(args)

    if args.features:
        run_pilot(adapter, args)
    else:
        run_validation(adapter, args)


if __name__ == "__main__":
    main()


