r"""Check pretrained DiMoDif on one cached clip, or run validation inference.

From the repository root:

    python tests/member1/day6_integration_test.py
    python tests/member1/day6_integration_test.py --feature path/to/features.npz
    python tests/member1/day6_integration_test.py --manifest path/to/manifest.jsonl \
        --cache-root path/to/cache --mode reference \
        --output artifacts/predictions/predictions_ref_val.jsonl

Validation uses the extractor layout <cache-root>/val/<clip_id>/features.npz.
"""

import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from avfuse.member1_fusion.dimodif_adapter import DiMoDifAdapter
from avfuse.member2_data.manifest import read_manifest
from avfuse.shared.contracts import PRECISION_MODES, PrecisionMode
from avfuse.shared.jsonl import write_jsonl

PARITY_TOLERANCE = {"atol": 1e-5, "rtol": 1e-4}
TENSORS = ("Q", "K", "V", "attention_output")
QUANTIZED_TENSORS = {
    "reference": set(),
    "low_precision": set(TENSORS),
    "restore_value_output": {"Q", "K"},
    "restore_query": {"K", "V", "attention_output"},
}


def load_npz(path: Path) -> dict[str, torch.Tensor]:
    with np.load(path, allow_pickle=False) as features:
        return {
            "video_tokens": torch.from_numpy(
                np.asarray(features["video_features"], dtype=np.float32)
            ),
            "audio_tokens": torch.from_numpy(
                np.asarray(features["audio_features"], dtype=np.float32)
            ),
        }


def tensor_stats(baseline: torch.Tensor, actual: torch.Tensor) -> dict:
    if baseline.shape != actual.shape or baseline.dtype != actual.dtype:
        raise RuntimeError("Unexpected output shape or dtype")
    if not torch.isfinite(actual).all():
        raise RuntimeError("Model output contains NaN or Inf")
    diff = (baseline - actual).abs()
    return {
        "max_abs_error": diff.max().item(),
        "mean_abs_error": diff.mean().item(),
        "bit_exact": torch.equal(baseline, actual),
        "a_range": [baseline.min().item(), baseline.max().item()],
        "b_range": [actual.min().item(), actual.max().item()],
    }


def fake_score(scores: torch.Tensor) -> float:
    if scores.shape != (1,):
        raise RuntimeError(f"Expected one fake score, got shape {tuple(scores.shape)}")
    score = scores.item()
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise RuntimeError(f"Invalid fake probability: {score}")
    return score


def check_trace(trace: list[dict], mode: PrecisionMode, config: dict) -> None:
    expected = {
        (layer, tensor) for layer in range(config["nlayers"]) for tensor in TENSORS
    }
    observed = {(record["layer"], record["tensor"]) for record in trace}
    if len(trace) != len(expected) or observed != expected:
        raise RuntimeError(f"{mode}: expected one Q/K/V/output record per attention layer")

    heads = config["nhead"]
    expected_shape = [1, heads, 2 * (config["max_length"] + 1), config["d_model"] // heads]
    for record in trace:
        tensor = record["tensor"]
        layer = record["layer"]
        if record["mode"] != mode or record["shape"] != expected_shape:
            raise RuntimeError(f"{mode} layer {layer} {tensor}: unexpected trace record")
        changed = record["fraction_changed"] > 0
        if changed != (tensor in QUANTIZED_TENSORS[mode]):
            raise RuntimeError(f"{mode} layer {layer} {tensor}: unexpected quantization")


def check_integration(adapter: DiMoDifAdapter, feature_path: Path) -> dict:
    features = load_npz(feature_path)
    attention_outputs = []

    def capture_output(_module, _inputs, output):
        attention_outputs.append(output[0].detach().clone())

    layers = adapter.model.transformer_encoder.layers
    reference_trace = []
    with ExitStack() as hooks:
        for layer in layers:
            hooks.callback(layer.self_attn.register_forward_hook(capture_output).remove)
        baseline = adapter.forward_reference(features)
        native_attention = attention_outputs.copy()
        attention_outputs.clear()
        reference = adapter.forward_with_mode(features, mode="reference", trace=reference_trace)

    if len(native_attention) != len(attention_outputs) or len(native_attention) != len(layers):
        raise RuntimeError("Expected a native and instrumented output from every attention layer")

    attention_parity = []
    for layer, (native, wrapped) in enumerate(zip(native_attention, attention_outputs)):
        torch.testing.assert_close(wrapped, native, **PARITY_TOLERANCE, msg=f"Layer {layer} parity")
        attention_parity.append({"layer": layer, **tensor_stats(native, wrapped)})

    base_logits = baseline["logits"]
    base_score = fake_score(baseline["score_fake"])
    torch.testing.assert_close(reference["logits"], base_logits, **PARITY_TOLERANCE)
    torch.testing.assert_close(reference["score_fake"], baseline["score_fake"], **PARITY_TOLERANCE)

    print(f"Native score={base_score:.10f}, logits={tuple(base_logits.shape)}")
    print(f"FP32 attention parity passed for {len(attention_parity)} layers")
    mode_results = {}
    for mode in PRECISION_MODES:
        if mode == "reference":
            result, trace = reference, reference_trace
        else:
            trace = []
            result = adapter.forward_with_mode(features, mode=mode, trace=trace)

        check_trace(trace, mode, adapter.config)
        score = fake_score(result["score_fake"])
        logit_stats = tensor_stats(base_logits, result["logits"])
        if result["valid_length"] != baseline["valid_length"]:
            raise RuntimeError(f"{mode}: input length changed")
        if mode != "reference" and logit_stats["bit_exact"]:
            raise RuntimeError(f"{mode}: expected a logit change on this sample")
        mode_results[mode] = {
            "score_fake": score,
            "score_delta": score - base_score,
            "logit_vs_baseline": logit_stats,
            "trace_records": len(trace),
            "per_layer_trace": trace,
        }
        print(
            f"{mode:<22} score={score:.10f}  "
            f"delta_score={score - base_score:+.3e}  "
            f"max_abs_logit_delta={logit_stats['max_abs_error']:.6g}  "
            f"trace_records={len(trace)}"
        )

    return {
        "step": 6,
        "description": "Original vs. instrumented DiMoDif on a real cached sample",
        "feature_file": str(feature_path),
        "checkpoint_sha256": adapter.checkpoint_sha256,
        "device": str(adapter.device),
        "torch_version": torch.__version__,
        "valid_length": baseline["valid_length"],
        "max_length": adapter.max_length,
        "config": {
            key: adapter.config[key]
            for key in ("nlayers", "nhead", "d_model", "feature_pyramid")
        },
        "baseline": {"score_fake": base_score, "logits_shape": list(base_logits.shape)},
        "reference_attention_parity": attention_parity,
        "reference_parity_tolerance": PARITY_TOLERANCE,
        "precision_modes": mode_results,
        "passed": True,
    }


def validation_cache(manifest: Path, cache_root: Path) -> list[tuple[str, Path]]:
    clips = []
    for row in read_manifest(manifest):
        if row["split"] == "val":
            clip_id = row["clip_id"]
            clips.append((clip_id, cache_root / "val" / clip_id / "features.npz"))

    if not clips:
        raise ValueError(f"No validation clips in {manifest}")
    missing = [clip_id for clip_id, path in clips if not path.is_file()]
    if missing:
        examples = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"Missing {len(missing)}/{len(clips)} validation caches under {cache_root} "
            f"(first missing clip IDs: {examples})"
        )
    return clips


def run_validation(
    adapter: DiMoDifAdapter,
    clips: list[tuple[str, Path]],
    mode: PrecisionMode,
    run_id: str,
    output: Path,
) -> None:
    rows = []
    for clip_id, path in clips:
        score = fake_score(adapter.predict_from_cached_features(load_npz(path), mode))
        rows.append({
            "run_id": run_id,
            "clip_id": clip_id,
            "split": "val",
            "config": mode,
            "score_fake": score,
            "checkpoint_sha256": adapter.checkpoint_sha256,
        })
    write_jsonl(output, rows)
    print(f"Wrote {len(rows)} {mode} predictions to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--feature", type=Path, help="One cached NPZ for the integration check")
    source.add_argument("--manifest", type=Path, help="Manifest JSONL for validation inference")
    parser.add_argument("--cache-root", type=Path, help="Root of the extracted validation NPZs")
    parser.add_argument("--mode", choices=PRECISION_MODES, help="Validation mode (default: reference)")
    parser.add_argument("--run-id", help="Prediction run identifier for validation")
    parser.add_argument("--upstream", type=Path, default=REPO / "third_party" / "dimodif")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, help="Integration JSON or validation JSONL path")
    args = parser.parse_args()

    if args.manifest is not None:
        if args.cache_root is None or args.output is None:
            parser.error("--manifest requires --cache-root and --output")
        try:
            clips = validation_cache(args.manifest, args.cache_root)
        except FileNotFoundError as error:
            parser.error(str(error))
    else:
        if args.cache_root is not None or args.mode is not None or args.run_id is not None:
            parser.error("--cache-root, --mode and --run-id require --manifest")
        feature = args.feature or REPO / "data/pilot/day2_features/features1.npz"
        if not feature.is_file():
            parser.error(f"Feature file not found: {feature}")

    adapter = DiMoDifAdapter(upstream_root=args.upstream, checkpoint_name="lavdf", device=args.device)
    print(f"LAV-DF checkpoint SHA-256: {adapter.checkpoint_sha256}")

    if args.manifest is not None:
        mode = args.mode or "reference"
        run_validation(adapter, clips, mode, args.run_id or f"dayB-{mode}-val", args.output)
    else:
        output = args.output or REPO / "artifacts/day6/integration_report.json"
        report = check_integration(adapter, feature)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Step 6 passed on {feature}. Report saved to {output}")


if __name__ == "__main__":
    main()
