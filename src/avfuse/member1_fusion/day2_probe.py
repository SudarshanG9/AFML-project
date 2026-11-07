
"""Day 2: inspect DiMoDif without changing its weights or outputs."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def load_features(path: Path):
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            features = {
                "video_tokens": torch.from_numpy(
                    data["video_features"].copy()
                ),
                "audio_tokens": torch.from_numpy(
                    data["audio_features"].copy()
                ),
            }
    elif path.suffix == ".pt":
        from avfuse.member2_data.cache import load_feature_cache
        features = load_feature_cache(path)
    else:
        raise ValueError(f"Expected .npz or .pt: {path}")

    for key in ("video_tokens", "audio_tokens"):
        x = features[key]
        if x.ndim not in (2, 3) or x.shape[-1] != 768:
            raise ValueError(f"Invalid {key} shape: {tuple(x.shape)}")

    return features


@torch.inference_mode()
def modality_ablation(adapter, features):
    """Change one modality at a time, keeping the model frozen."""
    scores = {}

    variants = (
        ("reference", None),
        ("zero_audio", "audio_tokens"),
        ("zero_video", "video_tokens"),
    )

    for name, zero_key in variants:
        inputs = dict(features)

        if zero_key is not None:
            inputs[zero_key] = torch.zeros_like(features[zero_key])

        score = adapter.predict_from_cached_features(
            inputs, "reference"
        )

        if score.numel() != 1:
            raise ValueError("Expected batch size 1")

        scores[name] = float(score.reshape(-1)[0].item())

    scores["delta_zero_audio"] = (
        scores["zero_audio"] - scores["reference"]
    )
    scores["delta_zero_video"] = (
        scores["zero_video"] - scores["reference"]
    )
    return scores


def cross_modal_mask_audit(model, valid_length):
    """Token order: CLS | video | SEP | audio."""
    maximum = int(model.max_length)
    n_tokens = 2 * maximum + 2

    if not (1 <= valid_length <= maximum):
        raise ValueError("Invalid valid_length")

    mask = model.mask

    if mask is None:
        allowed = torch.ones(
            (n_tokens, n_tokens), dtype=torch.bool
        )
    else:
        if tuple(mask.shape) != (n_tokens, n_tokens):
            raise ValueError("Unexpected attention mask shape")

        # MultiheadAttention: True means blocked.
        if mask.dtype == torch.bool:
            allowed = (~mask).cpu()
        else:
            allowed = torch.isfinite(mask).cpu()

    video = slice(1, 1 + valid_length)
    audio = slice(
        maximum + 2, maximum + 2 + valid_length
    )

    v_to_a = int(allowed[video, audio].sum().item())
    a_to_v = int(allowed[audio, video].sum().item())

    return {
        "mask_shape": list(allowed.shape),
        "mask_present": mask is not None,
        "valid_tokens_per_modality": valid_length,
        "video_to_audio_permitted_pairs": v_to_a,
        "audio_to_video_permitted_pairs": a_to_v,
        "all_cross_pairs_per_direction": valid_length ** 2,
        "cross_modal_attention_possible": bool(
            v_to_a and a_to_v
        ),
        "padding_tokens_per_modality": (
            maximum - valid_length
        ),
        # Based on upstream AVDeepFakeDetector.forward.
        "has_src_key_padding_mask": False,
    }


def trace_qkv(model, forward_call):
    """Observe actual MHA inputs and projection shapes."""
    trace = []
    handles = []
    layers = model.transformer_encoder.layers

    def make_hook(index):
        def pre_hook(mha: nn.MultiheadAttention, args, kwargs):
            q, k, v = args[:3]
            heads = mha.num_heads
            dimension = mha.embed_dim

            if not mha.batch_first:
                raise RuntimeError("Expected batch_first MHA")

            if mha.in_proj_weight is None:
                raise RuntimeError(
                    "Expected packed QKV projections"
                )

            weights = mha.in_proj_weight.chunk(3, dim=0)

            if mha.in_proj_bias is None:
                biases = (None, None, None)
            else:
                biases = mha.in_proj_bias.chunk(3, dim=0)

            projected = [
                F.linear(x, w, b)
                for x, w, b in zip(
                    (q, k, v), weights, biases
                )
            ]

            shapes = [
                [
                    x.shape[0],
                    heads,
                    x.shape[1],
                    dimension // heads,
                ]
                for x in projected
            ]

            mask = kwargs.get("attn_mask")

            trace.append({
                "layer": index,
                "module": (
                    f"transformer_encoder.layers."
                    f"{index}.self_attn"
                ),
                "qkv_head_shapes": shapes,
                "qkv_abs_means": [
                    float(x.abs().mean().item())
                    for x in projected
                ],
                "input_dtype": str(q.dtype),
                "attention_mask_shape": (
                    list(mask.shape)
                    if mask is not None else None
                ),
                "input_tensors_shared": (
                    q is k and k is v
                ),
            })

        return pre_hook

    for index, layer in enumerate(layers):
        handle = layer.self_attn.register_forward_pre_hook(
            make_hook(index),
            with_kwargs=True,
        )
        handles.append(handle)

    try:
        with torch.inference_mode():
            output = forward_call()
    finally:
        for handle in handles:
            handle.remove()

    if len(trace) != len(layers):
        raise RuntimeError(
            "Not every attention layer was executed"
        )

    return trace, output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--upstream",
        type=Path,
        default=Path("third_party/dimodif"),
    )
    parser.add_argument(
        "--features", nargs="+", type=Path, required=True
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/day2/lead_probe.json"),
    )
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be positive")

    from avfuse.member1_fusion.dimodif_adapter import (
        DiMoDifAdapter,
    )

    adapter = DiMoDifAdapter(
        args.upstream,
        checkpoint_name="lavdf",
        device=args.device,
    )

    records = []

    for path in args.features[:args.limit]:
        features = load_features(path)

        valid_length = min(
            features["video_tokens"].shape[-2],
            features["audio_tokens"].shape[-2],
            adapter.max_length,
        )

        scores = modality_ablation(adapter, features)

        clip_id = (
            path.parent.parent.name
            if path.name == "features.npz"
            else path.stem
        )

        record = {
            "clip_id": clip_id,
            "path": str(path),
            "valid_length": int(valid_length),
            **scores,
        }

        # Trace real QKV once; avoid repeated hook overhead.
        if not records:
            trace, hooked_score = trace_qkv(
                adapter.model,
                lambda: adapter.predict_from_cached_features(
                    features, "reference"
                ),
            )

            reference = torch.tensor(
                [scores["reference"]],
                device=hooked_score.device,
            )

            if not torch.allclose(
                hooked_score.reshape(-1),
                reference,
                atol=1e-6,
                rtol=1e-5,
            ):
                raise RuntimeError(
                    "Attention hooks changed the prediction"
                )

            record["fusion_qkv_trace"] = trace
            record["cross_modal_mask"] = (
                cross_modal_mask_audit(
                    adapter.model, int(valid_length)
                )
            )

        records.append(record)

        print(
            f"{clip_id}: "
            f"reference={scores['reference']:.6f}, "
            f"zero_audio={scores['zero_audio']:.6f}, "
            f"zero_video={scores['zero_video']:.6f}"
        )

    result = {
        "model": "dimodif",
        "checkpoint_name": "lavdf",
        "checkpoint_sha256": adapter.checkpoint_sha256,
        "device": str(adapter.device),
        "torch_version": torch.__version__,
        "n_samples": len(records),
        "samples": records,
        "mean_abs_delta_audio": float(np.mean([
            abs(r["delta_zero_audio"]) for r in records
        ])),
        "mean_abs_delta_video": float(np.mean([
            abs(r["delta_zero_video"]) for r in records
        ])),
    }

    args.output.parent.mkdir(
        parents=True, exist_ok=True
    )
    args.output.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
