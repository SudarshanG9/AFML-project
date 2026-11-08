
"""Day 2: Inspect DiMoDif's pretrained Q/K/V fusion boundary."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from avfuse.member1_fusion.dimodif_adapter import DiMoDifAdapter


def load_features(path):
    with np.load(path, allow_pickle=False) as data:
        video = np.array(
            data["video_features"], dtype=np.float32, copy=True
        )
        audio = np.array(
            data["audio_features"], dtype=np.float32, copy=True
        )

    return {
        "video_tokens": torch.from_numpy(video),
        "audio_tokens": torch.from_numpy(audio),
    }


def inspect_mask(model):
    length = model.max_length
    mask = model.mask

    if mask is None:
        return {"mask_present": False, "type": "unrestricted"}

    # Original DiMoDif: True means an attention pair is BLOCKED.
    blocked = mask.detach().cpu().bool()
    allowed = ~blocked

    video = slice(1, length + 1)
    audio = slice(length + 2, 2 * length + 2)
    cls = 0
    sep = length + 1

    return {
        "mask_present": True,
        "shape": list(mask.shape),
        "true_means": "blocked",
        "video_to_audio_allowed": int(
            allowed[video, audio].sum()
        ),
        "audio_to_video_allowed": int(
            allowed[audio, video].sum()
        ),
        "cls_to_video_allowed": int(
            allowed[cls, video].sum()
        ),
        "cls_to_audio_allowed": int(
            allowed[cls, audio].sum()
        ),
        "sep_to_video_allowed": int(
            allowed[sep, video].sum()
        ),
        "sep_to_audio_allowed": int(
            allowed[sep, audio].sum()
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument(
        "--upstream", default="third_party/dimodif"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        default="artifacts/day2/attention_audit.json",
    )
    args = parser.parse_args()

    adapter = DiMoDifAdapter(
        upstream_root=args.upstream,
        checkpoint_name="lavdf",
        device=args.device,
    )

    model = adapter.model
    features = load_features(args.features)
    records = []
    handles = []

    layers = list(model.transformer_encoder.layers)

    for index, layer in enumerate(layers):

        def inspect_attention(module, inputs, kwargs, idx=index):
            query, key, value = inputs[:3]

            # DiMoDif uses self-attention over the joint sequence.
            same_input = (
                query.data_ptr() == key.data_ptr()
                and key.data_ptr() == value.data_ptr()
            )

            # Reconstruct pretrained Q/K/V projections for inspection.
            weights = module.in_proj_weight.chunk(3, dim=0)

            if module.in_proj_bias is None:
                biases = (None, None, None)
            else:
                biases = module.in_proj_bias.chunk(3, dim=0)

            projected = [
                F.linear(tensor, weight, bias)
                for tensor, weight, bias in zip(
                    (query, key, value), weights, biases
                )
            ]

            batch, tokens, hidden = query.shape
            heads = module.num_heads
            head_dim = module.head_dim

            reshaped = [
                tensor.reshape(
                    batch, tokens, heads, head_dim
                ).transpose(1, 2)
                for tensor in projected
            ]

            observed_mask = kwargs.get("attn_mask")

            mask_matches = None

            if observed_mask is not None and model.mask is not None:
                if observed_mask.dtype == torch.bool:
                    observed_blocked = observed_mask
                else:
                    observed_blocked = torch.isneginf(observed_mask)

                mask_matches = bool(
                    torch.equal(
                        observed_blocked.detach().cpu(),
                        model.mask.detach().cpu().bool(),
                    )
                )

            records.append({
                "layer": idx,
                "self_attention": same_input,
                "input_shape": list(query.shape),
                "Q_shape": list(reshaped[0].shape),
                "K_shape": list(reshaped[1].shape),
                "V_shape": list(reshaped[2].shape),
                "dtype": str(query.dtype),
                "N_q": int(reshaped[0].shape[2]),
                "N_k": int(reshaped[1].shape[2]),
                "head_dim": int(head_dim),
                "attention_mask_shape": (
                    list(observed_mask.shape)
                    if observed_mask is not None
                    else None
    ),
                "Q_max_abs": float(reshaped[0].abs().max()),
                "K_max_abs": float(reshaped[1].abs().max()),
                "V_max_abs": float(reshaped[2].abs().max()),
                "mask_received": observed_mask is not None,
                "mask_matches_original": mask_matches,
                "key_padding_mask_received":
                    kwargs.get("key_padding_mask") is not None,
            })

        handle = layer.self_attn.register_forward_pre_hook(
            inspect_attention,
            with_kwargs=True,
        )
        handles.append(handle)

    try:
        with torch.inference_mode():
            result = adapter.forward_reference(features)
    finally:
        for handle in handles:
            handle.remove()

    report = {
        "checkpoint_sha256": adapter.checkpoint_sha256,
        "feature_file": args.features,
        "valid_length": int(result["valid_length"]),
        "fake_score": float(result["score_fake"][0]),
        "config": {
            "d_model": adapter.config["d_model"],
            "nhead": adapter.config["nhead"],
            "nlayers": adapter.config["nlayers"],
            "max_length": adapter.config["max_length"],
            "win_size": adapter.config["win_size"],
        },
        "attention_mask": inspect_mask(model),
        "layers": records,
    }

    assert len(records) == len(layers), (
        f"Expected {len(layers)} attention calls, "
        f"observed {len(records)}"
    )

    for record in records:
        assert record["self_attention"]
        assert record["mask_matches_original"] is True
        assert not record["key_padding_mask_received"]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))

    print("\n===== ATTENTION BOUNDARY AUDIT =====")
    print("Layers inspected:", len(records))
    print("Valid source tokens per stream:", result["valid_length"])
    print("Fake score:", report["fake_score"])

    for record in records:
        print(
            f"Layer {record['layer']}: "
            f"Q={record['Q_shape']} "
            f"K={record['K_shape']} "
            f"V={record['V_shape']}"
        )

    print("\nMask permissions:")
    for key, value in report["attention_mask"].items():
        print(f"{key}: {value}")

    print("\nSaved to:", output)


if __name__ == "__main__":
    main()
