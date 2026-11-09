
import argparse
import json
from pathlib import Path

import torch

from avfuse.member1_fusion.dimodif_adapter import (
    DiMoDifAdapter,
)
from avfuse.member1_fusion.day2_probe import (
    load_features,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--features",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--upstream",
        type=Path,
        default=Path("third_party/dimodif"),
    )

    args = parser.parse_args()

    # Load the same frozen checkpoint used during Day 2.
    adapter = DiMoDifAdapter(
        upstream_root=args.upstream,
        checkpoint_name="lavdf",
        device="cpu",
    )

    audit_path = Path(
        "artifacts/day2/attention_audit.json"
    )

    audit = json.loads(audit_path.read_text())

    assert (
        adapter.checkpoint_sha256
        == audit["checkpoint_sha256"]
    ), "Checkpoint differs from Day 2"

    features = load_features(args.features)

    video, audio, valid_length = (
        adapter.prepare_features(features)
    )

    max_len = adapter.max_length
    n_tokens = 2 * max_len + 2

    print("Valid frames:", valid_length)
    print("Video padded shape:", tuple(video.shape))
    print("Audio padded shape:", tuple(audio.shape))
    print("Total attention tokens:", n_tokens)

    # The original mask uses True = BLOCKED.
    original_mask = adapter.model.mask

    if original_mask is None:
        allowed = torch.ones(
            n_tokens, n_tokens,
            dtype=torch.bool,
        )
    else:
        original_mask = original_mask.cpu()

        assert original_mask.dtype == torch.bool
        assert original_mask.shape == (
            n_tokens, n_tokens
        )

        allowed = ~original_mask

    # Exact positions in the combined sequence.
    groups = {
        "cls": [0],

        "video_valid": list(
            range(1, valid_length + 1)
        ),

        "video_padding": list(
            range(valid_length + 1, max_len + 1)
        ),

        "sep": [max_len + 1],

        "audio_valid": list(
            range(
                max_len + 2,
                max_len + 2 + valid_length,
            )
        ),

        "audio_padding": list(
            range(
                max_len + 2 + valid_length,
                2 * max_len + 2,
            )
        ),
    }

    def count_allowed(source, target):
        q_indices = groups[source]
        k_indices = groups[target]

        if not q_indices or not k_indices:
            return 0

        selected = allowed[
            q_indices, :
        ][:, k_indices]

        return int(selected.sum().item())

    # Examine whether real queries can attend
    # to padded keys.
    query_groups = [
        "cls",
        "video_valid",
        "audio_valid",
        "video_padding",
        "audio_padding",
    ]

    key_groups = [
        "video_valid",
        "audio_valid",
        "video_padding",
        "audio_padding",
    ]

    permissions = {}

    for query_group in query_groups:
        permissions[query_group] = {}

        for key_group in key_groups:
            count = count_allowed(
                query_group, key_group
            )

            permissions[query_group][key_group] = count

            print(
                f"{query_group:16s} -> "
                f"{key_group:16s}: {count}"
            )

    # Confirm that prepare_features() really zero-pads.
    video_padding = video[:, valid_length:, :]
    audio_padding = audio[:, valid_length:, :]

    video_padding_zero = bool(
        torch.count_nonzero(video_padding) == 0
    )

    audio_padding_zero = bool(
        torch.count_nonzero(audio_padding) == 0
    )

    report = {
        "checkpoint_sha256":
            adapter.checkpoint_sha256,
        "feature_file": str(args.features),
        "valid_length": valid_length,
        "max_length": max_len,
        "attention_tokens": n_tokens,
        "padding_per_modality":
            max_len - valid_length,
        "video_input_padding_zero":
            video_padding_zero,
        "audio_input_padding_zero":
            audio_padding_zero,
        "mask_semantics": "True means blocked",
        "separate_key_padding_mask": False,
        "permitted_attention_pairs": permissions,
    }

    output = Path(
        "artifacts/dayA/padding_mask_audit.json"
    )

    output.parent.mkdir(
        parents=True, exist_ok=True
    )

    output.write_text(
        json.dumps(report, indent=2)
    )

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
