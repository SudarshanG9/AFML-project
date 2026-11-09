
import json
from pathlib import Path

from avfuse.member1_fusion.attention import score_matrix_bytes
import torch

SOURCE = Path("artifacts/day2/attention_audit.json")
DEST = Path("artifacts/dayA/attention_memory.json")


def main():
    with SOURCE.open() as f:
        audit = json.load(f)

    records = []

    for layer in audit["layers"]:
        B, H, Nq, Dq = layer["Q_shape"]
        Bk, Hk, Nk, Dk = layer["K_shape"]
        Bv, Hv, Nv, Dv = layer["V_shape"]

        assert (B, H, Dq) == (Bk, Hk, Dk)
        assert (Bk, Hk, Nk) == (Bv, Hv, Nv)
        assert layer["dtype"] == "torch.float32"

        bytes_per_element = 4

        scores = B * H * Nq * Nk * bytes_per_element
        qkv = B * H * (Nq * Dq + Nk * Dk + Nv * Dv) \
            * bytes_per_element

        # Verify existing Member 1 implementation.
        q = torch.empty(B, H, Nq, Dq)
        k = torch.empty(B, H, Nk, Dk)

        assert score_matrix_bytes(q, k) == scores

        records.append({
            "layer": layer["layer"],
            "q_shape": layer["Q_shape"],
            "k_shape": layer["K_shape"],
            "v_shape": layer["V_shape"],
            "dtype": layer["dtype"],
            "score_matrix_bytes": scores,
            "score_matrix_mib": scores / 2**20,
            "qkv_bytes": qkv,
            "qkv_mib": qkv / 2**20,
            "mask_shape": layer["attention_mask_shape"],
        })

    report = {
        "checkpoint_sha256": audit["checkpoint_sha256"],
        "source": str(SOURCE),
        "formula": "B * H * Nq * Nk * bytes_per_element",
        "interpretation": (
            "Theoretical dense score matrix only; "
            "not a measured SDPA memory allocation."
        ),
        "layers": records,
    }

    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(json.dumps(report, indent=2))

    for item in records:
        print(
            f"Layer {item['layer']}: "
            f"scores={item['score_matrix_mib']:.3f} MiB, "
            f"QKV={item['qkv_mib']:.3f} MiB"
        )


if __name__ == "__main__":
    main()
