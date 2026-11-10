"""
Member 3 Day 4 verification — real-shape operator benchmark sweep.

What this does (Member 1's role):
  - Loads the real Q/K/V tensors captured by dayA_model_parity.py and stored
    in artifacts/dayA/layer0_qkv.pt.  Shape: [1, 4, 1202, 16].
  - Verifies the tensors match the shapes confirmed in operator_handoff.json.
  - Runs benchmark_attention_operator() for every precision mode across both
    supported backends (sdpa, dense_oracle) using these real-shaped tensors.
  - Writes the full sweep to artifacts/m3_day4/operator_sweep.json.

What this does NOT do:
  - It does NOT claim these are GPU timings (runs on cpu).
  - It does NOT present simulated INT8 numbers as hardware INT8 speedups.
  - It does NOT replace full-model cached-feature latency (operator scope only).
"""

import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from avfuse.member3_runtime.operator_benchmark import (
    BACKEND_DENSE_ORACLE,
    BACKEND_SDPA,
    OperatorInputs,
    benchmark_attention_operator,
    run_operator_sweep,
    score_matrix_estimate,
    validate_operator_inputs,
)
from avfuse.shared.contracts import PRECISION_MODES

QKV_PATH       = REPO / "artifacts" / "dayA" / "layer0_qkv.pt"
HANDOFF_PATH   = REPO / "artifacts" / "dayA" / "operator_handoff.json"
OUTPUT_DIR     = REPO / "artifacts" / "m3_day4"
DEVICE         = "cpu"
WARMUP         = 2
REPEATS        = 5          # keep fast; adjust for real latency research runs

EXPECTED_SHAPE = (1, 4, 1202, 16)
CHECKPOINT_SHA = "e1e59006d1bb6ad36a3859af5486a6649ed41bba23258b7c41e24f873be7948e"

# Dense oracle score matrix for shape [1,4,1202,1202] x float32 = 23,116,864 bytes (~22 MiB).
# That is within the 128 MiB default limit, but dense needs ~3x for softmax buffer.
# We raise the limit slightly to allow it without hitting the safety guard.
DENSE_LIMIT = 128 * 1024 * 1024


def _load_and_check_qkv() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if not QKV_PATH.is_file():
        raise FileNotFoundError(
            f"Real Q/K/V file not found: {QKV_PATH}\n"
            "Run scripts/dayA_model_parity.py first to generate it."
        )
    saved = torch.load(QKV_PATH, map_location="cpu", weights_only=True)
    q, k, v = saved["q"], saved["k"], saved["v"]
    mask = saved.get("attn_mask")

    for name, tensor in [("q", q), ("k", k), ("v", v)]:
        assert tuple(tensor.shape) == EXPECTED_SHAPE, (
            f"{name} shape {tuple(tensor.shape)} != expected {EXPECTED_SHAPE}"
        )
        assert tensor.dtype == torch.float32, f"{name} dtype {tensor.dtype} != float32"
        assert torch.isfinite(tensor).all(), f"{name} contains NaN/Inf"

    return q, k, v, mask


def _check_against_handoff() -> None:
    if not HANDOFF_PATH.is_file():
        print("  WARNING: operator_handoff.json not found; skipping cross-check.")
        return
    handoff = json.loads(HANDOFF_PATH.read_text(encoding="utf-8"))
    assert handoff.get("passed") is True, "operator_handoff.json passed != true"
    assert handoff.get("checkpoint_sha256") == CHECKPOINT_SHA, (
        f"Handoff checkpoint SHA mismatch: {handoff.get('checkpoint_sha256')}"
    )
    assert handoff.get("q_shape") == list(EXPECTED_SHAPE), (
        f"Handoff Q shape {handoff.get('q_shape')} != {list(EXPECTED_SHAPE)}"
    )
    print(f"  Handoff cross-check: passed=True, SHA matches, shapes match.")


def main() -> None:
    print("=== Member 3 Day 4: Operator Benchmark — real-shaped Q/K/V sweep ===\n")

    # --- 1. Load and validate real tensors ---------------------------------------
    print(f"Loading Q/K/V from {QKV_PATH.relative_to(REPO)} ...")
    q, k, v, mask = _load_and_check_qkv()
    print(f"  Q shape : {tuple(q.shape)}  dtype: {q.dtype}")
    print(f"  K shape : {tuple(k.shape)}")
    print(f"  V shape : {tuple(v.shape)}")
    print(f"  mask    : {tuple(mask.shape) if mask is not None else None}")

    # --- 2. Cross-check against the existing handoff artifact --------------------
    print(f"\nCross-checking against operator_handoff.json ...")
    _check_against_handoff()

    # --- 3. Score-matrix estimate ------------------------------------------------
    estimate = score_matrix_estimate(q, k)
    print(
        f"\nScore-matrix estimate: {estimate['bytes']:,} bytes "
        f"({estimate['mib']:.2f} MiB) for shape [B=1, H=4, Nq=1202, Nk=1202]"
    )
    assert estimate["bytes"] == 23_116_864, (
        f"Score-matrix bytes mismatch: {estimate['bytes']} != 23116864"
    )

    # --- 4. Validate inputs against the benchmark's rules ------------------------
    dev = torch.device(DEVICE)
    # Move mask to same device as q/k/v (all cpu already)
    mask_on_dev = mask.to(dev) if mask is not None else None
    validate_operator_inputs(q, k, v, mask_on_dev, dev)
    print("  Input validation: passed")

    # --- 5. Run the full sweep ---------------------------------------------------
    # Build a single OperatorInputs from the real tensors
    real_case = OperatorInputs(
        q=q.to(dev), k=k.to(dev), v=v.to(dev),
        attn_mask=mask_on_dev,
        name=f"real_dimodif_B1_H4_Nq1202_D16",
    )

    shape_source = (
        f"Layer 0 Q/K/V from DiMoDif LAV-DF checkpoint "
        f"SHA-256={CHECKPOINT_SHA} | artifact artifacts/dayA/layer0_qkv.pt"
    )

    print(
        f"\nRunning sweep: backends=[sdpa, dense_oracle] x modes={list(PRECISION_MODES)} "
        f"(warmup={WARMUP}, repeats={REPEATS}, device={DEVICE}) ..."
    )

    results = run_operator_sweep(
        cases=[real_case],
        input_kind="real",
        shape_source=shape_source,
        device=DEVICE,
        modes=list(PRECISION_MODES),
        backends=[BACKEND_SDPA, BACKEND_DENSE_ORACLE],
        warmup=WARMUP,
        repeats=REPEATS,
        measure_memory=True,
        require_memory=False,           # CPU: memory not measured, never 0
        dense_max_score_bytes=DENSE_LIMIT,
    )

    # --- 6. Print results -------------------------------------------------------
    print(f"\n--- Results ({len(results)} runs) ---")
    print(f"{'backend':<15} {'mode':<25} {'p50_ms':>10} {'p95_ms':>10}  evidence")
    print("-" * 75)
    for r in results:
        print(
            f"  {r['backend']:<13} {r['precision_mode']:<25} "
            f"{r['p50_ms']:>10.3f} {r['p95_ms']:>10.3f}  {r['evidence_status']}"
        )

    # --- 7. Save artifact -------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "description": "Member 3 Day 4: real-shaped operator sweep over DiMoDif Q/K/V",
        "checkpoint_sha256": CHECKPOINT_SHA,
        "qkv_source": str(QKV_PATH.relative_to(REPO)),
        "shape_source": shape_source,
        "q_shape": list(q.shape),
        "k_shape": list(k.shape),
        "v_shape": list(v.shape),
        "dtype": str(q.dtype),
        "mask_shape": list(mask.shape) if mask is not None else None,
        "score_matrix_bytes": estimate["bytes"],
        "device": DEVICE,
        "warmup": WARMUP,
        "repeats": REPEATS,
        "results": results,
    }
    output_path = OUTPUT_DIR / "operator_sweep.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nOperator sweep saved to: {output_path.relative_to(REPO)}")

    # --- 8. Assertions ----------------------------------------------------------
    assert len(results) == len(PRECISION_MODES) * 2, (
        f"Expected {len(PRECISION_MODES) * 2} results, got {len(results)}"
    )
    for r in results:
        assert r["evidence_status"] == "real_shapes_caller_declared", (
            f"Expected real evidence, got {r['evidence_status']!r}"
        )
        assert r["p50_ms"] > 0, f"p50_ms must be positive for {r['backend']}/{r['precision_mode']}"
        assert r["p50_ms"] <= r["p95_ms"], "p50 > p95 is impossible"
        assert r["input_kind"] == "real"

    # Reference backend should not be significantly slower than low_precision on cpu
    # (fake quant is floating-point, so the difference is small — just check both finite)
    sdpa_ref = next(r for r in results if r["backend"] == "sdpa" and r["precision_mode"] == "reference")
    sdpa_lp  = next(r for r in results if r["backend"] == "sdpa" and r["precision_mode"] == "low_precision")
    assert sdpa_ref["p50_ms"] > 0 and sdpa_lp["p50_ms"] > 0

    print("\n✅  All assertions passed — Member 3 Day 4 verification complete.")


if __name__ == "__main__":
    main()

