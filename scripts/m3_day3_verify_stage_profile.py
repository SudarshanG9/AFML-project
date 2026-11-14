"""
Member 3 Day 3 verification — bind real fusion/head callables and run stage profiler.

What this does (Member 1's role):
  - Loads the real DiMoDif checkpoint via DiMoDifAdapter.
  - Binds the `fusion` and `head` stages to real callables using the pilot
    feature cache (data/pilot/test_day2/val/<id>/features.npz).
  - Runs profile_pipeline_stages() with those two stages.
  - Saves the stage profile artifact to artifacts/m3_day3/stage_profile.json.

What this does NOT do:
  - It does not time Member 2's preprocessing stages (video decode, manifest
    build, etc.); those remain `not_provided` because Member 2 must bind them.
  - It does not claim an end-to-end latency (end_to_end_measured stays False).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from avfuse.member1_fusion.dimodif_adapter import DiMoDifAdapter
from avfuse.member3_runtime.stage_profiler import (
    StageInput,
    StageSettings,
    profile_pipeline_stages,
)

UPSTREAM        = REPO / "third_party" / "dimodif"
FEATURE_DIR     = REPO / "data" / "pilot" / "test_day2" / "val"
OUTPUT_DIR      = REPO / "artifacts" / "m3_day3"
DEVICE          = "cpu"          # MPS has no resettable peak counter; use cpu for profiling
CHECKPOINT_NAME = "lavdf"
WARMUP          = 2
REPEATS         = 5             # keep fast; not a latency benchmark, just verification


def _load_npz(npz_path: Path) -> dict[str, torch.Tensor]:
    with np.load(npz_path, allow_pickle=False) as f:
        video = torch.from_numpy(np.asarray(f["video_features"], dtype=np.float32))
        audio = torch.from_numpy(np.asarray(f["audio_features"], dtype=np.float32))
    return {"video_tokens": video, "audio_tokens": audio}


def main() -> None:
    print("=== Member 3 Day 3: Stage Profiler — fusion + head (real callables) ===\n")

    # --- 1. Find a feature file -------------------------------------------------
    npz_files = sorted(FEATURE_DIR.rglob("features.npz"))
    if not npz_files:
        raise FileNotFoundError(
            f"No features.npz found under {FEATURE_DIR}. "
            "Run Member 2's feature extraction script first."
        )
    npz_path = npz_files[0]
    print(f"Using feature file: {npz_path.relative_to(REPO)}")

    features = _load_npz(npz_path)
    print(
        f"  video_tokens: {tuple(features['video_tokens'].shape)}  "
        f"audio_tokens: {tuple(features['audio_tokens'].shape)}"
    )

    # --- 2. Load the real pretrained checkpoint ---------------------------------
    print(f"\nLoading DiMoDifAdapter (checkpoint={CHECKPOINT_NAME!r}, device={DEVICE!r}) ...")
    adapter = DiMoDifAdapter(
        upstream_root=UPSTREAM,
        checkpoint_name=CHECKPOINT_NAME,
        device=DEVICE,
    )
    print(f"  checkpoint SHA-256: {adapter.checkpoint_sha256}")

    # --- 3. Build zero-argument callables for `fusion` and `head` ---------------
    # `fusion` = the full forward_reference pass (joint attention over both streams)
    # `head`   = just the final score extraction from an already-computed output
    #
    # DiMoDifAdapter.forward_reference() returns a dict that includes logits and
    # modality_scores.  For profiling purposes we time the entire forward pass as
    # `fusion`, and isolate the sigmoid + max reduction as `head`.

    cached_ref: dict = {}

    @torch.inference_mode()
    def run_fusion() -> None:
        result = adapter.forward_reference(features)
        cached_ref.update(result)

    @torch.inference_mode()
    def run_head() -> None:
        # Reproduce the detection-head reduction from DiMoDifAdapter.forward_reference:
        # sigmoid -> (optional pyramid average) -> amax
        logits = cached_ref["logits"]
        probs  = torch.sigmoid(logits)
        if adapter.config.get("feature_pyramid", False):
            probs = probs.mean(dim=1)
        _score = probs.amax(dim=-1)

    # Prime the cache so run_head() has valid tensors during profiling
    run_fusion()

    # --- 4. Run the profiler with these two real callables ----------------------
    print(f"\nRunning profile_pipeline_stages (warmup={WARMUP}, repeats={REPEATS}) ...")

    real_source = (
        f"features.npz from {npz_path.relative_to(REPO)} "
        f"| checkpoint SHA-256 {adapter.checkpoint_sha256}"
    )

    profile = profile_pipeline_stages(
        stages={
            "fusion": run_fusion,
            "head":   run_head,
        },
        device=DEVICE,
        input_kind="real",
        input_source=real_source,
        stage_settings={
            "fusion": StageSettings(warmup=WARMUP, repeats=REPEATS),
            "head":   StageSettings(warmup=WARMUP, repeats=REPEATS),
        },
        stage_inputs={
            "fusion": StageInput(kind="real", source=real_source),
            "head":   StageInput(kind="real", source=real_source),
        },
    )

    # --- 5. Print summary -------------------------------------------------------
    print("\n--- Measured stages ---")
    for record in profile["stages"]:
        stage  = record["stage"]
        status = record["status"]
        if record.get("measurement_completed"):
            p50 = record.get("p50_ms", "?")
            p95 = record.get("p95_ms", "?")
            print(f"  {stage:<35} status={status}  p50={p50:.3f} ms  p95={p95:.3f} ms")
        else:
            print(f"  {stage:<35} status={status}")

    scope_sum = profile["scope_summary"]
    for group, info in scope_sum.items():
        complete = "COMPLETE" if info["complete"] else "INCOMPLETE"
        measured = info["measured_stages"]
        missing  = info["missing_stages"]
        print(f"\n  Scope group '{group}': {complete}")
        print(f"    measured : {measured}")
        if missing:
            print(f"    missing  : {missing}")

    print(f"\n  eligible_as_real_pipeline_profile : {profile['eligible_as_real_pipeline_profile']}")
    print(f"  evidence_status                   : {profile['evidence_status']}")

    # --- 6. Write artifact ------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "stage_profile.json"
    output_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(f"\nStage profile saved to: {output_path.relative_to(REPO)}")

    # --- 7. Assertions for CI-style pass/fail -----------------------------------
    fusion_record = next(r for r in profile["stages"] if r["stage"] == "fusion")
    head_record   = next(r for r in profile["stages"] if r["stage"] == "head")

    assert fusion_record["status"] == "measured", \
        f"fusion stage not measured: {fusion_record['status']}"
    assert head_record["status"] == "measured", \
        f"head stage not measured: {head_record['status']}"
    assert fusion_record["p50_ms"] > 0, "fusion p50_ms must be positive"
    assert head_record["p50_ms"] > 0,   "head p50_ms must be positive"
    assert fusion_record["p50_ms"] <= fusion_record["p95_ms"], \
        "fusion p50 > p95 (impossible)"
    assert profile["input_kind"] == "real", \
        f"Expected input_kind='real', got {profile['input_kind']!r}"

    print("\n✅  All assertions passed — Member 3 Day 3 verification complete.")


if __name__ == "__main__":
    main()

