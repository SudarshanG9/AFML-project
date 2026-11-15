# B2 outstanding owner evidence — no new review gate

B1 is accepted by C8A. B2 must finish C8.3–C8.7 before the lead reviews it at C8B. The five implementation-block reviews are C8A (B1), C8B (B2), C9 (B3), C10 (B4), C10R (B5); B0 is already accepted baseline preparation. A user-run Colab handoff is execution, not an extra scientific review.

## Member 2: exact unresolved interface requirements

The unchanged `scripts/m2_evaluate.py evaluate` already emits all four overall metric blocks, the direct query-versus-value paired contrast, fixed-threshold recall/FPR and per-type AUROC/AUPRC/recall. `scripts/m2_paired.py` emits damage/restoration results. These interfaces and their source remain unchanged.

Inspection found two unmet complete-plan requirements: `cluster_resamples` in `src/avfuse/member2_data/evaluate.py` draws whole connected source/identity clusters without explicit manipulation-stratum preservation; `compare` and `m2_paired.py` do not emit the complete per-stratum paired CIs. Per-type blocks also do not supply BCE intervals. Do not silently label current output stratified, reduce resamples or replace the evaluator.

Required owner handoff after the genuine query output is returned: full four-mode metrics/report/plots using the original calibration (`358c037242f6980fed2a8d0ff0ef830f17b4789676b391c8f6dff245842b37dd`), same 1200 validation IDs and original checkpoint; paired source-group CIs with 1000 resamples, seed 0, explicit approved stratification method and handling of connected clusters spanning strata; paired lowp/reference, both restoration/lowp and direct V/output/query contrasts, overall and each of audio_fake-vs-real, video_fake-vs-real and both_fake-vs-real. Include AUROC/AUPRC/BCE, fixed-threshold recall/FPR and exact provenance. Do not optimize a new threshold. No test feature or test prediction access.

The lead or Member 2 must provide the approved resampling implementation/protocol and owned results; Member 1 will not invent it or edit Member 2 code. The current Member 1 research reader exposes missing stratification/per-stratum evidence as BLOCKED, rather than a complete research result. Exact paths/hashes will be bound to the real returned output, never filled with invented values.

Existing commands for owner review (execute only when all input paths exist and this method gap is resolved): `m2_check_predictions.py`, `m2_evaluate.py evaluate --split val --n-boot 1000 --seed 0`, and `m2_paired.py --scope cached_feature_model --n-boot 1000 --seed 0`. Original canonical manifest and original frozen calibration must be used. Typical cost input is the new `benchmarks/four_modes_typical.jsonl` only, not a concatenation of short/typical/long.

## Member 3: real matched-cost review

The B2 Colab handoff invokes the byte-unchanged `scripts/dayB_model_benchmark.py` for the original approved short `6d8daea9cc33fe63`, typical `7cea6ad75b4ebf41`, long `891214bf9e5f17d0` validation clips. Each has all four modes, batch 1, CPU/SDPA/FP32 with fake INT8 boundaries, 10 warmups and 50 measured repetitions, full raw timing samples, same checkpoint/input hashes and device/runtime. CPU memory stays null/not_measured. Whole cached-model timing includes padding/feature preparation, quantize/dequantize/restoration, joint fusion and the head; raw decoding/feature extraction are excluded.

Member 3 must review those returned records, input/runtime identities, variability, typical primary vs short/long sensitivity, and whether extra costs are sufficiently reliable for descriptive normalization. No automatic equal-cost claim or fake INT8 hardware acceleration claim. Member 1 keeps normalization undefined without positive comparable owner-reviewed reliable extra costs. BCE intervals do not automatically account for timing uncertainty.

This document prepares the complete owner request; no message has been sent to other people or services.


## Resolution after successful B2 return

The accepted Day B final script already supplies a precedent for calling the unchanged Member 2 public bootstrap APIs separately within each fake-versus-real analysis. The earlier assertion that a Member 2 source patch was required is superseded. `COLAB_B2_EVALUATION_ONLY.md` prepares the complete four-mode evaluation using that existing route. Analysis-stratum subsetting is explicit; source-cluster resampling does not keep class counts fixed in each replicate. No new resampling algorithm is assumed, no owner approval is invented, and the exact method is presented at the existing C8B science gate. Member 3 final benchmark freeze/release remains a later C9 dependency; real B2 raw costs are now available and technically checked.


## Latest user steering: joint B2/B3 audit

The user subsequently authorized preparing B3 source before reviewing B2/B3 together. Use `COLAB_B2_B3_COMBINED_AUDIT.md` as the current self-contained handoff. Earlier ZIPs and instructions are preserved; do not run multiple evaluation packages automatically. Source preparation does not sign owner releases, select/freeze a policy or authorize a held-out attempt.
