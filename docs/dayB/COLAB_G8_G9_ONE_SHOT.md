# One final Colab run: G8 and G9

The user supplied the G7 evidence, acknowledged G7 after its review, and explicitly authorized the remaining Day B gates as one batch with a single final review. G7 technical audit passed. This package does not require an intermediate G8 upload or acknowledgement.

## Scope

Run all 1,200 restore_value_output validation predictions with the original accepted CPU checkpoint/code, preserving the G5 reference and G6 lowp bytes and exact provisional threshold. Then produce three-mode metrics, per-type source-cluster CIs and paired changes, restoration recovery, matched model and original Day A operator benchmarks, G9 end-to-end completeness evidence and the Member 2 handoff. All bootstrap analyses use 1,000 resamples and seed 0. All timings use 10 warmups and 50 repeats.

**Day 8 equal-cost control comparison not yet evaluated.** Day B ends at G9 and contains reference, low_precision and restore_value_output. The later restore_query experiment, Day 9 final policy freeze and held-out test are outside this batch. restore_query_val.jsonl and all test predictions remain outstanding, rather than being fabricated or replaced with pilot results.

## Existing runtime only

Keep the existing Colab CPU runtime and the original manually mounted validation cache. The agent does not access Colab, mount Drive, connect accounts or retrieve remote features. Do not download the cache to the Mac.

- G5 code: `/content/AFML_G5_CPU_Colab_20261111T180139Z`
- G5 run: `/content/g5_results/G5_CPU_20261111T181935Z_422cf72f`
- G6 run: `/content/g6_results/G6_CPU_20261111T194903Z_6a2bf3c1`
- G7 run: `/content/g7_results/G7_CPU_20261111T202044Z_ce6a0493`
- Original .pth remains under the G5 run's `runtime/dimodif/ckpt/dfd/` with SHA-256 `e1e59006d1bb6ad36a3859af5486a6649ed41bba23258b7c41e24f873be7948e`.
- Exact provisional calibration SHA-256 remains `358c037242f6980fed2a8d0ff0ef830f17b4789676b391c8f6dff245842b37dd` (threshold 0.6627789735794068).

No installation, new checkpoint upload or runtime replacement is required. Expected versions remain Python 3.13.18, Torch 2.14.1+cpu, NumPy 2.1.3, jsonschema 4.23.0, scikit-learn 1.6.1, matplotlib 3.10.0 and pytest 8.3.4, with one Torch and one interop thread. Guards stop if the runtime, original manifests, cache paths or recorded assets differ. Missing directories are not permission to rewrite the ready manifest, reduce the cohort or rerun prior gates silently.

## Execute once

Upload `AFML_DayB_G8_G9_CPU.zip` via Colab's Files sidebar. Use the checksum-verified cell in the separately delivered **DAYB_FINAL_QUICKSTART.md**, which contains the exact released archive checksum. It extracts to a fresh `/content/AFML_DayB_G8_G9_CPU` directory and invokes:

```python
subprocess.run([
    sys.executable, '-B', '/content/AFML_DayB_G8_G9_CPU/scripts/dayB_finish/run_g8_g9.py',
    '--g5-root', '/content/AFML_G5_CPU_Colab_20261111T180139Z',
    '--g5-run', '/content/g5_results/G5_CPU_20261111T181935Z_422cf72f',
    '--g6-run', '/content/g6_results/G6_CPU_20261111T194903Z_6a2bf3c1',
    '--g7-run', '/content/g7_results/G7_CPU_20261111T202044Z_ce6a0493',
], check=True)
```

The program prints stage progress and a fresh FINAL_RUN_DIR. It validates input packages/assets and G5/G6/G7 provenance before new inference. It preserves all existing outputs and performs:

1. Original unchanged full restore_value_output inference over all 1,200 approved validation clips. The original adapter enforces finite logits and scores per clip.
2. Three-file schema, checkpoint, split, exact-ID and category checks; original prediction/orientation checker; three-mode evaluation with the exact G5 calibration.
3. Fresh three-mode cached-model benchmarks on the pre-approved short, typical and long inputs. The typical set alone is `dayB_model_matched.jsonl`; short/long stay separate. All records must be schema-valid and identical in workload identity, shapes, device, backend, dtype, environment and sampling across modes.
4. Original paired recovery evaluation with ONLY the typical three-mode benchmark set. Nonpositive damage makes share-recovered unavailable; nonpositive added latency makes recovery-per-ms unavailable. No positive restoration narrative is assumed.
5. Original Day A real Q/K/V operator timings, both SDPA and dense-oracle backends, all three Day B modes. These raw reports stay separate from schema-v0.2 model benchmark JSONL. Fake quant/dequant executes inside timing; there are no integer kernels. CPU peak memory is null/not_measured, never zero. No raw-media latency claim.
6. Full per-type paired AUROC/AUPRC/BCE/recall/FPR differences and BCE recovery CIs using the existing bootstrap engine.
7. G9 completeness/hash report; final execution evidence table; exact config definitions and Member2_DayB_handoff.zip. It records the pre-existing limitations and local test results without inventing human interpretation.

## One final upload

The printed **FINAL REVIEW ZIP: .../DayB_G8_G9_review.zip** is the only evidence ZIP to return. Download it manually from the Files sidebar. It includes the three full prediction files, fixed calibration, three-mode metrics/plots, paired and per-type results, model/operator benchmarks, G4/G5/G6/G7 provenance, local test evidence, logs, hashes, status and a nested Member 2 handoff ZIP. It excludes feature arrays, original weights, credentials and raw media.

The handoff includes `reference_val.jsonl`, `low_precision_val.jsonl`, `restore_value_output_val.jsonl`, configuration definitions, calibration, metrics, recovery evidence, one matched benchmark set, README, provenance and checksums. The agent will audit the returned final ZIP before claiming Day B complete. The current prepared package is not a completed G8/G9 research run.

On failure, preserve the notebook traceback and download the printed final review ZIP; available failure evidence is bundled after run creation. If a guard fails before run creation, return the traceback. Do not skip clips, flip scores, relax thresholds or overwrite prior results to force a pass.

Stop after this batch. No intermediate human gate is inserted, but computational integrity failures still stop execution. Do not begin the later query-control, policy-freeze or held-out-test stages.

## Scientific context retained

G6 found no established positive quantization damage. Whole-model restoration is cumulative across five layers, and deeper-layer upstream inputs can differ. The separate G7 diagnostic establishes matched target inputs only for its single approved clip and target arms, not population forensic accuracy. Validation checkpoint-selection bias, source-versus-speaker grouping limitations, independently unverified training overlap and trailing-silence/other shortcut risks remain open. Final conclusions will be reviewed once all returned batch evidence is available.
