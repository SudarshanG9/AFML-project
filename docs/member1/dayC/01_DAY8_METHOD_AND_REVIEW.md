# Original Day 8 method and review — source preparation only

Status: **SOURCE CHECKS PASS / REAL PILOT BLOCKED / C8A NOT READY**.

The user acknowledged C0 and subsequently requested continuation. This work prepares the C8.1 diagnostic source and synthetic tests. The complete Day 8 entry conditions are still unsatisfied: the full approved validation cache is not accessible inside the clone and the current runtime differs from the accepted Day B CPU environment. No original Day 8 experiment, real pilot, population analysis or benchmark has been executed. C0 acceptance does not grant C8A or held-out authorization.

## Implemented source

- `src/avfuse/member1_fusion/dayc_precision_audit.py`: the exact four-mode boundary table, trace-summary validation, frozen-model/checkpoint/architecture checks, reference parity and state/input/method cleanup diagnostics. It adds no inference mode, detector, kernel, quantizer or changed production attention mathematics.
- `tests/member1/test_dayc_query_math.py`: independent operand-level oracles and five-layer synthetic attention fixtures. The oracle quantizer is test-only and does not call the production `prepare_qkv` or `fake_quant_int8` to obtain expected values.
- This method/review document. All files and test outputs are inside the selected GitHub clone.

## Four numerical definitions

| Mode | Q | K | V | Attention output before unchanged FP32 projection |
|---|---|---|---|---|
| reference | original FP32 | original FP32 | original FP32 | FP32 |
| low_precision | fake INT8 | fake INT8 | fake INT8 | fake INT8 |
| restore_value_output | fake INT8 | fake INT8 | original FP32 | FP32 |
| restore_query | original FP32 | fake INT8 | fake INT8 | fake INT8 |

The existing symmetric per-tensor quantize/dequantize uses dynamic max-absolute scale divided by 127 and signed rounded/clamped values. Tensors and surrounding projections stay FP32. All five existing joint audio-video self-attention layers are included; no new cross-attention model or integer kernel is introduced.

## Verified synthetic results

The full Member 1 suite passed **157 tests, 0 failures, 0 skips**: the prior 103 tests plus 54 new numerical/diagnostic tests. The run used the existing local Python/PyTorch installation and made no dependency changes. These results establish implementation behavior on TEST_FIXTURE tensors, not forensic accuracy or full-model pretrained DiMoDif parity.

New checks include:

1. Hand-checked signed rounding, a single tensor-wide scale and all-zero behavior.
2. Four independent Q/K/V/output oracles, with both SDPA and the independently calculated dense formula, odd token/head dimensions, boolean/additive masks, padded columns and a fully blocked row.
3. Every boundary compared to the independent oracle at all five synthetic layers: exactly 20 trace records per mode. Query preserves Q; K, V and attention output follow fake quantization. Dtype, shape, weights, input tensors and original attention methods remain unchanged.
4. Summary rejection for duplicate boundaries, wrong mode, invalid shape/dtype, nonfinite diagnostics, changed preserved Q and impossible quantization error bounds. Exactly representable or zero quantized tensors may have zero error.
5. Synthetic reference parity with specified logit tolerances (atol 1e-5, rtol 1e-4) and score tolerances (atol 1e-6, rtol 1e-5). Cleanup is checked both after success and an injected exception; another test raises during forward. Concurrent/nested diagnostic use of one model is rejected.
6. Fail-closed model checks for training mode, trainable weights, checkpoint mismatch, wrong heads, active monkey-patching and wrong layer count.

**Trace-summary limitation:** errors, changed fractions and shapes cannot by themselves prove exact quantizer equality. `validate_precision_trace` explicitly reports this limitation. The separate operand-level tests compare actual before/after tensors to independent expected tensors. No unavailable tensors are inferred from summary statistics.

## Still required before C8A

- Resolve all 1,200 approved validation feature paths inside this clone; the existing historical ready manifest resolves 0 locally. Do not substitute the available pilot or rewrite the canonical manifest.
- Restore an approved comparable runtime or obtain an explicit, concrete research protocol decision on environment differences. The local Mac/PyTorch 2.10.0 suite is not the accepted Linux/PyTorch 2.14.1+cpu research environment.
- Implement the guarded `scripts/dayC_member1/query_audit.py` orchestration once its manifest/runtime entry requirements can be bound to actual authorized inputs. The library parity helper requires caller-verified validation membership; tensor contents cannot establish the split.
- Execute the actual approved validation pilot once, with the original checkpoint and adapter. Require reference full-model parity, all-five-layer real traces, original scores, state/feature integrity and code/runtime/checkpoint provenance.
- Then present real pilot evidence and the planned full validation/cost protocol for C8A. No C8A request or approval was generated in this source-only preparation.

## Planned full validation and cost review, not executed

After C8A, use the unchanged Day B runner for exactly 1,200 query scores, with identical approved IDs and four categories of 300. Preserve the old three prediction hashes. Require unchanged Member 2 schema/evaluation/source-group bootstrap interfaces (1,000 resamples, seed 0) and a direct paired V/output-versus-query contrast.

Member 3 owns matched four-mode cost evidence for the approved short, typical and long cached-validation workloads. Keep typical as the primary normalization cohort, retain raw timings and use the accepted 10 warmups / 50 measured repetitions. Include fake-quant/dequant and restoration overhead. CPU peak memory stays null/not_measured when unsupported. Cached-model timing excludes raw-video extraction.

**Equal cost is not established.** Q restoration and V/output restoration bypass different boundaries. Actual added p50 latency relative to low precision must be measured on the same device, environment and workload. Normalize descriptively only with strictly positive reliable denominators. Never substitute theoretical dimensions or old three-mode timing for query-arm cost.

## Research interpretation remains for the lead

Prior Day B damage is nonpositive and paired damage/recovery CIs include zero; do not claim established harm, rescue or equivalence. H3 requires a direct paired contrast, coherent evidence and measured costs. The all-five-layer intervention is cumulative, so deeper hidden activations may differ; the tests do not prove matched upstream activations or population causality.

Retain the original limitations: validation checkpoint-selection bias; source grouping not independently verified speaker separation; unverified checkpoint-training overlap; silence/duration shortcuts; previously disclosed test-label-derived summary inspection; fake INT8 not hardware integer acceleration. No held-out input, labels or scores were read or produced by this preparation.

## Local evidence

`artifacts/dayC/member1/day8/SOURCE_ONLY_20261112T063939Z_55a670/`

- `member1.log`, `member1.xml`, `tests.json`: actual command, exit status and test results.
- `source_preparation_evidence.json`: file hashes, C0 approval reference, preservation and explicit outstanding dependencies.

No staging, commits, pushes, branch changes, production-code fixes, cache changes or real research executions were performed.
