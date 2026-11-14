# Original Day 8 method and review — source preparation only

Status: **SOURCE CHECKS PASS / REAL PILOT BLOCKED / C8A NOT READY**.

The user acknowledged C0 and subsequently requested continuation. This work prepares the C8.1 diagnostic source and synthetic tests. The complete Day 8 entry conditions are still unsatisfied: the full approved validation cache is not accessible inside the clone and the current runtime differs from the accepted Day B CPU environment. No original Day 8 experiment, real pilot, population analysis or benchmark has been executed. C0 acceptance does not grant C8A or held-out authorization.

## Implemented source

- `src/avfuse/member1_fusion/dayc_precision_audit.py`: the exact four-mode boundary table, trace-summary validation, frozen-model/checkpoint/architecture checks, reference parity and state/input/method cleanup diagnostics. It adds no inference mode, detector, kernel, quantizer or changed production attention mathematics.
- `tests/member1/test_dayc_query_math.py`: independent operand-level oracles and five-layer synthetic attention fixtures. The oracle quantizer is test-only and does not call the production `prepare_qkv` or `fake_quant_int8` to obtain expected values.
- `scripts/dayC_member1/query_audit.py`: guarded pilot orchestration with C0/file-hash checks, full-cohort local-cache readiness, exact runtime comparison, validation membership, unchanged adapter reuse, original reference parity, three diagnostic modes, 20 trace rows each and no-clobber reports. No bypass flags.
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

The full Member 1 suite passed **165 tests, 0 failures, 0 skips**: the prior 103 tests, 54 numerical/diagnostic tests and eight pilot orchestration/guard tests. The run used the existing local Python/PyTorch installation and made no dependency changes. These results establish implementation behavior on TEST_FIXTURE tensors, not forensic accuracy or full-model pretrained DiMoDif parity.

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
- Bind the implemented guarded `scripts/dayC_member1/query_audit.py` to actual approved inputs after readiness is resolved. The library parity helper requires caller-verified validation membership; tensor contents cannot establish the split. The CLI has no public synthetic-fixture or guard-bypass switches.
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

## Guarded pilot runner verification

The runner was implemented after the user committed the prior source work at `ca37c87479486555734a179c028ed22151790747`. The reviewed file hashes still match C0; the runner records the current HEAD separately and does not mistake a user commit with unchanged reviewed bytes for a numerical implementation change. No user commit/index state was reset or rewritten.

Synthetic orchestration tests check one adapter construction and one feature load, all three diagnostic arms with 20 rows each, input/state preservation, rejection of unreleased or held-out input, and errors from nonfinite scores or mutation. A separate ordering test verifies that cohort/runtime blockers prevent even resolving the requested feature.

The actual metadata-only CLI check returned **BLOCKED, exit 2**, with exactly the missing full local cache and runtime mismatch as blockers. `real_model_inference_started=false`; no detector was constructed or feature content read. This was a readiness check, not a failed or completed research pilot.

Latest evidence: `artifacts/dayC/member1/day8/PILOT_RUNNER_SOURCE_20261112T085649Z_c2a25a/` (`member1.xml`, `tests.json`, `actual_entry_check.json`, command/log records and `implementation_evidence.json`). C8A remains not ready.

## Approved Colab CPU pilot arrangement

The user subsequently explicitly approved the existing Colab CPU runtime and recorded Drive cache for C8.2, with a local package for manual execution. That narrow authorization supersedes the prior local-runtime-only blocker for this pilot; it does not establish that a live Colab session is accessible or change the original accepted paths/versions. The agent has not accessed the service. Use `COLAB_C8_2_PILOT.md` and its exact checksum-pinned package; return real evidence before C8A. The local-only `query_audit.py` remains strict and has not acquired a hidden external-path bypass.

New source: `scripts/dayC_member1/colab_query_pilot.py`, `build_colab_pilot_package.py`, `configs/dayC/member1/colab_pilot_release.json` and `tests/member1/test_dayc_colab_pilot.py`. The numerical parity helper now returns already-computed native/instrumented scores without adding forward calls or altering production math. The current full Member 1 suite is **189 passed, 0 failed, 0 skipped**. No real pilot or population experiment has been executed locally.

## B1 continuation under the batched controller and approved version drift

The user reported v1 preflight failure at `versions` and supplied exact differences (NumPy 2.4.6, jsonschema 4.26.0, pytest 8.4.2). The old review ZIP has not been returned. The user explicitly approved pilot-only recorded version drift with Torch unchanged. Original v1 release/archive/quickstart and the user-reported failed run are retained; no old evidence is rewritten.

Use the separate `COLAB_C8_2_PILOT_DRIFT_V2.md` handoff and v2 archive. It enforces exactly those transitions, records observed runtime on failure, and compares the typical native score with the stored Day B anchor before diagnostic arms. The complete local Member 1 suite now has **204 passed, 0 failed, 0 skipped**. B1 remains `BLOCKED: AWAITING USER-RUN COLAB EVIDENCE`; no real pilot, C8A approval, full query validation, policy freeze or held-out stage has been completed by this continuation.


## B1 C8.2 fresh validation mount and exact identity handoff

The user reported the cache available at `/content/afml_drive/MyDrive/AFML_Project/dimodif_emb/val` and explicitly requested read-only mapping of obsolete original paths. `COLAB_C8_2_PILOT_MAPPED_V3.md` is the current manual handoff. The original ready manifest, context, G4 evidence, checkpoint and numerical source remain unchanged. Before loading the fixed pilot, v3 requires every approved validation clip and all 2400 feature/metadata hashes to match accepted G4 evidence. The test directory is not enumerated or read. 244 local Member 1 tests passed with no failures or skips. Actual Colab identity/parity/trace evidence is still pending; no full query inference or C8A gate has completed. Earlier dated blocker descriptions above are preserved history, superseded only by the explicitly approved Colab arrangement and latest handoff.


## C8.2 returned real-model evidence: technical PASS, C8A pending

The user returned `C8_2_pilot_review (3).zip` (SHA-256 `960b93c6ba93e9c17536bd718ad591ad2f5c0a465ac7241183e274c43e759a9a`). All 1200 approved IDs and all 2400 feature/metadata hashes match accepted G4 evidence; original paths are preserved through the approved read-only mapping. Exact packaged source, release and runtime hashes reconcile. Native/instrumented reference parity and exact Day B reference score consistency pass, as do all three 20-row diagnostic traces and reported state/input/cleanup checks. This establishes one real-model validation pilot, not population benefit or measured cost. Complete evidence and the explicit proposed B2 runtime amendment are recorded at `/Users/sudarshan/Documents/GitHub/AFML-project/artifacts/dayC/member1/day8/C8_2_RETURN_AUDIT_20261112T120751551450Z/B1_C8A_REVIEW.md`. B1 is technically complete and halted at mandatory human C8A; no full validation, policy freeze or test stage has started.


## C8A accepted; B2 full-block source preparation

The user supplied `ACK DAYC C8A: AUTHORIZE FULL VALIDATION QUERY CONTROL`, bound to `4b0063628fe44161aceae1a8b782c8f65240c41feb97157cd24fc94ce027933e`, and reiterated five implementation-block reviews (B1–B5). B2 C8.3–C8.7 source preparation now includes the guarded unchanged-runner wrapper, read-only loader bridge, exact full-cohort/benchmark validators, source-only research reader and sign/undefined-cost tests. The complete Member 1 suite is 291 passed, no failures/skips; synthetic 1200-loop delegation and packaged runtime rejection pass. Use `COLAB_DAY8_B2_C8A.md` for the single manual inference/cost handoff. C8.3 real outputs, C8.4 actual costs and Member 2/3 complete evidence are still pending; no full validation or benchmark ran locally. Existing Member 2 resampling does not explicitly preserve strata and complete per-stratum paired intervals are absent; `B2_MEMBER2_MEMBER3_HANDOFF_REQUIREMENTS.md` records the unresolved owned work without changing their code. No C8B request, Day 9 policy freeze or held-out work occurs before the entire B2 packet is complete.


## B2 returned preflight-only failure; exact-original recovery handoff

The returned B2 ZIP SHA-256 is `6eb1bf4e6abb5870371ce69f53e607040f91368c63b00d50f986bb25be9b80b3`. Its only run record reports missing original canonical manifest at the approved `/content/AFML_G5_CPU_Colab_20261111T180139Z/data/manifests/manifest_whole.jsonl` path during preflight. No full-validation predictions, benchmark records or new scientific result were returned. The source-verified runner checks that input before model launch. `COLAB_B2_PREFLIGHT_RECOVERY.md` provides one exact-byte restore-and-continue cell using the previously preserved original-input ZIP, verifying no model-attempt markers first and preserving all failure evidence. Original B2 runner/protocol/source and C8A context remain unchanged. 19 focused recovery/restoration tests pass; B2 remains externally blocked, and no C8B review is requested.


## B2 full inference and cost evidence received; evaluation-only completion prepared

The successful returned ZIP SHA-256 `ff71f1549b1c63850757378290b03f38de0f1241767047eb1f20d42b65308185` verifies all 1200 query predictions and 2400 original cache hashes, exact input/source/approval/runtime and eight restored originals, plus 12 matched benchmark records with all 600 raw measured samples. Exact query bytes were exclusively imported to `artifacts/predictions/restore_query_val.jsonl` (SHA-256 `855a90bf1a72c59043ac3e80f04ec15df16d10ea0b98056ff10c56d751d5dbf1`). Typical p50 reference/lowp/V-output/query are 118.4543895/161.434481/119.91815/167.4740195 ms. E_value is -41.516331 ms and E_query 6.0395385 ms; comparative recovery per positive extra millisecond is undefined. No result is forced into a positive-cost interpretation.

The accepted Day B per-type routine resolves the earlier incomplete-CLI concern without editing Member 2 code. The remaining `COLAB_B2_EVALUATION_ONLY.md` job uses existing owner APIs for overall and per-fake-versus-real paired CIs, 1000 replicates/seed0, original fixed threshold, and explicit disclosure that category counts fluctuate within group resamples. It performs no inference/benchmark rerun. Local Member 1 suite: 315 passed, 0 failed/skipped. No real CI/population metric is claimed before returned evaluation output. Audit and input hashes are at `/Users/sudarshan/Documents/GitHub/AFML-project/artifacts/dayC/member1/day8/B2_RETURN_AUDIT_20261112T160215000947Z`. B2 continues; its single C8B review and all B3/held-out work remain pending.


## User-authorized joint B2/B3 source preparation completed

The user explicitly requested implementing B3 before joint B2/B3 review. The source-preparation receipt is recorded without inventing C8B/C9/C10 acknowledgements. Day9 preflight/decision/release/proposal/finalize guards and the required source-only held-out verifier/runner/audit/handoff implementation are complete. Full Member1 suite: 376 passed, 0 failed/errors/skipped. An actual local validation-only Day9 preflight reports the expected missing statistics/owner/final-policy dependencies, with no ready-test row or feature access. No final freeze or real held-out ledger exists.

Use `COLAB_B2_B3_COMBINED_AUDIT.md` and its self-contained source/score/metadata capsule for one joint audit job. It preserves original bytes and can evaluate the saved validation scores without old Colab folders, Drive, weights, feature tensors, inference or benchmark reruns. It includes original canonical metadata for the unchanged Member2 val-only/split audit; no ready-test manifest or test feature data is included. The returned DAYC_B2_B3_review.zip will contain real CIs/plots and an unapproved B3 proposal with unresolved owner/freeze prerequisites explicit. Earlier separate B2-only handoffs remain preserved history and are superseded for the next execution.


## Joint returned B2/B3 audit — actual statistics complete, lead review pending

The joint bundle SHA-256 `e12c866890ee5d1eb66455f86400009d64bb717cdced2d92649bb00328327cc6` passes complete input/source/runtime/hash/coverage checks. All four modes use the same 1200 clips and original threshold, 1000 source-cluster replicates/seed0, with overall and complete fake-versus-real contrasts. D=-0.00004314435345702944 (CI [-0.0006248129777084077, 0.0004649242890685698]); Rv=-0.000043916609145489005 (CI [-0.00047609479379089465, 0.00042285118253031326]); Rq=0.00016738907786786883 (CI [-0.00044803103368849047, 0.0007681759813717088]); direct query-minus-value BCE=-0.00021130568701335783 (CI [-0.0010083570282118776, 0.0006097277131264898]). All primary and per-type BCE intervals span zero. No positive precision damage, recovery advantage, direct superiority or equivalence is established. Share recovery and the planned positive-extra-time comparison are undefined. Preserved DayB points/CI endpoints agree in 45 checks; no local CI/model rerun occurred.

The proposed interpretation is FP32 reference primary with all four modes retained as experimental contrasts, subject to lead review. See `/Users/sudarshan/Documents/GitHub/AFML-project/artifacts/dayC/member1/day9/JOINT_RETURN_AUDIT_20261112T204538343795Z/B2_B3_JOINT_REVIEW.md` for the complete joint audit, owner release drafts and exact packet hashes. B3 source remains tested (376 passing), but actual freeze remains blocked by genuine Member2/Member3 acknowledgements and the opaque ready-test identity. No C8B/C9/C10 ACK was inferred from the successful run or file upload.
