# 3. Research decisions and pivots

## Reading the timeline correctly

The original plan allocated twelve implementation/test days and two documentation days. “Day 8,” “Day B,” C9 and C10 are workflow and approval identifiers, not interchangeable calendar dates. The evidence spans preparations in October and early November, validation and benchmarking on 11–12 November, freeze and held-out recovery on 13–14 November, and final evaluation on 14 November 2026. Earlier notes remain evidence of what was known then; later acceptance supersedes their pending status. This chapter follows that order without retroactively treating the original plan as completed implementation.

The principal source anchors are the original PDF, [PLAN_12_DAYS.md](PLAN_12_DAYS.md), [Member 1's baseline inventory](member1/dayC/00_BASELINE_INVENTORY.md), [Day 8 method](member1/dayC/01_DAY8_METHOD_AND_REVIEW.md), [Day 9 decision](member1/dayC/02_DAY9_POLICY_DECISION.md), retained release records and the directly verified final handoff/evaluation packages indexed in [Chapter 10](10_EVIDENCE_AND_REFERENCES.md).

## Pivot 1: select a runnable pretrained detector

**Original intention.** AVFF was attractive because its representation learning explicitly used audiovisual correspondence rather than treating audio as an optional second classifier. A pretrained joint model would let the team change arithmetic while keeping learned weights fixed, avoiding the confound of training a new model for each precision condition.

**Evidence encountered.** Page 15 of the supplied PDF explicitly says official AVFF code/checkpoint availability was unverified at that search, with only an unverified community reimplementation reference. The assessment therefore treated checkpoint feasibility as an early gate. The proposal named wav2vec2 anti-spoofing/W2V2-AASIST audio models and FaceForensics++ visual classifiers as fallback components for a small new fusion head. The Elicit assessment also discussed AuViRe and other architectural comparators. Those are examined candidates or fallbacks, not evidence that each checkpoint was loaded and tested. No separate original pretrained-model assessment was available to recover a more exhaustive comparison.

**Decision.** Use the available DiMoDif detector rather than spend the project recreating AVFF or train the fallback head. The recorded DiMoDif source, model config and checkpoint supplied an inspectable and hashable pretrained path. The precise date and complete candidate-scoring deliberation are not retained in the supplied sources; assigning a more elaborate selection story would be speculation.

**Implementation.** `dimodif_adapter.py` loads the original upstream class and strict state dictionary, freezes parameters and preserves upstream alignment/scoring. The attention audit locates five joint self-attention layers. Zero-audio and zero-video pilot interventions change outputs, demonstrating input dependence on both streams in those samples.

**Consequence.** The research concerns DiMoDif, not AVFF reproduction. The target is joint self-attention over audiovisual tokens, rather than a standalone directional cross-attention module. The pilot ablations are feasibility checks, not proof that every later prediction relies on semantic cross-modal consistency.

## Pivot 2: use compatible whole LAV-DF videos

**Original intention.** FakeAVCeleb was the preferred dataset because its authentic/audio-fake/video-fake/both-fake categories map naturally onto the research question. Its request-based access was explicitly recognized as a potential schedule blocker.

**Evidence encountered.** The historical [dataset card](member2/DATASET.md) states that FakeAVCeleb access did not arrive in time. LAV-DF was accessible and carried modality-manipulation flags. DiMoDif supplied a LAV-DF DFD checkpoint, making dataset, preprocessing and pretrained detector compatible. These are two supported considerations; the records do not show that dataset compatibility alone caused the access delay or that all other datasets were rejected experimentally.

**Decision and implementation.** Member 2 first prepared short windows. The 9,200 selected records yielded 9,143 usable windows after two too-short and 55 edit-outside-window exclusions. Once the team used DiMoDif's whole-video path, [whole-video preparation](member2/notes/WHOLE_VIDEO_NOTES.md) retained 1,200 validation and 2,000 test videos, with all four categories balanced within each split. The final arrays are upstream-compatible speech encoder features, not the older four-second raw clips or generic PT placeholders.

**Consequence.** The final evaluation has 2,000 test videos, not the earlier 1,992 windowed examples. Whole-video processing prevents an edit being excluded merely because a short window misses it; the selected videos fit within the model's 600-frame limit. Source separation can be audited, but speaker separation cannot be verified because metadata lacks speaker IDs. The canonical manifest must retain its original bytes and hash.

## Pivot 3: use existing attention rather than write a custom kernel

**Original intention.** The scope-reduced plan already expected a likely no-go for custom tiling, based on anticipated modest token counts. It requested an actual shape/memory check before deciding.

**Evidence encountered.** Real padded attention was larger than the proposal's rough 1–2 MiB illustration: batch 1, four heads and 1,202 query/key positions require about 22.046 MiB for one FP32 dense score matrix per layer. Both modalities are padded to 600 positions with two special tokens. A structural mask preserves the pretrained behavior, including some valid-to-padded interactions and no separate padding mask at the audited boundary.

**Decision.** The [Day A decision](member1/DayA_TILING_DECISION.md) retained a dense mathematical oracle and PyTorch SDPA. A theoretical matrix estimate did not establish a bottleneck requiring another kernel; custom code would add masking, numerical and device-verification work to a precision study.

**Implementation and consequence.** The team tested oracle/SDPA and original-model parity, then instrumented the existing attention modules. No custom tiled kernel was produced. This is a justified scope choice, not proof that another kernel could never help. CPU SDPA evidence must not be called a measured FlashAttention GPU speedup, and the dense storage estimate is not measured peak memory.

## Pivot 4: narrow precision interventions and retain a query control

**Original intention.** The scoped PDF had already removed the broad combinatorial study of weights, accumulation, granularity and many bit widths. It chose one symmetric per-tensor simulated INT8 scheme and four configurations.

**Evidence and decision.** A simple activation round/dequantize function made exact boundaries inspectable while leaving pretrained weights intact. Native low-bit kernels would introduce another implementation and hardware axis. V/output was the motivated information-carrying restoration, while Q tested a different part of the computation. Equal cost remained a requirement to measure.

**Implementation.** Reference preserves Q/K/V/output. Low precision rounds all four. V/output restoration rounds Q/K only. Query restoration rounds K/V/output. The output boundary is before the original output projection. Softmax, projections, model parameters and arithmetic remain floating point. All five layers receive each policy.

**Consequence.** This is a controlled comparison of cumulative mode trajectories, not a separate per-layer sensitivity map or native INT8 deployment. V and output are restored together. Reliable equal costs were not established, and the eventual runtime differences cannot be converted into an equal-budget causal claim by assertion.

## Pivot 5: preserve unexpected validation findings

**Original expectation.** Low precision would degrade label-grounded performance, making restoration useful; V/output would then outperform Q. The plan permitted an honest negative outcome.

**Evidence encountered.** Day B established reference, low-precision and V/output validation outputs; the Day C B2 run on 12 November added complete query validation and three four-mode CPU workloads. Matched Member 2 analysis found validation damage \(L_{low}-L_{ref}=-0.0000431444\), with a 95% interval spanning zero. The direct Q-minus-V/output validation loss difference was also negative with an interval spanning zero. The low-precision configuration had not established positive damage to repair.

**Decision.** Continue the comparison and freeze a reference-preferred interpretation rather than tune until a positive story appeared. “Reference preferred” is a conservative reporting/policy choice, not a statistically proven win for reference over every alternative.

**Implementation.** The C9 policy fixes all four modes, threshold, checkpoint, canonical cohort and evidence identities. It records H2/H3 as not supported and retains limitations including validation checkpoint-selection exposure and historical test-label summaries.

**Consequence.** Held-out evaluation tests the frozen comparison; it does not select a restoration winner or rescue a failed positive hypothesis by searching more configurations. The original “if damage exists” condition matters: without demonstrated damage, a recovery fraction is scientifically unstable or undefined.

## Pivot 6: recover execution while retaining the failed attempts

**Original expectation.** One final four-mode held-out pass would follow approval. Execution was not that simple.

| Event | Original UTC timestamp | Meaning |
| --- | --- | --- |
| C9 approval | 2026-11-13 19:01:27.555408 | approval of the frozen scientific context |
| C9 freeze | 2026-11-13 19:01:35.345815 | reference-preferred policy and all four modes fixed |
| Parent C10 opened / failed | 2026-11-13 19:16:40.697804 / 19:16:41.372613 | missing feature/metadata file; no completed published predictions |
| R1 opened / interrupted | 2026-11-13 20:53:01.119599 / 21:09:46.097473 | KeyboardInterrupt; no completed published predictions |
| R2 opened / completed | 2026-11-14 02:53:23.461011 / 03:17:42.352402 | complete 2,000 predictions in each mode |
| C10R accepted | 2026-11-14 07:29:40.084277 | accepted final handoff identity |

These timestamps have explicit UTC offsets in the original ledgers. Add 05:30 for India time: R2 completed at 08:47:42 on 14 November, and C10R acceptance was at 12:59:40 that day. Member 2's later evaluation ledger uses timezone-naive 08:36:25 and 08:39:56; their offsets are unknown, so their clock times must not be compared directly to this UTC sequence.

**Decision and implementation.** Recovery required distinct authorization and preserved parent failure/interruption ledgers, locks and original inventories. The successful execution remains `C10_RECOVERY_R2_C10_20261113T191638908596Z`. An operational package named R3 does not create a fourth inference attempt. Output files bind run, split, mode and checkpoint and contain no labels. The accepted public handoff hash identifies the completed predictions; private recovery evidence retains additional history.

**Consequence.** There were three recorded attempts, not one uninterrupted execution. Zero completed outputs in earlier ledgers does not prove no transient computation occurred. Hashes and acknowledgements establish retained lineage, not a cryptographic guarantee that no undocumented activity ever occurred. The recovery changed execution availability, not the frozen policy or evaluation cohort.

## Pivot 7: final evaluation, evidence recovery and interpretation

Member 2 evaluated the saved four-mode test predictions on 14 November using the fixed threshold, 1,000 source-group bootstrap replicates and seed 0. A second ledger entry records a same-session trailing-silence control. Later evidence review initially lacked the original final evaluation package; that access-limited verdict was superseded when the supplied copy made its metrics, logs, ledger, figures and checksums directly inspectable. All eleven listed payload hashes match. This is verification of saved original outputs, not a fresh held-out evaluation.

The test BCE damage estimate changed sign relative to validation but its interval still spans zero. V/output did not show an overall advantage over Q or low precision. The accepted CPU model benchmark used fake quantization and did not establish a native INT8 speedup. Later runtime integration supports four modes even though older Segment E prose describes reference-only support.

The completed project therefore contributes a reproducible negative precision/restoration result within a specific checkpoint and cohort, plus explicit measurement and provenance infrastructure. It does not deliver the expected optimization win, a new detector, an equal-cost restoration policy, speaker-disjoint validation, or a raw-video production service. Preserving those boundaries is the final research decision, not an omission to be hidden by a more confident README.
