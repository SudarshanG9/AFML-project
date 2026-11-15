# 5. Dataset and experimental methodology

## Study unit and dataset choice

The experimental unit is a selected whole LAV-DF video with one detection score per precision mode. It is not a frame, a four-second crop, or an independent pair of audio/video observations. All four modes operate on the same selected inputs and frozen pretrained checkpoint. This pairing makes small arithmetic effects interpretable without introducing differences in training or sample selection.

FakeAVCeleb was the original candidate because its four modality combinations fit the question directly. Delayed access and the available DiMoDif/LAV-DF path led to the final choice. [LAV-DF's official repository](https://github.com/ControlNet/LAV-DF) distributes a content-driven audiovisual forgery benchmark with access terms and a CC BY-NC 4.0 project license. The historical data handoff identifies the same noncommercial research restriction. Users must follow the dataset's terms for media and derived features; a repository checkout does not provide redistribution permission. No media or full feature cache is newly published by this cleanup.

The final task deliberately reduces localized manipulations to whole-video authenticity. An authentic video has label 0. Setting only the audio manipulation flag gives `audio_fake`; setting only video gives `video_fake`; setting both gives `both_fake`, all with label 1. This is a category analysis of one binary detector, not a trained four-class classifier or temporal localization evaluation.

## Why the canonical manifest matters

Earlier preparation selected 9,200 windows, of which 9,143 remained after duration/edit-window exclusions. Its validation/test counts were 1,195/1,992. These numbers describe a superseded preparation, not missing predictions in the final experiment. The whole-video manifest retains the full selected validation/test cohorts:

| Split | Real | Audio fake | Video fake | Both fake | Total | Source groups |
| --- | --- | --- | --- | --- | --- | --- |
| Validation | 300 | 300 | 300 | 300 | 1,200 | 1,114 |
| Test | 500 | 500 | 500 | 500 | 2,000 | 1,738 |

These counts come from the hash-matched original manifest/split reports, not newly recomputed test-label statistics. The canonical byte identity is `8339b2476a71e62c6c6fca7d937ba4cc7a704b09d26f08506c20d1f10de8f3dd`. The separate validation projection has hash `22d1d9e02403cfdbb05f2116da16bc4015f540547a4ac00bf6b7127c119e359c`. Neither the old window manifest nor a 2,000-row media mapping CSV can substitute for this identity.

A `clip_id` identifies one selected video and its cache directory. `source_id` links an original and its derivatives; connected source/identity groups are kept together by the split/evaluation helpers. LAV-DF metadata does not provide verified speaker IDs, so `identity_id` equals `source_id`. The original audit reports zero groups crossing validation/test. The train-overlap report checks 78,703 training videos and 21,254 sources and reports zero selected validation/test source overlap. These checks address shared recordings; the same speaker could still occur through distinct recordings. The upstream checkpoint also used LAV-DF development data for model selection, making our validation findings selection-exposed.

## Offline extraction and cached inputs

Member 2 prepares whole-video mappings and runs the same speech-related encoder families used by DiMoDif: auto-AVSR visual `LRS3_V_WER19.1` and audio `LRS3_A_WER1.0`. The visual path uses MediaPipe landmarks and the auto-AVSR loader/crop; the audio path uses that library's audio loading conventions. The earlier windowed preprocessing used 25-fps video and mono 16-kHz audio, but its four-second stored clips and two-second model window are not the final whole-video protocol. The separate extraction runbook preserves the compatible Python 3.10/PyTorch 2.5.1-era dependency route.

Each selected video yields `features.npz` with FP32 `video_features (T_v,768)` and `audio_features (T_a,768)`, plus `meta.json`. Lengths are stored separately because the encoders can differ slightly. The model truncates to their minimum and 600, then pads both modalities to 600. The selected whole-video report records no video exceeding the 600-frame budget. Caching removes repeated encoder/decoding work from the precision comparison and fixes the input representation across modes.

The historical `feature_path` value is a PT placeholder. The actual NPZ path is assembled from an explicit root, split and clip ID. Both filename and mapping are part of provenance, not cosmetic choices. Extraction provenance binds manifests, mapping reports and original environments. The saved coverage report verifies directory/metadata coverage for all 3,200, but explicitly states that full array-load/finite/shape verification reached 2,800 before interruption. Later successful inference demonstrates usable inputs for that execution; it does not retroactively supply an independent historical per-feature byte baseline for every original extraction.

## Frozen checkpoint and intervention design

The accepted pretrained LAV-DF DFD checkpoint is frozen: no fine-tuning, new head, retraining or weight quantization is part of the experiment. The intervention changes dynamic symmetric per-tensor activation precision at Q/K/V and the pre-output-projection attention output, cumulatively in all five joint layers. [Chapter 4](04_FINAL_SYSTEM_ARCHITECTURE.md) specifies every arithmetic boundary. A single quantizer avoids conflating granularity, calibration strategy, bit width and native backend changes.

The four modes provide reference, broad low-precision, V/output restoration and query restoration comparisons. Inputs, pretrained parameters, masks and score aggregation remain fixed. Bundling V and output means the experiment cannot distinguish those contributions. Cumulative intervention means modes are matched at external input, not at every intermediate hidden state. This limits mechanistic interpretation even when a paired configuration effect is well defined.

## Fixed operating point and sealed-test discipline

Member 2 calibrated a threshold using **reference validation scores only**, targeting at most 5% false positives among authentic validation videos. The implementation sorts authentic scores, allows at most \(\lfloor0.05n_{real}\rfloor\) positives, and places the threshold just above the next score with `nextafter`. Predictions use `score_fake >= threshold`. The frozen threshold is **0.6627789735794068**, applied to every mode and to test without retuning. This is a fixed operating-point comparison; calibrating each mode independently would answer a different question.

C9 freezes policy, mode definitions, checkpoint, canonical cohort and evidence identities before final scoring. Historical test-label-derived summaries had already been inspected and were disclosed. The record therefore supports separation from test-score tuning, but not a claim that absolutely no test-derived information had been seen. Member 1 inference and Member 2 label-based evaluation have separate access histories.

The accepted parent C10 failure, interrupted R1 and completed R2 remain preserved. The successful return has exactly 2,000 rows per mode, common run/checkpoint, unique canonical IDs and finite probabilities. Member 2's final package logs one final four-mode evaluation followed by a same-session reference silence control in two ledger entries. Software flags and append-only conventions discourage accidental access; hashes and logs cannot prove absence of undocumented external access. This documentation task never unseals, reruns inference or invokes the held-out evaluator.

## Metrics and source-grouped uncertainty

AUROC measures ordering of fake scores against real scores. AUPRC is implemented as **average precision**, not trapezoidal integration. BCE measures mean negative log probability assigned to the true labels, clipping probabilities to \([10^{-7},1-10^{-7}]\) for numerical safety. Recall is the fraction of fakes at or above the threshold; real FPR is the fraction of authentic examples at or above it. Higher AUROC/AP/recall is favorable; lower BCE/FPR is favorable. AP depends on class prevalence, which differs between the overall 3:1 fake/real cohort and the category-versus-real subsets.

The evaluator draws whole connected source groups with replacement, retaining all selected members whenever a source is drawn. Final outputs use 1,000 replicates, seed 0 and percentile 95% intervals. Group resampling respects within-source dependence better than resampling clips independently. Category counts can vary across replicates: this is not fixed-count stratification. All recorded valid replicate counts in the final metric blocks are 1,000.

Paired comparisons align IDs across modes and use the same sampled source indices for both conditions. For each replicate the metric difference is computed directly; its interval therefore captures the correlation between two modes on the same clips. Comparing overlap of two separate confidence intervals is not a substitute. Differences are always labeled \(b-a\). For BCE, a positive low-minus-reference value is damage, while a negative restoration-minus-low value is improvement.

Each manipulation analysis compares its 500 test fakes with all 500 real videos. Per-type recall intervals use the corresponding fake-only groups, while rank metrics include the real group. The three subsets share real examples and are correlated. Four contrasts across five metrics and overall/three categories yield 80 saved paired intervals. Their nominal 95% levels are not multiplicity-controlled, and a category result alone does not establish a between-category interaction.

## Shortcut control and runtime protocol

The retained control matches equal real/fake counts within ten trailing-silence bins, then evaluates the saved reference scores on the resulting subset. It tests whether a particular shortcut can explain performance within that selection, without retraining or changing the detector. Matching reduces the shortcut's discriminative value but also changes the evaluated population. Duration, speaker effects and other artifacts remain uncontrolled; this is not proof of causal independence from silence.

Runtime measurement is separate from accuracy evaluation. The accepted Member 3 CPU benchmark has three validation anchor workloads, batch one, ten warmups and fifty timed repeats per mode. Tokens are resident before timing; model preparation/padding, attention/head and applicable fake-quant conversions are inside the timed call. Linear p50/p95 summaries come from the saved samples. The original input lengths differ but all are padded to the same 1,202 attention positions. Isolated operator timing excludes projections/full model; offline preprocessing is reported separately. CPU peak memory is unmeasured. These boundaries permit a descriptive cost comparison without claiming native INT8 acceleration or an end-to-end raw-media latency.
