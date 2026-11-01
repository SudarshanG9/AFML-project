# Experimental protocol and limitations

## Four *frozen* configurations

- **reference**: original Q, K, V, and fusion output.
- **low_precision**: symmetric per-tensor fake quantization of Q/K/V and fusion output.
- **restore_value_output**: same low-precision Q/K, original pre-quant V, unquantized fusion output.
- **restore_query**: original pre-quant Q, low-precision K/V, low-precision fusion output.

These preserve the report's intended configurations but **do not automatically form an equal-cost pair**. Check tensor sizes, restore costs, and report recovery per measured overhead. Separate the value restoration from output restoration in additional validation experiments *only if the protocol is revised before test freeze*.

## Sampling and labels

Four groups: real; audio_fake; video_fake; both_fake. Identity and source-family groups must not cross splits. Check whether the pretrained checkpoint saw test samples during its original training. Check input duration/silence/metadata shortcuts.

## Metrics

For each fake class, compute the binary discriminator against **real** samples in the same split. Report AUROC, average precision/AUPRC, and recall at a single validation-fixed threshold (also report false-positive rate on real). Use source- or identity-cluster bootstrap for CIs and paired contrasts. Avoid interpreting AUROC when there are not both positive and negative cases.

## Freeze rules

Use validation data for selecting the threshold, deciding if restoration is promising, and any control-cost normalization. Freeze code/config/checkpoint/manifest/hash before final held-out test scoring. No further tuning against test outcomes.

## Runtime

Time isolated operator and cached-feature model separately. Synchronize accelerator around every measured iteration, warm up, and record device/backend/version/dtype/shapes, p50/p95 and supported peak memory. Report offline preprocessing separately. Simulated INT8 is NOT a real integer kernel and may have higher wall-clock cost.

## Current incomplete sections

The scaffold includes working low-level attention/quantization primitives, a source-group validator, lightweight evaluation helpers, and benchmark harness. It does **not** include verified real-model adapter, licensed dataset decoder/face crop, checkpoint download, calibrated evaluation pipeline, cluster bootstrap CIs, or any measured benchmark/report. These require real inputs and integration work.
