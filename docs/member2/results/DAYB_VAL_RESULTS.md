# Day B validation results (reference, low_precision, restore_value_output)

Independent Member 2 check of Member 1's Day B handoff. Totals only; no clip-level data.

## Inputs
- Predictions: `reference_val.jsonl`, `low_precision_val.jsonl`, `restore_value_output_val.jsonl` (1,200 val videos each)
- Manifest: `manifest_whole.jsonl`, sha256 `8339b247...`. This is the same file Member 1 used (checked).
- Checkpoint: DiMoDif LAV-DF, sha256 `e1e59006...7948e` (one value on all 3,600 rows)
- Config definitions (Member 1): symmetric dynamic per-tensor fake INT8 in all five joint AV self-attention layers.
  `low_precision` = Q, K, V and attention output in INT8; `restore_value_output` = V and output back to FP32.

## Checks
| Check | Result |
|---|---|
| sha256 of the 11 handoff files vs `SHA256.json` | all match |
| Contract (schema, duplicates, every val video predicted, one checkpoint) | passed |
| Score direction | ok for all 3 configs (median score real 0.0001, fake 1.0000) |
| Threshold recomputed from reference val (5% real FPR) | 0.662779, the same as Member 1's |
| Metrics recomputed vs Member 1's `metrics.json` | identical (max difference 0) |

## Results (val, 300 real / 900 fake, 1,114 source clusters, 1,000 cluster-bootstrap resamples)
| Config | AUROC [95% CI] | AUPRC | Recall @ 0.6628 |
|---|---|---|---|
| reference | 0.9984 [0.9972, 0.9993] | 0.9995 | 0.9933 |
| low_precision | 0.9984 [0.9972, 0.9993] | 0.9995 | 0.9933 |
| restore_value_output | 0.9984 [0.9972, 0.9993] | 0.9995 | 0.9933 |

Reference AUROC per fake type vs real: audio_fake 0.9993, video_fake 0.9959, both_fake 0.9999.

Paired differences (same videos): every CI contains 0. low_precision - reference: dAUROC +0.00001, dBCE -0.00004 [-0.0006, +0.0005].
So simulated INT8 attention causes **no measurable damage** on val, and restoring V/output has nothing to recover
(R and the share recovered are undefined because the damage is not positive). Reading: robustness, not rescue.

## Trailing-silence control
| | all videos | matched subset (520 of 1,200) |
|---|---|---|
| trailing silence alone (shortcut AUROC) | 0.682 | 0.502 |
| detector AUROC, reference | 0.998 [0.997, 0.999] | 0.999 [0.997, 1.000] |

The matching removes the shortcut (0.682 -> 0.502), and the detector's AUROC does not drop, which is the same for all 3 configs.
So the detector's performance is not explained by trailing silence.

## Caveats
- Val = LAV-DF dev, which was used to select the checkpoint, so val AUROC may be optimistic. Final claims come from the sealed test split.
- Simulated INT8 (float arithmetic), no integer kernels.
- Timings in `dayB_model_matched.jsonl` are not reliable for cost: samples within one run jump between ~110 and ~170 ms,
  and reference (p50 168 ms) is slower than low_precision (119 ms). "R per ms" isn't reported.
- Still outstanding: `restore_query` on val, the Day 9 freeze, test predictions.
