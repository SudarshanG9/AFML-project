# Validation results, all 4 modes

Added restore_query to the Day B results. Same 1,200 val videos, same frozen threshold (0.6628), 1,000 bootstrap resamples, seed 0.

All four files passed the checks (contract, one checkpoint, score direction ok).

| Mode | AUROC | AUPRC | Recall |
|---|---|---|---|
| reference | 0.9984 | 0.9995 | 0.9933 |
| low_precision | 0.9984 | 0.9995 | 0.9933 |
| restore_value_output | 0.9984 | 0.9995 | 0.9933 |
| restore_query | 0.9984 | 0.9995 | 0.9933 |

AUROC CI is [0.9972, 0.9993] for every mode.

## Paired differences (BCE, the most sensitive one)
- low_precision vs reference: -0.0000 [-0.0006, +0.0005]
- restore_value_output vs low_precision: +0.0000 [-0.0004, +0.0005]
- restore_query vs low_precision: -0.0002 [-0.0008, +0.0004]
- restore_value_output vs restore_query: +0.0002 [-0.0006, +0.0010]

Every CI crosses 0, and AUROC, AUPRC and recall don't move at all.

## What this means
INT8 attention doesn't hurt the model on val, so neither restore has anything to fix.
We can't say V/output matters more than Q (or the other way round) from this data.
The honest result is "robust to INT8 attention", not "restoring X rescues it".

Trailing-silence control for restore_query gives the same answer as the other modes: the detector stays at 0.999 on the matched 520 videos.

## Notes
- No timing column: the Day B CPU timings were too noisy to use.
- Val is the dev split that was used to pick the checkpoint. The test run decides the final claims.
