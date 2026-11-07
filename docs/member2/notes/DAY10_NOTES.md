# Day 10 notes: the final test report

Files: `scripts/m2_evaluate.py` (paired contrasts added), `src/avfuse/member2_data/report.py`
(paired tables), `tests/member2/test_seal.py` (one extra check).

## The job in one line

Run the frozen rules **once** on the sealed test set and produce the final numbers for all four
configs, plus the comparisons that answer our research question.

## What the final report contains

**1. Metrics for each config** (from Day 5-6), on test:
AUROC, AUPRC, loss, recall and false alarms at the frozen threshold, each with a confidence range,
and the same per fake type (audio / video / both).

**2. Four paired comparisons** (from Day 7-8), all on the same test clips:

| Comparison | Question it answers |
| --- | --- |
| `low_precision` vs `reference` | How much did rounding hurt? |
| `restore_value_output` vs `low_precision` | How much does restoring value/output win back? |
| `restore_query` vs `low_precision` | How much does restoring query (the control) win back? |
| `restore_value_output` vs `restore_query` | Is our targeted restore better than the control? |

A difference counts only if its 95% range does **not** include 0.

## How to read one row

Example: `low_precision - reference, recall: -0.08 [-0.12, -0.04]`
- the rounded model catches **8 percentage points fewer** fakes,
- the range is entirely below 0, so the drop is **real**, not noise.

If the range were, say, [-0.05, +0.02], it crosses 0, so we could **not** claim a difference.

## Outputs

- `metrics.json`: every number (for the paper's tables),
- `summary.md`: the same as readable tables, per config + the paired comparisons,
- test-access count from Day 9 (should be 1).

## Running it

Same command as Day 9 (with `--split test --unseal-test --purpose ...`). The paired comparisons
are added automatically for every config pair that has predictions.

## Tests

`test_test_split_is_sealed_and_logged` now also checks that the test report contains the paired
`low_precision` vs `reference` comparison.

## Status

Code done and tested on synthetic data. Real numbers come after Member 1's final test predictions.

## Connection

- Uses the frozen protocol from Day 9 and the methods from Days 5-8.
- Day 11 re-checks the data for shortcuts before the results are signed off.
