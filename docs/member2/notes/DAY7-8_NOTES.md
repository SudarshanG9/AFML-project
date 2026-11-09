# Day 7-8 notes: did rounding hurt, and does restoring one part fix it?

Files: `src/avfuse/member2_data/paired.py`, `scripts/m2_paired.py`, `tests/member2/test_paired.py`.

## The question in one line

Member 1 runs the **same detector** in four ways (configs):

| Config | What it is |
| --- | --- |
| `reference` | normal full precision |
| `low_precision` | fusion step rounded (simulated INT8) |
| `restore_value_output` | rounded, but the value/output part kept at full precision |
| `restore_query` | rounded, but the query part kept at full precision (the control) |

Day 7-8 answers: **how much did rounding hurt, and how much does each restore win back?**

## Idea 1: compare on the same clips ("paired")

All four configs score **exactly the same clips**. So instead of comparing two averages, we
compare clip by clip: "for this clip, how did the score change?"

**Why this matters:** some clips are easy, some hard. Comparing on the same clips cancels that
out, so only the effect of the config change is left. This catches much smaller differences.

The confidence interval is also paired: in each of the 1,000 re-draws, **the same groups** are
drawn for both configs (whole groups, as on Day 5-6).

A difference counts as real only if its 95% interval **does not include 0**.

## Idea 2: damage and rescue (validation only)

We measure everything with the loss **L** (BCE from Day 5-6; lower = better).

```
damage  = L(low_precision) - L(reference)        how much rounding hurt
R       = L(low_precision) - L(restored)         how much a restore won back
share   = R / damage                             fraction of the damage recovered
```

**Tiny example:** reference L = 0.20, low_precision L = 0.40, restored L = 0.25
- damage = 0.40 - 0.20 = 0.20
- R = 0.40 - 0.25 = 0.15
- share = 0.15 / 0.20 = **75% of the damage recovered**

"Positive evidence" for a restore = R > 0 **and** its interval doesn't include 0.

R is also split **per fake type**, to see *where* the rescue happens (e.g. mostly on audio fakes).

## Idea 3: cost, measured not assumed

Restoring the value/output part may cost more time than restoring the query part. So each
restore is also judged per **extra millisecond**, using Member 3's measured timings:

```
extra ms  = p50 time(restored config) - p50 time(low_precision)
R per ms  = R / extra ms
```

Rules: only timings from the **same device and same scope** are compared; if there are no
measured timings, the cost column stays empty. Timings are never typed in or guessed.

## The verdict (follows the plan's gates)

| What we see | Verdict printed |
| --- | --- |
| rounding did not hurt (damage <= 0) | report robustness; do **not** invent a rescue |
| a restore has positive evidence | restoration has label-grounded support |
| no restore has positive evidence | no rescue: report the negative finding |

**Why validation only?** This step decides which restore is "good". Deciding while looking at
test would make the final test result biased.

## Running it

```
python scripts/m2_paired.py --manifest <manifest.jsonl> --predictions <preds.jsonl> \
    --protocol protocol.json --benchmarks <member3 benchmark .jsonl> --out <folder>
```
Output: `paired.json` (all numbers) and `paired.md` (readable tables + verdict).

## Tests (synthetic data, no model needed)

| Test | Checks |
| --- | --- |
| `test_identical_runs_give_zero` | same scores twice gives a difference of exactly 0 |
| `test_low_precision_damage_detected` | a real drop is detected, with an interval above 0 |
| `test_restoration_table_identifies_value_output` | the helpful restore is found, the control is not; R, share and R per ms add up |
| `test_costs_from_mixed_devices_rejected` | timings from two different devices are refused |

## Status

Code done and tested. Real numbers wait for Member 1's predictions and Member 3's timings.

## Connection

- Uses Day 5-6 (loss, groups, threshold).
- Day 9 freezes the decision; Day 10 repeats these paired comparisons once on test.
