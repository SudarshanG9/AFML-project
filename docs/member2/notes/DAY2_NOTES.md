# Day 2 notes: keeping train, val and test apart

Files: `src/avfuse/member2_data/splits.py`, `tests/member2/test_splits.py`.

## The problem in one line

If two very similar clips end up on **both sides** of the train/test line, the test score is
too good to be true: the model has effectively already seen the answer.

## Example

Real video **R** is edited into fake **F**. They look almost the same.

- R in train, F in test: the model memorised R's face and voice, so F looks "familiar" at test
  time. Its test score goes up even though it did not truly detect the fake.
- R and F both in test: fair.

So the rule is: **clips that are related must stay in the same split.**

## What counts as "related"

Two clips are related if they share:

- the same **source video** (`source_id`), or
- the same **person** (`identity_id`).

And it chains: if A shares a source with B, and B shares a person with C, then A, B and C are
all one **group**. They must all go in the same split.

## What the code does (3 functions)

| Function | In simple words |
| --- | --- |
| `group_clusters(rows)` | Puts every clip into its group, following the chains above. |
| `split_audit(rows)` | Counts clips per split and type, and reports any group that appears in more than one split. `passed = True` means no group crosses splits. |
| `resplit_by_group(rows)` | Fixes a bad split by moving whole groups. |

### How the fix works

1. If a group has **any** clip in test, the **whole group goes to test**.
   The test set only grows; it is never reshuffled, so it stays untouched.
2. If a group is split between train and val, it goes wholly to one of them. The choice uses a
   fixed hash of the group name, so everyone gets the same answer on every machine.
3. Groups that were already in one split are not touched.

## Tests

| Test | Checks |
| --- | --- |
| `test_groups_are_transitive` | the chain rule: A-B by source + B-C by person = one group |
| `test_transitive_leak_is_caught_and_repaired_without_redrawing_test` | a hidden leak is found, fixed, test only grows, an untouched clip stays where it was |
| `test_resplit_is_deterministic` | same input gives the same split every time |

## One-line answers

- **Why not just split clips randomly?** Related clips would land in different splits and inflate the score.
- **Why move leaking groups into test instead of train?** So the test set is never redrawn after we have seen it.
- **Limitation:** LAV-DF has no person IDs, so `identity_id = source_id`. Groups are by source video only.

## Connection

- Day 1 defined `source_id` and `identity_id`. Day 2 uses them to check the splits.
- The same groups are reused on Days 5-10 for confidence intervals: the bootstrap resamples whole groups, not single clips.
