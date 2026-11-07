# Day 5-6 notes: scoring the detector honestly

Files: `src/avfuse/member2_data/predictions.py`, `src/avfuse/member2_data/evaluate.py` (new part),
`src/avfuse/member2_data/report.py`, `scripts/m2_evaluate.py`, `tests/member2/test_evaluate.py`,
`tests/member2/synthetic.py`.

## The job in one line

Take the model's scores (from Member 1) and our labels (from the manifest) and answer:
**how well does each model version catch each type of fake?**

## How the pieces fit

```
predictions (Member 1)  ---+
  clip_id, config,         |--> join on clip_id --> metrics per config --> summary.md
  score_fake (0-1)         |                          (+ per fake type)
manifest (Member 2)     ---+
  clip_id, label, type
```

The model's prediction file **never contains labels**. Labels come only from our manifest.
That way nobody can accidentally tune the model on the answers.

## Step 1: Check the predictions (`predictions.py`)

Before any number is computed, every prediction row is checked against the shared contract.
It is **rejected** if:
- a field is missing, or `score_fake` is not a probability between 0 and 1,
- it carries any extra field such as `label` (so labels can't sneak in),
- the same clip is predicted twice for the same config,
- predictions from two different checkpoints are mixed in one evaluation.

Then predictions are matched to manifest rows by `clip_id`. It **stops with an error** if a
clip is missing a prediction, if a prediction has no clip, or if the split doesn't agree.
Clips are never silently dropped, because dropping hard clips would make the scores look better.

## Step 2: The numbers (`evaluate.py`)

| Metric | In simple words |
| --- | --- |
| **AUROC** | Pick one fake and one real clip at random: how often does the fake get the higher score? 1.0 = always, 0.5 = coin toss. |
| **AUPRC** | How clean the top of the ranking is: are the highest-scored clips really fakes? |
| **BCE loss** | How confident and correct the scores are. Lower is better. Used later to measure "rescue". |
| **Recall** | Out of all fakes, how many get caught at our chosen threshold. |
| **Real FPR** | Out of all real clips, how many get wrongly flagged as fake. |

### Per fake type
Each fake type (`audio_fake`, `video_fake`, `both_fake`) is compared **against all real clips**.
That shows if, for example, the low-precision model gets worse only at audio fakes.

## Step 3: Choosing the threshold (on validation only)

A score of 0.7 means nothing until we decide "above what score do we call it fake?".

Rule: on the **validation** set, pick the threshold so that only **5% of real clips** get flagged.
That threshold is then **frozen** in `protocol.json` and used unchanged for every config.

**Why validation and not test?** If we picked it while looking at test, the test result would
be tuned to the test set and would no longer be honest.

**Why one threshold for all four configs?** It is fixed from the `reference` model. If the
low-precision model shifts its scores, the fixed threshold shows how many more fakes it now misses.

## Step 4: Confidence intervals (cluster bootstrap)

A single number like "AUROC = 0.91" hides how sure we are. So each metric gets a range,
e.g. 0.91 [0.88, 0.94].

How: recompute the metric 1,000 times, each time on a random re-draw of the data, and take the
middle 95% of the results.

**The important twist:** clips are re-drawn in **whole groups** (the Day 2 groups: a real video
plus the fakes made from it), not one clip at a time. Those clips are near-copies, so counting
them separately would make us look more certain than we are.

## Running it

```
python scripts/m2_evaluate.py calibrate --manifest <manifest.jsonl> --predictions <preds.jsonl> --out protocol.json
python scripts/m2_evaluate.py evaluate  --manifest <manifest.jsonl> --predictions <preds.jsonl> \
    --protocol protocol.json --split val --out <folder>
```
Output: `metrics.json` (all numbers) and `summary.md` (readable tables).

## Tests (on synthetic data, no model needed)

| Test | Checks |
| --- | --- |
| `test_threshold_hand_example` | threshold on a 10-clip example worked out by hand |
| `test_threshold_never_exceeds_target` | the 5% limit is never broken (200 random cases) |
| `test_cluster_resamples_keep_groups_whole` | bootstrap always draws whole groups |
| `test_evaluate_config_matches_sklearn_and_strata` | AUROC/AUPRC match scikit-learn; per-type uses all real clips |
| `test_predictions_contract` | bad rows (label inside, score > 1, duplicates, mixed checkpoints, missing clips) are rejected |
| `test_cli_calibrate_and_evaluate` | the script runs end to end and reports all 4 configs |

The synthetic predictions are clearly marked (`checkpoint_sha256 = SYNTHETIC-NOT-A-CHECKPOINT`)
and are never results.

## Status

Code done and tested. Real numbers wait for Member 1's prediction files.

## Connection

- Uses Day 1 (labels and types) and Day 2 (groups for the confidence intervals).
- Day 7-8 compares configs **on the same clips** to measure damage and rescue.
