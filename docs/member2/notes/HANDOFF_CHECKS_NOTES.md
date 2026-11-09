# Notes: checks after Member 1's Day 1 handoff

## What Member 1 confirmed
- Checkpoint: DiMoDif DFD LAV-DF, `dfd_lavdf_reduceonplateau_64_4_1_5_True.pth`, sha256 `e1e59006...7948e`.
- Our `features.npz` files are used **as they are**: `video_features (T_v, 768)` and `audio_features (T_a, 768)`, float32.
  He aligns them to min(T_v, T_a, 600) and pads to 600. No `.pt` conversion is needed,
  so `feature_path` in the manifest is only a placeholder.
- Pilot clip `2961dde91d90fee1` (val, both_fake): video 193 x 768, audio 195 x 768, 193 valid positions.
- Same 4 configs and the same prediction fields as our contract. First file: `reference_val.jsonl` (1,200 val videos).

## His task list vs what already exists
| His request | Where it is |
|---|---|
| Join predictions to the whole-video manifest by clip_id | `predictions.join` (Day 5-6): errors on missing/unknown clips, never drops |
| Schema validation, duplicate/missing checks | `predictions.validate_predictions` + `join` |
| AUROC, AUPRC, recall per fake type vs real | `evaluate.evaluate_config` (Day 5-6) |
| Val-only threshold, source-cluster bootstrap CIs | `m2_evaluate.py calibrate` + `cluster_ci` |
| Keep test sealed | `seal.py` (Day 9) |
| Check the score's direction | **new**: `scripts/m2_check_predictions.py` |
| Overlap with the checkpoint's training data | **new**: `scripts/m2_train_overlap.py` |

## The two new checks, in simple words
**Score direction.** `score_fake` should be high for fakes. If the model's scoring is the wrong way round,
AUROC comes out well below 0.5. The check reports `ok`, `INVERTED` or `unclear` for each config on val.
If it is inverted, Member 1 fixes his scoring; we never flip scores ourselves, because that would hide a bug.
It also confirms that the checkpoint hash is the agreed one and that all 1,200 val videos have a prediction.
Test rows are only counted, never scored.

**Training overlap.** The checkpoint learned from LAV-DF's train split. If one of our val/test videos was made
from the same real original as a train video, the model may have seen that face and voice already,
which would make results look better than they are. The check looks up every evaluation video's source in train.
Limits: LAV-DF has no speaker IDs, so the same person in two different source videos can't be detected;
val is LAV-DF dev, which was used to pick the checkpoint (already listed as a limitation).

## Order once reference_val.jsonl arrives
1. `m2_check_predictions.py` (contract + direction)
2. `m2_evaluate.py calibrate` (freeze the threshold on reference val)
3. `m2_evaluate.py evaluate --split val`
4. Later configs: re-run 1 and 3 with all files; then `m2_paired.py` and `m2_shortcut_control.py`.
