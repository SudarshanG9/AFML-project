# Final test results

This is the one held-out evaluation. Totals only, no clip-level data.

## How it was run
- Predictions: Member 1's run `C10_RECOVERY_R2`, 4 modes x 2,000 test videos. I checked all the file hashes and confirmed that each file has exactly the 2,000 canonical test IDs before opening the test split.
- Manifest: `manifest_whole.jsonl` (sha `8339b247...`), `split == test`: 500 real, 500 audio_fake, 500 video_fake, 500 both_fake, 1,738 source groups.
- Threshold: 0.6627789735794068, frozen on val (protocol sha `358c0372...`). It was not changed.
- Bootstrap: 1,000 resamples, seed 0, whole source groups. Per fake type = that type + all real videos.
- Code: commit `70e3a57`.
- Test seal: opened once on 2026-11-14 (ledger has 2 lines from the same session: the evaluation, then the trailing-silence control).

Member 1's inference took 3 attempts. The first failed before scoring anything (one feature file was missing from his Drive copy), the second was stopped by hand, and the third finished. No test scores existed before the third attempt.

## Results
| Mode | AUROC [95% CI] | AUPRC | BCE | Recall | Real FPR |
|---|---|---|---|---|---|
| reference | 0.9980 [0.9969, 0.9989] | 0.9994 | 0.0678 | 0.992 | 0.042 |
| low_precision | 0.9980 [0.9969, 0.9989] | 0.9993 | 0.0681 | 0.992 | 0.042 |
| restore_value_output | 0.9980 [0.9969, 0.9989] | 0.9993 | 0.0682 | 0.992 | 0.042 |
| restore_query | 0.9980 [0.9969, 0.9989] | 0.9994 | 0.0677 | 0.992 | 0.042 |

Reference, per fake type vs real:
| Fake type | AUROC | Recall |
|---|---|---|
| audio_fake | 0.9990 | 0.994 |
| video_fake | 0.9955 | 0.984 |
| both_fake | 0.9996 | 0.998 |

Video-only fakes are the hardest, same as on val.

The val threshold carried over well: real FPR on test is 4.2% [2.5%, 6.1%] against the 5% target.

## Did INT8 change anything?
Paired BCE differences (same videos):
- low_precision vs reference: +0.0003 [-0.0002, +0.0010]
- restore_value_output vs low_precision: +0.0001 [-0.0004, +0.0005]
- restore_query vs low_precision: -0.0004 [-0.0010, +0.0001]
- restore_value_output vs restore_query: +0.0005 [-0.0002, +0.0012]

AUROC, AUPRC, recall and FPR don't move in any comparison.

I ran 80 paired comparisons in total (4 contrasts x 5 metrics, overall and per fake type).
Only 1 has a CI that excludes 0: restore_value_output vs restore_query, BCE on audio_fake, +0.0012 [+0.0002, +0.0025].
With 80 comparisons at 95%, about 4 would do that by chance alone, the effect is tiny, and nothing else changes with it, so I don't read anything into it.

So the test result matches val: INT8 attention doesn't hurt this detector, and neither restore has anything to recover.
This means "robust to simulated INT8 attention", not "restoring V/output (or Q) rescues it".

## Trailing-silence control (reference)
| | all 2,000 | matched 912 |
|---|---|---|
| trailing silence alone (shortcut AUROC) | 0.660 | 0.506 |
| detector AUROC | 0.998 [0.997, 0.999] | 0.997 [0.996, 0.999] |

Matching removes the shortcut and the detector barely moves (0.001), so its accuracy isn't coming from trailing silence.

## Limits
- Simulated INT8 (float arithmetic), all five attention layers at once. Not a hardware speed claim.
- Source groups aren't verified speaker separation (LAV-DF has no speaker IDs). Source-level overlap with LAV-DF train is 0.
- Summaries derived from test labels (counts, source overlap, the media shortcut audit) were looked at before the freeze. No test model scores were.
- Duration wasn't controlled.
- No multiple-comparison correction; the single CI excluding 0 above should be read with that in mind.

Figures: `test_figures/curves.png`, `test_figures/recall_by_type.png`.
