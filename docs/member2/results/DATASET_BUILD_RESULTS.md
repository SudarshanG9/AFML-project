# Dataset build results: LAV-DF (Member 2)

Run on Google Colab (CPU), 2 Nov 2026, with the code of the Day 3-4 commit:
`scripts/m2_lavdf.py` (select + clips) and `scripts/m2_build_manifest.py`.
Only totals are recorded here. Clips, audio, video and the manifest itself stay off git
(LAV-DF is CC BY-NC 4.0, research use only).

## Summary

- **9,200** videos selected and decoded (1,500 / 300 / 500 per type for train / val / test)
- **57** excluded: 2 too short, 55 fakes whose edited words fall outside the 2-second evaluation window
- **9,143** usable clips in `manifest.jsonl`, validated against the shared schema
- **Split audit passed** using LAV-DF's official splits: no source video appears in more than one split, so no re-split was needed

## Clips per split and type

| Split | real | audio_fake | video_fake | both_fake | Total | Source groups |
| --- | --- | --- | --- | --- | --- | --- |
| train | 1,500 | 1,498 | 1,458 | 1,500 | 5,956 | 5,205 |
| val | 300 | 300 | 295 | 300 | 1,195 | 1,111 |
| test | 500 | 499 | 493 | 500 | 1,992 | 1,733 |
| **Total** | 2,300 | 2,297 | 2,246 | 2,300 | **9,143** | 8,049 |

"Source groups" = distinct source videos. A group holds a real video and the fakes edited from
it, and always stays inside one split.

## Exclusions by type

| Split | audio_fake | video_fake | Total excluded |
| --- | --- | --- | --- |
| train | 2 | 42 | 44 |
| val | 0 | 5 | 5 |
| test | 1 | 7 | 8 |
| **Total** | 3 | 54 | **57** |

No real or both_fake clips were excluded. **54 of the 57 exclusions are video_fake**: for those, the
edited segment more often lies near the start or end of the video, so the centred 2-second window
cannot contain it. The categories stay balanced (smallest: video_fake, 97% of target).

## Checks

| Check | Result |
| --- | --- |
| Test set identical to the earlier Colab run (28 Oct) | ✅ same counts (500 / 499 / 493 / 500 = 1,992), confirming the selection is reproducible |
| Official LAV-DF splits source-disjoint | ✅ for this sample (0 groups crossing splits) |
| Manifest valid against `contracts/schemas/manifest.schema.json` | ✅ `scripts/validate_manifest.py`: 9,143 rows, no source/identity leakage |

## Limitations

- `identity_id = source_id` (LAV-DF has no speaker IDs): splits are source-disjoint, not guaranteed speaker-disjoint.
- Labels come from LAV-DF's official metadata and were not checked by hand.

## Still pending

- **Leakage / shortcut audit** on these clips (Day 11 code; the audio needed is saved).
- **Encoder feature cache** (`data/cache/<clip_id>.pt`): waits on Member 1's checkpoint choice.
- **Checkpoint training-data overlap check**: waits on the same choice.
