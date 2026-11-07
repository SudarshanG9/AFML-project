# Notes: whole-video manifest for DiMoDif

Files: `src/avfuse/member2_data/wholevideo.py`, `scripts/m2_whole_video_manifest.py`,
`tests/member2/test_wholevideo.py`.

## Why this exists

The team chose **DiMoDif** as the detector. DiMoDif reads a **whole video** (up to 600 frames,
about 24 s at 25 fps), not a 2-second window. LAV-DF videos are at most 20 s, so every video fits.

So for DiMoDif, one manifest row = **one whole video**, not one 4-second clip.

## What stays the same

- **The same videos**: picked with the same caps and seed as the clip pipeline (Days 3-4).
- The same labels (Day 1), the same source groups and split audit (Day 2), the same shared schema.

## What changes

| | Clip manifest (Days 3-4) | Whole-video manifest (this) |
| --- | --- | --- |
| Unit | 4 s clip, model sees 2 s | the full video |
| Fakes with the edit outside the window | excluded (55) | **kept**: a whole video always contains its edit |
| Splits | train / val / test | **val + test only**: DiMoDif is already trained, so no train features are needed |
| IDs | `md5(path@start)` | `md5("whole:" + path)`, a separate set so the two can never be mixed up |

## What the script writes (into `data/`, never git)

- `manifest_whole.jsonl`: the shared-schema manifest (one row per whole video)
- `videos_whole.csv`: the raw video paths, durations and frame counts, i.e. the input list for
  feature extraction
- `manifest_whole_report.json`: counts, split audit, shortest/median/longest video, and how many
  are longer than 600 frames (expected: 0)

`--check-files` also confirms every raw video is on disk. Use it after downloading and before
feature extraction.

## Tests

| Test | Checks |
| --- | --- |
| `test_same_videos_as_clip_pipeline_and_no_exclusions` | exactly the same videos as the clip pipeline; a fake whose edit is at 0.1 s is kept |
| `test_ids_never_collide_with_clip_ids` | a real video's whole-video ID differs from its clip ID |
| `test_script_writes_valid_manifest` | the script writes a schema-valid manifest; `--check-files` stops when videos are missing |

## Next

Feature extraction: run DiMoDif's encoders on every video in `videos_whole.csv` and save
`data/cache/<video_id>.pt`.
