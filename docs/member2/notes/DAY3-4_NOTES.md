# Day 3-4 notes: turning LAV-DF videos into clips and a manifest

Files: `src/avfuse/member2_data/lavdf.py`, `src/avfuse/member2_data/manifest.py` (new part),
`scripts/m2_lavdf.py`, `scripts/m2_build_manifest.py`, `tests/member2/test_lavdf.py`.

## The job in one line

Take raw LAV-DF videos and produce two things the team can use:
**short, aligned clips** and a **manifest** (one row per clip saying what it is).

## The pipeline

```
metadata.json -> pick clips -> cut 4 s clips -> clips.csv -> check -> manifest.jsonl
   (Day 1 labels)  (sample)    (ffmpeg)        (index)      (Day 2 groups)
```

## Step by step

**1. Read the metadata (`index_lavdf`)**
One record per video: its split (LAV-DF's `dev` becomes our `val`), its type
(using the Day 1 mapping), its source video, and where the fake words are.

**2. Pick a balanced sample (`sample_per_split`)**
At most 1,500 / 300 / 500 clips per type for train / val / test.
Each split is sampled separately, so the test set is the same whatever we pick for train.
This is exactly the same selection as the earlier Colab run, so the 1,992 test clips already
saved are still valid.

**3. Download only what we need (`m2_lavdf.py select`)**
LAV-DF is 25.6 GB. This writes the list of the ~9,200 chosen files, so the download keeps only those.

**4. Cut the clips (`extract_clip`, `m2_lavdf.py clips`)**
With ffmpeg, each video becomes:
- `frames.npy`: video at 25 frames per second, 96x96 pixels (the face is already centred), and
- `audio.npy`: mono sound at 16,000 samples per second.

Both are cut from the **same start time**, so sound and picture stay in sync.

**5. Where to cut**
- Real clips: from the start of the video (0 s).
- Fake clips: **centred on the fake words**. Only a few words are fake, so a clip cut anywhere
  else might contain no fake part at all.

**6. Throw out clips that can't be used (`long_enough`, `window_covers_fake`)**
The model looks at a 2-second window inside each 4-second clip. A clip is dropped if:
- it is too short, or
- it is fake but the fake words fall outside that 2-second window
  (otherwise it would be a "fake" label on real content).

The script prints how many were dropped, so the numbers go in the report.
On the test split: 1 too short, 7 with the edit outside the window, 1,992 kept.

**7. Write the manifest (`m2_build_manifest.py`)**
Turns `clips.csv` into `manifest.jsonl` in the exact format of the team contract, runs the Day 2
split check, and refuses to write the manifest if a group crosses splits
(unless `--resplit` is given, which fixes it).

## Two files, two purposes

| File | What it holds | Who uses it |
| --- | --- | --- |
| `clips.csv` | paths to the `.npy` clip files | only Member 2 (it's an internal index) |
| `manifest.jsonl` | `clip_id`, split, label, type, groups, `feature_path` | the whole team |

## What is NOT done yet (honest status)

`feature_path` points to `data/cache/<clip_id>.pt`, the **encoder features** for each clip.
Those can only be made once Member 1 picks the model, because the features must be made the
way that model expects. Until then, the clips are ready but the feature cache is not.

## Tests

| Test | Checks |
| --- | --- |
| `test_index_maps_types_sources_and_cuts` | types, shared source between a fake and its original, cut times (also when the metadata gives frames instead of seconds) |
| `test_sampling_caps_per_split_category` | caps are respected, and the test sample doesn't change when the train cap changes |
| `test_window_rules` | too-short clips and "edit outside the window" clips are rejected |
| `test_build_manifest_script` | the script drops the bad clip, writes a valid manifest, and reports the count |

## One-line answers

- **Why cut fakes around the fake words?** Only a few words are fake; a random cut could miss them.
- **Why the same start time for audio and video?** Otherwise sound and lips drift apart, which
  would look like a fake even on a real clip.
- **Why drop clips?** A "fake" clip with no fake content in view would teach and test the wrong thing.

## Connection

- Uses Day 1 (label mapping) and Day 2 (split check).
- Days 5-6 read `manifest.jsonl` to attach labels to the model's predictions.
