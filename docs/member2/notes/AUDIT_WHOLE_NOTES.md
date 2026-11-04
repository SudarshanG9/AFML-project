# Notes: shortcut audit on whole videos

Files: `scripts/m2_audit_whole.py`, `src/avfuse/member2_data/shortcuts.py` (whole-video part),
`tests/member2/test_audit_whole.py`.

## Why a second audit

Day 11's audit checked the **4-second clips**. DiMoDif uses **whole videos**, so the shortcut check
has to run on exactly what DiMoDif sees.

Good news first: the main expected shortcut disappears. In the clip pipeline, real clips started
at 0 s while fakes were cut around the edit. With whole videos, **both are used from start to end**,
so there is no "cut position" difference left.

## What is measured per video

| Feature | Why it could leak |
| --- | --- |
| silence at the start / end, loudness | edited audio may have been re-mixed |
| duration, number of frames | fakes may be longer or shorter |
| video / audio codec, sample rate, fps, resolution | fakes may have been re-encoded differently |
| **bitrate, file size** | re-encoding a fake often changes how many bits it uses |

The audio is read straight from each `.mp4` (mono, 16 kHz, same settings as before), and the
file details come from `ffprobe`.

## Same rule as Day 11

For each feature: "using only this, how well can we tell real from fake?" (shortcut AUROC).
**>= 0.65 is flagged**, per split, overall and per fake type. A flag must be reported. It doesn't
prove the model uses the shortcut, but it means we need to check.

## Running it (CPU is enough)

```
python scripts/m2_audit_whole.py --videos data/manifests/videos_whole.csv \
    --manifest data/manifests/manifest_whole.jsonl --out artifacts/reports/audit_whole --workers 2
```
Output: `shortcut_audit.json` and `audit.md` (tables per split and type, plus the list of flags).

## Tests (on real tiny `.mp4` files made by ffmpeg)

| Test | Checks |
| --- | --- |
| `test_features_from_real_file` | 0.5 s of leading silence is measured as about 500 ms; codecs, resolution and bitrate are read |
| `test_flags_silence_shortcut_on_whole_videos` | if fakes get extra silence on purpose, the audit flags it |
| `test_script_runs_end_to_end` | the script reads the files and writes the report |

## The follow-up control check (`scripts/m2_shortcut_control.py`)

If a feature is flagged, we test whether the **detector** depends on it:

1. Sort the videos into 10 groups by that feature (e.g. trailing silence).
2. In each group, keep the **same number** of real and fake videos.
   Now the feature looks the same in both classes and can no longer tell them apart.
3. Measure the detector's AUROC on this matched set and compare it with all videos.

- AUROC **stays about the same**: the detector isn't using the shortcut.
- AUROC **drops a lot**: part of its score came from the shortcut. Report it.

The report also shows the feature's own AUROC on the matched set (it should be about 0.5),
which proves the matching worked. `m2_audit_whole.py` saves `video_features.csv` (one row per
video) as the input for this step.
