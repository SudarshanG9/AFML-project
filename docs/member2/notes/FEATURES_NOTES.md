# Notes: DiMoDif feature extraction

Files: `scripts/m2_dimodif_features.py`, `src/avfuse/member2_data/dimodif_features.py`,
`tests/member2/test_dimodif_features.py`.

## The job in one line

Turn each whole video into two lists of numbers, one per frame, that DiMoDif's detector reads:
**video features** (from the lips) and **audio features** (from the speech).

## How it works, per video

| Stage | What happens |
| --- | --- |
| 1. Landmarks | MediaPipe finds the face in every frame |
| 2. Video crop | the mouth region is cut out, aligned and turned grey (96x96) |
| 3. Audio load | sound is loaded and resampled to mono 16 kHz |
| 4. Video encode | a frozen **lip-reading** model (auto-AVSR, `LRS3_V_WER19.1`) turns the mouth frames into features |
| 5. Audio encode | a frozen **speech-recognition** model (auto-AVSR, `LRS3_A_WER1.0`) turns the sound into features |

These are exactly the two models DiMoDif was trained with, so its official checkpoint can read
our features directly.

**Why lip-reading + speech models?** DiMoDif's idea: in a real video, what the lips "say" and what
the audio "says" match. In a fake, they drift apart. These two models describe both in the same
"speech language", so the detector can compare them.

## What gets saved (in `data/`, never git)

```
data/dimodif_emb/<split>/<video_id>/features.npz   video_features (T_v, D)  audio_features (T_a, D)
data/dimodif_emb/<split>/<video_id>/meta.json      shapes, frame counts, timings, status
data/dimodif_emb/feature_report.json               summary for the whole run
```
`features.npz` uses the same names DiMoDif uses.

**T_v and T_a are recorded separately and not assumed equal.** DiMoDif cuts both to the shorter
one and pads to 600 frames.

## Two deliberate differences from DiMoDif's own script

1. **No transcription.** DiMoDif's script also writes out the spoken text (slow), but its
   detector never uses it. Skipping it saves a lot of time and doesn't change the features.
2. **No facenet-pytorch.** It's only needed for an alternative face detector, and it forces an old
   PyTorch. We use MediaPipe, DiMoDif's default.

## Safe to stop and restart

Each finished video is skipped next time, so a Colab disconnect only loses the video in progress.
A video that fails (e.g. no face found) is recorded as `failed` with the reason, and the run continues.

## The report (for Member 3 and the results)

`feature_report.json` gives: how many videos worked or failed, feature sizes, the T_v / T_a ranges
and their difference, frames without a detected face, and **per-stage timings**
(min / median / 95th percentile / max) with the GPU and library versions.

## Tests

| Test | Checks |
| --- | --- |
| `test_pending_resumes_and_filters` | finished videos are skipped; `--split` and `--limit` work |
| `test_summary_reports_shapes_alignment_and_failures` | the report counts ok/failed, shapes, T_v vs T_a, timings |

The model part needs a GPU and the two auto-AVSR models, so it is checked on Colab with one
video first.
