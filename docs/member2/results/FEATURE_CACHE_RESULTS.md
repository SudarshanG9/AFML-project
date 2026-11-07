# Feature cache results: DiMoDif encoders on LAV-DF (Member 2)

Run on Google Colab, 4 Nov 2026, with `scripts/m2_dimodif_features.py` over the whole-video
manifest (`scripts/m2_whole_video_manifest.py`, val 300 + test 500 per type).
Only totals are recorded here. The features themselves (4.0 GB) stay off git, on the team Drive.

## Summary

- **3,200 / 3,200 videos extracted, 0 failed** (val 1,200, test 2,000)
- Encoders: DiMoDif's own frozen auto-AVSR models, visual `LRS3_V_WER19.1` and audio
  `LRS3_A_WER1.0`, MediaPipe face tracker
- Output per video: `features.npz` with `video_features` and `audio_features`, DiMoDif's own format

## Shapes and alignment

| Quantity | min | median | p95 | max | mean |
| --- | --- | --- | --- | --- | --- |
| T_v (video feature frames) | 70 | 182 | 413 | 497 | 212.6 |
| T_a (audio feature frames) | 72 | 184 | 414 | 497 | 213.7 |
| T_v − T_a | −3 | −1 | 0 | +1 | −1.07 |
| T_v − video frames | 0 | 0 | 0 | 0 | 0 |
| frames without a detected face | 0 | 0 | 0 | 0 | 0 |

- Feature size: **768** for both modalities, `float32`.
- **One video feature per video frame (25 fps)** for every video.
- **T_v and T_a are not always equal** (they differ by up to 3 frames). DiMoDif truncates both to
  min(T_v, T_a) and zero-pads to 600. Consumers must not assume equal lengths.
- **0 videos exceed DiMoDif's 600-frame maximum**, so no video is truncated by the model.
- A face was found in every frame of every video.

## Per-stage timing (per video)

| Stage | Runs on | median ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| Face landmarks (MediaPipe) | CPU | 1,690 | 3,851 | 5,916 |
| Video load + mouth crop | CPU | 342 | 764 | 1,241 |
| Audio load | CPU | 7 | 12 | 23 |
| Video encode | GPU | 100 | 212 | 279 |
| Audio encode | GPU | 64 | 123 | 147 |
| **Total (sum of medians)** | | **≈ 2,200** | | |

Face landmark detection is about **77%** of the per-video time; the two GPU encoders together are
about 7%. Durations are wall-clock per stage (`time.perf_counter`, with CUDA synchronisation around
the GPU stages), one video at a time. The first video in a session is slower (GPU warm-up) and is
included in the max.

## Environment

Python 3.10.22 (separate venv; DiMoDif's versions have no wheels for Colab's default Python),
PyTorch 2.5.1+cu121, torchvision 0.20.1, torchaudio 2.5.1, MediaPipe 0.10.15, Tesla T4.

## Differences from DiMoDif's `external/features.py`

None of these change the features:
- no transcription (beam search): DiMoDif's detector does not read transcripts;
- no facenet-pytorch import (only needed for the MTCNN option; MediaPipe is DiMoDif's default).

## Still pending

- Conversion to the shared `.pt` contract (`video_tokens` / `audio_tokens`), if Member 1 needs it,
  and whether tokens are the encoder outputs (these) or after DiMoDif's projection.
- Shortcut audit on the whole videos.
