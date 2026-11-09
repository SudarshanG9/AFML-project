# Dataset card: LAV-DF (Member 2)

## Source and licence

- **Dataset:** LAV-DF (Localized Audio-Visual DeepFake), Cai et al., 2022.
  Hugging Face: `ControlNet/LAV-DF` (gated: accept the terms on the dataset page).
- **Licence:** CC BY-NC 4.0, research and non-commercial use only. Clips come from VoxCeleb2,
  so they show real people. **No clip, frame, audio or derived feature is committed to git**
  (see `.gitignore`).
- **Why LAV-DF first:** access to FakeAVCeleb did not arrive in time. FakeAVCeleb stays an optional
  second dataset (plan: "second-dataset transfer is optional and clearly scoped").

## What a label means

LAV-DF fakes replace **a few words** inside an otherwise real video: the audio, the video, or both are
edited in a short segment (`fake_periods` in `metadata.json`). Mapping to the shared contract:

| `modify_video` | `modify_audio` | `manipulation_type` | `label` |
| --- | --- | --- | --- |
| 0 | 0 | `real` | 0 |
| 0 | 1 | `audio_fake` | 1 |
| 1 | 0 | `video_fake` | 1 |
| 1 | 1 | `both_fake` | 1 |

Because only a short segment is fake, every fake clip is cut **centred on its first edited segment**, and
a fake whose edit falls outside the model's evaluation window is excluded (otherwise a "fake" label
would sit on real content). The exclusion counts are reported.

## Groups used for splitting and confidence intervals

- `source_id`: the original video a clip comes from (`original` in the metadata). A fake and the real
  video it was edited from share it.
- `identity_id`: **LAV-DF's metadata has no speaker ID**, so `identity_id` cannot be filled with a real
  person identity. It is set equal to `source_id`, and this is recorded in each row's `notes`.
  Consequence: splits are **source-disjoint, not guaranteed speaker-disjoint**. The same VoxCeleb2
  speaker may appear in two splits through different source videos. This is a stated limitation.
- Splits use LAV-DF's official train / dev (= val) / test partition, audited for source leakage, and
  repaired by moving whole groups if needed (test is never redrawn).

## Sample used

| Split | per category | categories | target clips |
| --- | --- | --- | --- |
| train | 1,500 | 4 | 6,000 |
| val | 300 | 4 | 1,200 |
| test | 500 | 4 | 2,000 |

Measured so far (test split, 28 Oct): 2,000 sampled, 1 too short, 7 fakes with the edit outside the
window → **1,992 usable** (500 real, 499 audio_fake, 493 video_fake, 500 both_fake).
Full build (2 Nov): 9,200 selected, 2 too short, 55 fakes with the edit outside the window ->
**9,143 usable** (train 5,956 / val 1,195 / test 1,992). Official splits passed the source audit.
Details: `docs/member2/results/DATASET_BUILD_RESULTS.md`.

## Preprocessing (raw clip stage)

25 fps video, centre crop 96×96 (VoxCeleb2 clips are already face-centred, so no face detector),
mono 16 kHz audio, 4 s stored per clip. The model sees a 2 s window. These raw clips are an
intermediate stage. The frozen-encoder feature cache (`data/cache/<clip_id>.pt`) must follow the
Lead's checkpoint preprocessing and is built after the checkpoint is locked.
