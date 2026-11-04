# Shortcut audit results: whole videos (Member 2)

Run on Google Colab (CPU), 5 Nov 2026, with `scripts/m2_audit_whole.py` over the whole-video
manifest (val 1,200 + test 2,000). 3,200 videos read, 0 audio errors. Only totals are recorded here.

Rule fixed in advance: a feature is **flagged** if it alone separates real from fake with
shortcut AUROC >= 0.65 (0.5 = no information, 1.0 = perfect shortcut).

## Result per feature (all fake types vs real)

| Feature | val AUROC | test AUROC | Verdict |
| --- | --- | --- | --- |
| **trailing silence** | **0.682** | **0.660** | **FLAG** |
| duration | 0.628 | 0.621 | below the rule, noted |
| audio duration | 0.628 | 0.621 | below the rule, noted |
| number of frames | 0.624 | 0.616 | below the rule, noted |
| container bitrate | 0.631 | 0.604 | below the rule, noted |
| file size | 0.581 | 0.582 | below the rule |
| loudness (rms dB) | 0.518 | 0.506 | no shortcut |
| leading silence | 0.501 | 0.500 | no shortcut |
| audio codec, sample rate, video codec, fps, resolution | 0.500 | 0.500 | identical for real and fake |

Per fake type, trailing silence is flagged in val for audio_fake (0.688), video_fake (0.681) and
both_fake (0.677), and in test for video_fake (0.667) and both_fake (0.673): 7 flags in total.
Test audio_fake is not flagged (below 0.65).

## Direction

| | real median | fake median |
| --- | --- | --- |
| trailing silence (ms), val / test | 40 / 40 | 70 / 70 |
| duration (s), val / test | 6.6 / 6.3 | 8.3 / 7.6 |
| container bitrate (kbps), val / test | 175.9 / 175.8 | 167.0 / 167.5 |

**Fake videos end with slightly more silence** (about 30 ms more at the median, under one video
frame), and they are a little longer, with a slightly lower bitrate. This is consistent across val
and test and across all fake types, so it most likely comes from how LAV-DF produced its fakes,
not from our pipeline: whole videos are used unchanged, and real and fake videos share the same
codecs, sample rate, fps and resolution.

## What it means

- DiMoDif hears the whole audio track, including the silence at the end, so it **could** partly
  rely on this cue. The flag does not show that it does.
- **The leading-silence shortcut seen in other audio-visual datasets is absent here** (0.50), and
  so is the cut-position effect of the earlier 4 s clip pipeline.

## Follow-up (once predictions exist)

`scripts/m2_shortcut_control.py` measures the detector's AUROC on a subset where real and fake
videos have matched trailing-silence distributions (10 bins, equal counts per bin), on validation
first. A small drop means the detector does not depend on the cue; a large drop is reported as a
limitation. The per-video features needed for it are saved in `video_features.csv` (Drive, not git).
