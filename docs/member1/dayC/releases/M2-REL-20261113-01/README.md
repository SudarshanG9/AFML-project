# Member 2 handoff

## Where things are (Google Drive)
- `dimodif_emb/val/<clip_id>/features.npz` + `meta.json`: 1,200 val videos
- `dimodif_emb/test/<clip_id>/...`: 2,000 test videos (storage/compatibility only; no scored test runs before the protocol freeze)
- `m2_handoff/`: this folder

## Mapping
`clip_id` in `manifest_whole.jsonl` = folder name. Path = `dimodif_emb/<split>/<clip_id>/features.npz`.
`feature_path` in the manifest is a `.pt` placeholder: don't open it; build the path above.
Use `manifest_whole.jsonl` (whole videos), not the 4-second clip `manifest.jsonl`.
Use its ids and splits for inference, never its labels.

## Arrays
`video_features (T_v, 768)`, `audio_features (T_a, 768)`, float32: auto-AVSR outputs before DiMoDif's 768->64 projection.
T_v and T_a can differ by up to 3 frames; no video exceeds 600.

## Files here
- `manifest_whole.jsonl`: clip_id, split, label, manipulation_type, source_id, identity_id (= source_id; LAV-DF has no speaker IDs)
- `videos_whole.csv`: clip_id -> original LAV-DF path, frames, duration
- `feature_coverage_report.json`: 3,200 unique ids -> one valid features.npz each (val 1,200, test 2,000), 0 problems, 0 extra folders
- `train_overlap.json`: no val/test source appears in LAV-DF train
- `video_features.csv`: per-video media measurements incl. trailing silence (shortcut control)
- `feature_report.json`: extraction log, shapes and timings of all 3,200 (3,200 ok, 0 failed)
- `extraction_provenance.json`: code commits, environment, encoders, checkpoint hash, sha256 of these files

## Code
Branch `feature/member2-data`; manifest commit `95bd42d`, extraction commit `9eeea65`.
Reproduce: `docs/member2/RUNBOOK.md` ("Whole-video manifest" and "DiMoDif features").

## Limitations
- Overlap is checked by source video; the same speaker across different sources can't be detected (no speaker IDs).
- Val = LAV-DF dev, which was used to select the checkpoint.
- Trailing silence differs between real and fake (AUROC 0.66-0.69); the control check runs once predictions exist.

LAV-DF is CC BY-NC 4.0: team use only.

## Validation handoff (for reference_val)
- `manifest_whole_val.jsonl`: the 1,200 val rows copied unchanged from `manifest_whole.jsonl` (same clip_ids, labels,
  source groups, metadata; same order). IDs = folder names in `dimodif_emb/val/` (checked 1:1).
- This is the whole-video cohort. Do NOT use the older 4-second windowed `manifest.jsonl` (1,195 val clips, different ids).
- `split_integrity.json`: source-group audit of the whole-video manifest (0 groups cross val/test).
- `train_overlap.json`: 0 val and 0 test sources in LAV-DF train (78,703 videos / 21,254 sources).
- Known limitation: val = LAV-DF's official dev split, which the DiMoDif authors used to select the pretrained checkpoint,
  so val metrics may be optimistic. That's why the threshold is frozen on val, but final claims come from the sealed test split.
