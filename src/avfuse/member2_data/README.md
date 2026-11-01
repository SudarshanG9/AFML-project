# Member 2: dataset + forensic evaluation

**Owns:** data licence and manipulation labels, source/identity-disjoint train/val/test manifest,
offline A/V preprocessing, temporal alignment, frozen-encoder feature caching, baseline/lowp/restored
metrics and cluster bootstrap CIs, shortcut audits.

Everything except the encoder feature cache runs **without PyTorch and without a model**: the evaluator
only reads `manifest.jsonl` + prediction JSONL, so it is tested on synthetic fixtures before real
predictions exist.

| Module | What it does | Plan day |
| --- | --- | --- |
| `labels.py` | dataset flags → `real` / `audio_fake` / `video_fake` / `both_fake` | 1 |
| `splits.py` | connected source+identity groups, split audit, leak repair (test never redrawn) | 2 |
| `lavdf.py` | LAV-DF index, per-split sampling, ffmpeg clip extraction, window rules | 3–4 |
| `manifest.py` | schema + leak validation (starter), build/read/write `manifest.jsonl` | 1–4 |
| `predictions.py` | validate prediction rows against the schema, join on `clip_id`, no silent drops | 5 |
| `evaluate.py` | threshold at fixed real FPR (val), cluster bootstrap CIs, per-attack-type metrics | 5–6, 10 |
| `paired.py` | paired cluster-bootstrap deltas, restoration table with measured cost | 7–8 |
| `seal.py` | test-set seal + access ledger | 9 |
| `shortcuts.py` | leakage audit (silence, duration, cut position, codec) | 11 |
| `report.py` | Markdown tables and figures | 12 |
| `cache.py`, `preprocess.py` | frozen-encoder cache (starter); **waits for the Lead's checkpoint** | 3–4 |

Scripts (in `scripts/`): `m2_lavdf.py`, `m2_build_manifest.py`, `m2_evaluate.py`, `m2_paired.py`,
`m2_audit.py`. Step-by-step commands: `docs/member2/RUNBOOK.md`. Dataset decisions: `docs/member2/DATASET.md`.

**Decisions worth reviewing:** one operating threshold, frozen on validation from the `reference`
config at 5% real FPR and applied to all four configs; LAV-DF has no speaker IDs, so
`identity_id = source_id` (noted in each row); `feature_path` names the planned cache file
`data/cache/<clip_id>.pt`, created once the checkpoint is locked.
