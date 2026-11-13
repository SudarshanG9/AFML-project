# Member 3 runtime architecture

```
manifest_whole.jsonl ──► load_clip_refs ──► ClipRef(clip_id, split, feature_path)   (no labels)
features.npz / shared .pt ──► feature_loader ──► {video_tokens, audio_tokens}
                                   │
Member 1 ModelAdapter ◄── run_precision_sweep (per-mode: completed | unsupported | failed)
                                   │
        predictions.generate_predictions ──► predictions.jsonl   (Member 2 validator, atomic publish,
                                   │                               sealed-test gate via Member 2 open_test)
        model_benchmark.benchmark_precision_modes ──► Segment B timing ──► Segment D records
                                   │
        experiment.execute_experiment ──► RunContext (config, env, provenance, artifacts, finalize/verify)
                                   │
                    cli.py  (validate-config | run | predict | benchmark | inspect)
```

| Layer | Module | Responsibility |
|---|---|---|
| A | `environment.py` | device/environment diagnostics |
| B | `benchmark.py` | timing engine (`benchmark_callable`), sync, percentiles, memory |
| C | `operator_benchmark.py`, `stage_profiler.py` | operator-level and stage timing (separate scopes) |
| D | `provenance.py`, `artifacts.py` | hashing, run ids, schema v0.2 records, atomic writes, integrity |
| E | `feature_loader.py`, `runner.py`, `predictions.py`, `model_benchmark.py`, `experiment.py` | orchestration |
| F | `cli.py`, `__main__.py` | user interface, exit codes, safeguards |

## Ownership boundaries

Member 1 owns the model, checkpoints, precision semantics and attention; Member 2 owns features, manifests, labels,
evaluation and the test seal; Member 3 only orchestrates. The CLI imports Member 1's adapter lazily and Member 2's
`validate_predictions`, `open_test`, `DIMODIF_LAVDF_SHA256`; it modifies none of them.

## Status model

Per mode: `completed | unsupported | failed`. Per sweep: `complete | partial | blocked | failed`. A run directory is
finalized (manifest + hash anchor) **only** for `complete`; blocked/partial/failed runs keep a summary and are never
finalized. `inspect` reports `complete_four_mode_validation_run` true only for a finalized, integrity-verified,
four-mode, full-split validation run. That flag is an implementation status, not release approval.

## Measurement scope

`cached_feature_model`, input boundary `cached_encoder_tokens`, backend `none`. Excludes raw-media decoding and frozen
encoders. Conversion overhead is `null` unless Member 1 provides a conversion-only callable. Simulated INT8 is never
described as a hardware INT8 speedup.

## Current real-model capability

`DiMoDifAdapter` supports `reference` only. The other three modes raise `NotImplementedError`; the runtime reports them
as unsupported (exit 3) and never substitutes scores.
