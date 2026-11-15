# Member 3: cached-feature runtime, benchmarks and provenance

The current CLI supports all four modes through the integrated Member 1 DiMoDif adapter. Older Segment E notes describing reference-only support preceded that integration. A real run requires the external upstream model, its verified checkpoint/config, a manifest and cached encoder features. The public smoke command uses a toy model and is not forensic evidence.

```bash
python -m avfuse.member3_runtime.cli --help
python -m avfuse.member3_runtime.cli predict --help
python -m avfuse.member3_runtime.cli benchmark --help
python -m avfuse.member3_runtime.cli inspect --help
```

## Responsibilities and stable interfaces

| Module | Responsibility |
| --- | --- |
| `cli.py`, `__main__.py` | validate-config, predict, run, validation-only benchmark and inspect commands |
| `feature_loader.py` | load NPZ or supported PT caches; map them to audio/video token tensors |
| `runner.py` | execute shared precision modes through `ModelAdapter` |
| `predictions.py` | six-field prediction schema, exact clip coverage, checkpoint checks and atomic publication |
| `experiment.py` | connect predictions, cached-model benchmarks and a finalized run directory |
| `benchmark.py` | common warmup/repeat/synchronization timing engine and percentile computation |
| `model_benchmark.py` | time the resident model on one validation clip, with applicable fake-quant conversions |
| `operator_benchmark.py` | isolate attention from projections, model and data loading |
| `stage_profiler.py` | independent preprocessing stage measurements, never summed into an end-to-end percentile |
| `environment.py` | device, backend and environment diagnostics |
| `provenance.py`, `artifacts.py` | input hashes, run manifests, sample storage and integrity checks |

These names already describe functional responsibility. Historical `scripts/dayB_model_benchmark.py` remains under its original frozen-package path. It is separate from the newer generic cached-model CLI; its accepted measurements must keep their original identities.

## Measurement interpretation

`cached_feature_model` includes model preparation/padding, fusion/head and applicable conversions on resident tokens. It excludes media loading/decoding, face tracking, encoder extraction, cache I/O and transfers prepared before the timer. `operator` excludes the full model and projections. A synthetic smoke benchmark validates wiring only. The generic CLI may label an adapter-backed model record backend `none` when it does not introspect internal kernels; the accepted historical DiMoDif measurements explicitly record SDPA. Neither label establishes a specific optimized CUDA kernel.

Simulated INT8 keeps floating tensors and performs rounding/dequantization. It can be slower than reference. CPU peak memory remains null when not measured. CUDA/MPS synchronization and supported allocator accounting exist in code, but passing mocked tests does not supply real accelerator measurements. The accepted study has CPU evidence; no native INT8, end-to-end speedup or measured peak-memory reduction is established.

## Integrity and safe failure

Only complete requested-mode runs are finalized. Unsupported modes and development subsets remain explicitly blocked/partial. A failure leaves an error summary rather than a fabricated successful result. Test prediction access requires explicit protocol, purpose and unseal flags; model benchmarks require validation. No cleanup work reopens the accepted cohort.

Run manifests retain paths, sizes and SHA-256 identities separately for configurations, checkpoints, cache manifests and cache contents. Manifest-only hashing is not full cache-byte verification. Integrity checks detect disagreement with saved inventories; they do not authenticate a human author or defend against an adversary who rewrites both payloads and inventories. Hashing stays outside timing. Exit codes: 0 complete, 1 failed, 2 usage, 3 blocked, 4 incomplete, 5 sealed, 6 missing input, 7 checkpoint mismatch.

The frontend imports these backend modules in its own branch. This cleanup preserves module paths, CLI arguments, feature formats and schemas; no frontend source is modified. Test with `PYTHONPATH=src python -m pytest tests/member3 -q` using synthetic fixtures.

## Operator API and stage inventory

The public `sdpa` label maps to `backend="sdpa"`; `dense_oracle` maps to `backend="dense"`. The modes are `reference`, `low_precision`, `restore_value_output`, and `restore_query`. Score-matrix bytes are `batch * heads * q_tokens * k_tokens * bytes_per_element`: theoretical dense storage, **not** the memory allocated by fused SDPA. Dense allocation is guarded separately.

The stage profiler recognizes the following individual and composite stage names:

- `metadata_indexing`
- `split_sampling`
- `selection_list_generation`
- `combined_selection_command`
- `dataset_transfer`
- `archive_extraction`
- `video_decode`
- `audio_decode`
- `combined_audio_video_decode`
- `raw_array_serialization`
- `combined_decode_and_array_serialization`
- `clips_csv_serialization`
- `clip_validation`
- `manifest_row_construction`
- `split_audit`
- `manifest_serialization`
- `combined_manifest_build`
- `feature_encoding`
- `fusion`
- `head`

A supplied callable is timed independently; an external measurement retains its stated source. An absent measurement stays `not_provided`. Composite stages cover inseparable preparation operations and never invent timings for their parts. Independent medians are never added into an end-to-end percentile.
