# Member 3 — Hardware benchmarks + integration + demonstration

**Owns:** experiment configuration, orchestration, device detection, synchronized
operator/model timing, supported peak memory, run hashes, final CLI/demo and CI.

**Implemented in starter:** generic synchronization-aware benchmark callable,
precision-mode orchestration, provenance primitives, synthetic smoke CLI.

**Requires real implementation:** checkpoint-backed CLI, full four-config output
writer, final run config storage, separate measured preprocessing cost, model/device
backend verification and statistically stable repeatable benchmarks.

**Segment A additions:** `environment.py` (device/environment diagnostics used by
`scripts/check_environment.py`), the proposed v0.2 benchmark record schema, the
benchmark config schema/template, and tests under `tests/member3/`.

**Segment B (implemented): benchmark engine.** `benchmark.py` provides
`benchmark_callable(fn, device, warmup, repeats, ...)`, the only timing engine in this
package: warm-up and timed repetitions, CPU/CUDA/MPS synchronization, p50/p95 with the
`linear` / `lower` / `nearest_rank` percentile methods, CUDA peak allocated/reserved memory,
and honest reporting of unavailable memory (`None`, never zero). An unavailable accelerator
raises instead of falling back to CPU. Segment C builds on it and does not duplicate its
timing, synchronization, percentile or memory logic. CUDA and MPS paths have not been
exercised on real hardware; their tests use mocks.

## Segment C: operator benchmark and generic stage profiling

Segment C adds two in-memory measurement APIs. Neither writes files; artifact writing and
provenance belong to Segment D. Results are plain dicts, not benchmark schema v0.2 records.
**Passing synthetic tests demonstrates the software works. It is not a research result:
no real Q/K/V shape, stage profile or timing has been measured yet.**

### `operator_benchmark.py` - isolated fusion attention

Times Member 1's actual API, `avfuse.member1_fusion.attention.fusion_attention(q, k, v,
mode=mode, backend=api_backend, attn_mask=attn_mask)`, through `benchmark_callable`. The timed
callable contains only that call; tensor creation, validation, device transfers and mask
construction all happen beforehand, and tensors are never moved implicitly (a device mismatch
is an error).

| Public backend label | Member 1 API | Notes |
|---|---|---|
| `sdpa` | `backend="sdpa"` | Primary performance path. SDPA kernel selection is not recorded. |
| `dense_oracle` | `backend="dense"` | Correctness oracle that materializes the score matrix; guarded by `dense_max_score_bytes` (default 128 MiB score matrix). Not a substitute for SDPA performance. |

Any other label (for example `custom_tiled`) is rejected; there is no custom tiled kernel.
The four precision modes are the shared-contract names `reference`, `low_precision`,
`restore_value_output`, `restore_query`.

- **Inputs:** Q `[B, H, Nq, D]`, K `[B, H, Nk, D]`, V `[B, H, Nk, Dv]`; `Nq` and `Nk` may
  differ. Rank, batch, heads, Q/K head dimension, K/V length, a single float dtype
  (fp32/fp16/bf16), device and mask shape/dtype are validated. Boolean mask `True` means
  attention is permitted (Member 1's semantics).
- **Synthetic vs real:** `input_kind` is mandatory. `synthetic` tensors (explicitly requested
  via `make_synthetic_inputs` / `synthetic_cases`) are for software validation only. `real`
  requires a non-empty `shape_source` (for example Member 1's shape report). That declaration
  is made by the caller and is **not verified** by the benchmark.
- **Sweeps:** `build_batch_head_sweep` varies batch size and head count over explicitly
  supplied token/head dimensions (nothing is defaulted); `run_operator_sweep` validates every
  case/backend/mode before timing any of them.
- **Simulated INT8:** non-reference modes run `fake_quant_int8`, which keeps floating-point
  tensors. That work is inside `fusion_attention`, so it is included in the measured latency;
  it is **not** an isolated conversion-overhead measurement, not real INT8 storage, not
  integer-kernel execution and not evidence of INT8 hardware speedup.
- **Score-matrix estimate (H1):** every result records `batch_size * num_heads * q_tokens *
  k_tokens * bytes_per_element` (bytes and MiB) via Member 1's `score_matrix_bytes(q, k)`. It
  is a theoretical dense-matrix size, **not** the memory allocated by fused SDPA.

### `stage_profiler.py` - generic, independent stage profiling

`profile_pipeline_stages` times each supplied stage independently through
`benchmark_callable`. It does not import or re-implement Member 1's or Member 2's code:
the caller binds each stage to a zero-argument callable on prepared input, or supplies an
`ExternalMeasurement` (durations plus a required source description) for work measured
elsewhere, such as a recorded run or a shell-timed download. External durations are stored
as given; no percentiles are derived.

Stage names follow the actual LAV-DF pipeline on Member 2's `feature/member2-data` branch:

| Stage | Corresponds to | Group |
|---|---|---|
| `metadata_indexing` | `lavdf.index_lavdf()` | `preprocessing_once` |
| `split_sampling` | `lavdf.sample_per_split()` | `preprocessing_once` |
| `selection_list_generation` | list-writing branch of `m2_lavdf.py select` (writes a file list; **does not download**) | `preprocessing_once` |
| `dataset_transfer` (optional) | external network download | `preprocessing_once` |
| `archive_extraction` (optional) | external local archive extraction (disk I/O, not network) | `preprocessing_once` |
| `video_decode`, `audio_decode` | the two ffmpeg calls inside `lavdf.extract_clip()` | `preprocessing_once` |
| `raw_array_serialization` | `np.save` of `frames.npy` / `audio.npy` in `extract_record()` | `preprocessing_once` |
| `clips_csv_serialization` | `lavdf.write_clips_csv()` | `preprocessing_once` |
| `clip_validation` | `long_enough()` / `window_covers_fake()` | `preprocessing_once` |
| `manifest_row_construction` | `manifest.manifest_row_from_clip()` | `preprocessing_once` |
| `split_audit` | `splits.split_audit()` | `preprocessing_once` |
| `manifest_serialization` | `manifest.write_manifest()` (also runs `verify_manifest()`; not separable) | `preprocessing_once` |
| `feature_encoding` | frozen-encoder cache generation | `preprocessing_once` |
| `fusion`, `head` | cached-feature fusion and detection head | `cached_feature_model` |

The operator microbenchmark is a separate `operator` scope. The groups are never combined
into one latency number, and cached-feature latency is never raw-video end-to-end latency.

**Composite stages.** Member 2's current code has no separate callable boundary for several
operations: `extract_clip()` runs video then audio ffmpeg sequentially; `extract_record()`
adds `.npy` serialization (and skips decoding if the outputs already exist, so it must be
bound to a fresh output directory); `m2_build_manifest.py` runs validation, row
construction, audit and writing together; `m2_lavdf.py select` has no function boundary. Such
operations can only be profiled under an explicit combined name (`combined_selection_command`,
`combined_audio_video_decode`, `combined_decode_and_array_serialization`,
`combined_manifest_build`). A combined measurement is reported as itself, leaves the stages it
covers `not_provided` (nothing is inferred or apportioned) and never makes a profile
`complete`. If Member 2 later exposes separate callables, they plug into the individual stage
names with no redesign.

**Missing stages.** A stage with neither a callable nor an external measurement has
`status = "not_provided"` and contains no timing keys: never `0 ms`. Each record also states
`callable_supplied`, `measurement_completed` and `measurement_origin`, so a known stage name
is never mistaken for a measurement. `missing_stages` lists unmeasured required stages;
`dataset_transfer` and `archive_extraction` are optional. A callable that raises aborts the
profile with `StageExecutionError` (original exception chained); it is never recorded as a
timing.

**Settings.** Preprocessing stages default to `StageSettings(warmup=0, repeats=1)` so
one-time operations are not repeated by accident; `fusion` and `head` default to
`warmup=10, repeats=50` (configurable via `repeatable_settings`). Any stage can be overridden
through `stage_settings`, and `stage_devices` allows a per-stage device.

**Input metadata.** `input_kind` is `synthetic`, `real` or `unspecified` (the default, treated
as non-research). `real` requires a non-empty source description. The profiler cannot verify
the declaration.

**No end-to-end figure.** Independent stage statistics are never summed and
`end_to_end_measured` is always `False`; no end-to-end callable is supported yet.

### Still pending (not measured by Segment C)

- `feature_encoding`, `fusion` and `head` need Member 1's checkpoint/model handoff and
  Member 2's feature cache (`preprocess.py` on Member 2's branch is still a placeholder).
- The real operator benchmark needs Member 1's validated Q/K/V shape report.
- Separate `video_decode` / `audio_decode` / `raw_array_serialization` timings need
  separate callable boundaries from Member 2 (otherwise only combined stages are possible).
- `dataset_transfer` and `archive_extraction` need an externally supplied measurement.
- Real CUDA/MPS measurements need that hardware.
