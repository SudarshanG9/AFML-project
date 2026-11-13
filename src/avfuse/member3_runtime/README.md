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

## Segment D: provenance, run management and artifact integrity

Segment D stores and verifies measurements; it runs no benchmark, model or dataset
preprocessing and writes **no research artifacts**. Everything below is exercised only with
tiny synthetic temporary files in `tests/member3/test_provenance.py` and `test_artifacts.py`.
Benchmark schema v0.2 is still a **proposal** pending Member 1/2 review; nothing here edits it.

### Hashing (`provenance.py`)

| Function | Identity |
|---|---|
| `sha256_file(path)` | raw bytes, 1 MiB chunks; raises if the file changes while read |
| `sha256_directory(path)` / `sha256_directory_detailed` | algorithm `sha256-directory-v1` |
| `sha256_json(value)` / `canonical_json_bytes` | `sha256-json-sorted-v1` (**not** RFC 8785) |
| `hash_config`, `hash_checkpoint`, `hash_cache_manifest`, `hash_feature_cache` | hash *records* naming kind, algorithm and `verified_scope` |

Directory hash: files sorted by UTF-8 bytes of the NFC, `/`-separated relative path; each
contributes `len(path) | path | size | raw file SHA-256` (fixed-width big-endian integers) under
a versioned header. Only regular files count: empty directories, timestamps, permissions and
absolute paths do not. Symlinks (including the root) and special files are rejected, never
skipped; unreadable files raise; a changed file list or file mid-hash raises. Names with
backslashes, or that collide after NFC normalization, are rejected.

Canonical JSON: sorted keys, `(",", ":")` separators, UTF-8 without ASCII escaping, NaN/Infinity,
non-string keys and unsupported types rejected. `1` and `1.0` hash differently.

Identities stay separate: config, checkpoint, cache **manifest** (`manifest_file_only`),
feature-cache **content** (`directory_contents`), run manifest and individual artifacts. A
manifest-only digest is never reported as cache content: the record gets
`input_identity_sha256 = null` and `extensions.provenance.cache_manifest_sha256` plus a note.
Hashing a requested but missing input raises `MissingInputError`. Hash outside timed callables.

### Run identity and layout (`artifacts.py`)

`create_run_context(config, runs_root=..., ...)` generates `run-<UTC>-<16 hex>` (unique, **not** a
content hash), validates the config against the existing config schema, hashes requested inputs
*before* creating the directory, and writes:

```text
<runs_root>/<run_id>/
    resolved_config.json            (reproducibility.snapshot_config)
    environment_report.json         (existing collect_environment_report / write_report)
    provenance.json                 hashes, git state, environment summary, disclaimers
    benchmark_records.jsonl         schema v0.2, one record per line
    timing_samples/<record_id>.json (only output.timing_samples_storage = separate_file)
    stage_profile.json              (optional, avfuse_stage_profile)
    run_manifest.json
    integrity/artifact_hashes.json
```

Timestamps are timezone-aware UTC, ISO 8601 with a trailing `Z`. Git commit and dirty state come
from the existing `collect_git`; a dirty tree is recorded as dirty with a warning, and an absent
git is `null` / `state: "unavailable"`. Device names are never inferred (MPS has none). Run
directories are not overwritten unless `output.overwrite` is true.

Config interpretation: Segment D records `seed` and `torch_num_threads` but does not apply them.
`output.directory` and `reproducibility.environment_report_path` are **not** write targets (a
single shared report path would be overwritten across runs and fall outside the run manifest);
each run keeps its own copy. `runs_root` is an explicit argument; the config default
(`artifacts/benchmarks`, `artifacts/runs/...`) is flagged for team review.

### Benchmark records

`build_benchmark_record` / `RunContext.build_record` convert a `benchmark_callable()` dictionary
into a v0.2 record following `docs/CONTRACTS.md`: `percentile_method` and `synchronization` go
under `measurement`; engine diagnostics with no schema home go under
`extensions.benchmark_engine`; unavailable memory stays `null`. Precision: `reference` is
`native_float`; the other three are `simulated_int8` with the actual execution dtype,
`real_int8_kernel: false` and a disclosure that the INT8 is simulated.

`validate_benchmark_records` runs `Draft202012Validator` with a `FormatChecker` (the optional
`rfc3339-validator` is not installed, so timestamps are also checked in code), then: sample count
equals `repeats`; samples finite and non-negative; `mean/median/p50/p95` recomputed from the
samples with the declared percentile method; `p50 <= p95`; device/synchronization/memory
consistency (zero is allowed when measured; unavailable must be `null`); scope/boundary/backend rules;
config-to-precision-mode mapping; `real_int8_kernel` false; no label/prediction keys in
`extensions`; one `run_id` per batch; duplicate measurement series rejected. The whole batch is
validated before anything is written; every problem is reported.

### Writing and timing samples

`write_benchmark_records` writes validated JSONL atomically (temporary sibling, fsync, atomic
replace, or a hard-link for no-clobber). `inline` keeps `timing_samples_ms` in the record.
`separate_file` writes `timing_samples/<record_id>.json`, reads each back, and stores a
**run-relative** `timing_samples_ref`; `record_id` is deterministic from the series. Existing
files are never replaced unless `overwrite`; on failure created files are removed and, with
overwrite, replaced files are restored.

### Manifest and integrity

`RunContext.finalize()` (once) writes `run_manifest.json` (relative path, type, size, SHA-256,
creation time, required flag per artifact; no absolute paths) and the anchor
`integrity/artifact_hashes.json`. The manifest excludes itself and the anchor; its own SHA-256
lives in the anchor, so there is no circular hash. The anchor catches accidental corruption and
inconsistent edits; it is **not** tamper-proof against someone who rewrites both files.
`verify_run_integrity(run_dir)` reports `valid`, `missing` (with `required`), `modified`
(`size_mismatch` / `hash_mismatch`), `unreadable`, `invalid_entries` (traversal, absolute,
symlink, duplicate or reserved paths), `reference_problems` and informational `unlisted_files`.
It never modifies or repairs anything.

### Stage profiles are not benchmark records

`RunContext.write_stage_profile` stores a `profile_pipeline_stages()` result unchanged under
`{"artifact_type": "avfuse_stage_profile", "artifact_version": "1.0", ...}`. No precision
configuration or batch size is invented, `not_provided` stages stay unmeasured, composite stages
stay distinct and no end-to-end figure is added.

### Example (synthetic, temporary)

```python
import json, tempfile
from pathlib import Path
from avfuse.member3_runtime.artifacts import create_run_context
from avfuse.member3_runtime.benchmark import benchmark_callable

config = json.loads(Path("configs/benchmark.example.json").read_text())
config["experiment_id"] = "demo-synthetic"
with tempfile.TemporaryDirectory() as tmp:
    ctx = create_run_context(config, runs_root=Path(tmp) / "runs", allow_unset_inputs=True)
    m = benchmark_callable(lambda: sum(range(100)), "cpu", warmup=1, repeats=5)
    rec = ctx.build_record(m, config="reference", scope="operator", backend="sdpa",
                           batch_size=1, execution_dtype="float32",
                           input_boundary="synthetic_tensors", input_shapes={"q": [1, 2, 3, 4]})
    ctx.write_benchmark_records([rec])
    ctx.finalize()
    assert ctx.verify()["ok"]
```

### Limitations

- One writer per run directory; no file locking or concurrent appending. JSONL is written in one
  piece, not appended.
- Overwrite is atomic per file, not across the set of files; stale sample files from an earlier
  run with the same `run_id` can remain (and appear as `unlisted_files`).
- Hard-link no-clobber falls back to check-then-replace on filesystems without hard links.
- Directory hashes need a filesystem whose names survive NFC normalization unambiguously.
- `preprocessing_once` fits the benchmark schema only with an invented precision/batch size, so
  stage profiles use their own artifact type; this is a schema limitation for Member 1/2 review.
- Real checkpoint/cache hashing awaits Member 1's checkpoint and Member 2's feature cache;
  CUDA/MPS records have only been built from synthetic measurement dictionaries (no GPU/MPS run).

## Segment E: four-mode orchestration, predictions and cached-feature benchmarking

Modules: `feature_loader.py`, `runner.py` (extended), `predictions.py`, `model_benchmark.py`,
`experiment.py` (run glue; the user-facing CLI is Segment F). Everything below consumes Member 1's
`ModelAdapter` interface and Member 2's files as they are; nothing of theirs is modified.

### What works with the current real DiMoDif adapter

| Mode | Status with `DiMoDifAdapter` |
|---|---|
| `reference` | supported |
| `low_precision`, `restore_value_output`, `restore_query` | **unsupported** (`NotImplementedError` from Member 1) |

A real four-mode experiment cannot be completed until Member 1 integrates all four modes into
the model's fusion attention. Segment E reports the unsupported modes and writes nothing that could
be mistaken for a four-mode result. The standalone `fusion_attention()` is deliberately not used.

### Feature formats (explicit, never guessed)

```python
from avfuse.member3_runtime.feature_loader import (
    load_dimodif_features, load_shared_pt_features, dimodif_npz_path, make_feature_provider, ClipRef)

feats = load_dimodif_features("<feature_root>/val/<video_id>/features.npz")   # dimodif_npz
feats = load_shared_pt_features("data/cache/<id>.pt")                         # shared_pt (Member 2 loader)
provider = make_feature_provider("dimodif_npz", feature_root)                 # provider(ClipRef) -> dict
```

The whole-video manifest's `feature_path` column still holds the planned `.pt` path; for the NPZ layout
it is ignored and the path is built from `feature_root/split/clip_id/features.npz`. `T_v` and `T_a` are
kept as stored; alignment/padding/FP32 belong to `DiMoDifAdapter.prepare_features`. The loader never
sees labels (`ClipRef` has `clip_id`, `split`, `feature_path` only).

### Per-mode status

```python
from avfuse.member3_runtime.runner import run_precision_sweep
r = run_precision_sweep(model, features, modes=("reference",))            # reference-only development
r = run_precision_sweep(model, features)                                  # strict four-mode
r.status                # complete | blocked | partial | failed
r.unsupported_modes     # e.g. ('low_precision', ...)
r.results["low_precision"].error_message   # the reason; no score, never 0
```

`NotImplementedError` => `unsupported`; any other error raises `ModeFailedError`; scores must be
finite, shape `(batch,)` and in `[0,1]` (never clipped or inverted). `allow_partial=True` is a
development aid and is labelled `partial`.

### Predictions

```python
from avfuse.member3_runtime.predictions import generate_predictions, load_clip_refs
clips = load_clip_refs("manifest_whole.jsonl", "val")      # label-free ClipRef list
report = generate_predictions(model, clips, provider, run_id=ctx.run_id,
                              checkpoint_sha256=model.checkpoint_sha256,
                              output_path="predictions.jsonl", split="val",
                              modes=("reference",), require_lavdf_hash=True)
```

One row per `clip_id x config` with only the six required schema fields. Rows are validated with
Member 2's `validate_predictions`, buffered, and published atomically (no file on failure; an existing
file is preserved unless `overwrite=True`). A strict run with an unsupported mode stops at the first clip,
publishes nothing and returns `status="blocked"`. `val` is the default. `split="test"` needs an existing
frozen `protocol_path`, `unseal=True` and a non-empty `purpose`; Member 2's `open_test` is then called once
with the full config list before any test feature is read. A hash mismatch
(`expected_checkpoint_sha256`, or `require_lavdf_hash=True` for the agreed LAV-DF hash) raises
`CheckpointMismatchError` before any inference.

### Cached-feature model benchmark

```python
from avfuse.member3_runtime.model_benchmark import benchmark_precision_modes
res = benchmark_precision_modes(model, features, ctx=ctx, device="cpu", warmup=10, repeats=50)
```

Times `model.predict_from_cached_features(features, mode)` on one fixed sample with Segment B's
`benchmark_callable`, scope `cached_feature_model`, input boundary `cached_encoder_tokens`, backend `none`
unless a verified backend is passed. This is not raw-video end-to-end latency. `conversion_overhead_ms` is
`null` unless Member 1 supplies a dedicated conversion-only callable
(`ConversionBenchmark`); it is never derived by subtracting latencies. Unsupported modes yield no record;
a strict run with any unsupported mode times nothing. An unavailable device raises (no CPU fallback).
Use a validation sample (never test) for development benchmarking. The record's `input_identity_sha256` is
a hash of the exact benchmarked tensors (`sha256-tensor-bytes-v1`), not of the whole cache.

### Run integration

```python
from avfuse.member3_runtime.experiment import execute_experiment, run_dimodif_reference
summary = execute_experiment(model, ctx, clips, provider, checkpoint_sha256=sha,
                             modes=("reference",))     # predictions + benchmark + summary
```

Registers `predictions.jsonl` and `execution_summary.json` with the `RunContext`; calls `finalize()` and
`verify()` only when every requested mode and every clip of the requested split completed, and the requested
benchmark was not blocked. Blocked/partial/failed runs are not finalized and keep
a summary (`execution_summary.json` or `failure_summary.json`). `run_dimodif_reference(...)` wires the real
reference path (whole-video manifest + NPZ features + `DiMoDifAdapter`) from caller-supplied paths and
raises `MissingInputError` for absent inputs. **Pass `cache_dir=<feature_root>/val` to
`create_run_context` for validation runs** so provenance hashing does not read test features.

### Segment E limitations

- Real `DiMoDifAdapter` runs, GPU/MPS timings and actual score orientation checks were not executed here
  (no checkpoint, upstream repo or accelerator in the test environment); tests use mock adapters and
  synthetic temp files and say nothing about detector performance.
- Prediction generation processes one video per call (batch size 1), matching the whole-video feature layout.
- The test-access ledger is written before test inference starts; a later failure therefore still leaves
  a logged access (intentional: the attempt is recorded). Run a strict validation pass first.
- For the sealed test prediction pass, set `benchmark=False`; model-level timing must be collected
  from validation features, never from the held-out test split.
