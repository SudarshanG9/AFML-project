# Shared interfaces: version 0.1

JSON Schemas are in `contracts/schemas/`. Member 3 coordinates changes; Members 1 and 2 approve effects on inference/evaluation.

## 1. Manifest from Member 2

One JSONL row per clip with: `clip_id`, `dataset`, `source_id` (source-video family), `identity_id` (person identity), `split` in `{train,val,test}`, `label` in `{0,1}`, `manipulation_type` in `{real,audio_fake,video_fake,both_fake}`, and `feature_path`.

**Do not** infer one person equals one original source. Keep both groups and enforce disjointness across val/test/train. Dataset-specific parent-video/identity relationships may need a stronger connected-group partition strategy.

`feature_path` points to a .pt dictionary with `audio_tokens` and `video_tokens`, post frozen encoders. Model adapter defines expected token shapes and optional masks. Never store real dataset content in git.

## 2. Prediction from Member 1 to Member 2

One JSONL row per `clip_id` × `config` with `run_id`, `clip_id`, `split`, `config`, `score_fake` (probability 0–1), and `checkpoint_sha256`. Ground-truth labels are not part of prediction rows. The evaluator obtains labels only through the manifest.

## 3. Benchmark from Member 3 — **PROPOSED `schema_version` 0.2 (not frozen)**

> **Review required.** This section and `contracts/schemas/benchmark.schema.json` are a Member 3 *proposal*. Member 1 and Member 2 must review before the contract is considered frozen (see `docs/OWNERSHIP.md`). It does not redefine Member 1's precision semantics or Member 2's dataset semantics.

One JSONL row per benchmark series (one `config` x `device` x `backend` x `scope`). The legacy v0.1 field names (`run_id`, `config`, `device`, `backend`, `scope`, `batch_size`, `repeats`, `p50_ms`, `p95_ms`, `peak_memory_bytes`, `conversion_overhead_ms`, `notes`) are kept with the same meaning. **v0.1 rows are not valid v0.2 rows** (new required fields); no benchmark outputs had been generated when this was proposed. The schema rejects unknown fields; extra data goes in `extensions`.

| Group | Fields |
|---|---|
| Identity | `schema_version` (`"0.2"`), `run_id`, `timestamp_utc` (UTC, trailing `Z`), `experiment_id`, `config` |
| Scope | `scope`: `operator` (fusion operator only), `cached_feature_model` (cached-feature fusion + head), `preprocessing_once` (one-time raw-media preprocessing) |
| Execution | `device` (`cpu`, `cuda[:N]`, `mps`), `backend` (`sdpa`, `dense_oracle`, `custom_tiled`, `none` = no attention operator), `batch_size`, `warmup`, `repeats` |
| Precision disclosure | `precision.mode` (`native_float` / `simulated_int8`), `precision.execution_dtype`, `precision.simulated`, `precision.real_int8_kernel` (**always `false`**), `precision.disclosure` |
| Workload | `workload.input_boundary` (`synthetic_tensors` / `cached_encoder_tokens` / `raw_media`), `input_shapes`, `input_duration_seconds`, `checkpoint_sha256`, `input_identity_sha256` — all except the boundary may be `null` |
| Latency (ms) | `mean_ms`, `median_ms`, `p50_ms`, `p95_ms`, `measurement.percentile_method`, `measurement.synchronization`, and `timing_samples_ms` **or** `timing_samples_ref` (at least one) |
| Memory | `peak_memory_bytes` (`null` if unavailable), `memory_method` |
| Conversion | `conversion_overhead_ms` (nullable), `conversion_overhead_included_in_latency` (nullable) |
| Environment | `environment.python_version`, `torch_version`, `os_platform`, `device_metadata` (`type`, `name`, optional free-form `properties`), `git_commit` (nullable), optional `git_dirty`, `environment_report_ref` |

Rules encoded in the schema: measurements are `>= 0`; `peak_memory_bytes` is `null` **exactly when** `memory_method` is `not_measured` (CPU rows need no CUDA memory); `simulated_int8` implies `precision.simulated = true`; `preprocessing_once` requires `raw_media` input and `backend = none`; `operator` and `cached_feature_model` may not use `raw_media`; `operator` requires a real attention backend; `device` must agree with `environment.device_metadata.type`, the synchronization method, and cannot claim CUDA memory on CPU/MPS.

Rules the schema **cannot** express (the Segment B writer must enforce): `len(timing_samples_ms) == repeats`; `p50_ms <= p95_ms`; `p50_ms`/`p95_ms` computed with the declared `percentile_method`. `median_ms` is `statistics.median` and may differ slightly from `p50_ms` under `lower`/`nearest_rank`.

Note: the existing `benchmark_callable` computes p95 as `sorted[int(0.95*(n-1))]` (`lower`), while the config template defaults to `linear`. Segment B must reconcile these and record the method actually used.

## 4. Cross-modal attention API

`fusion_attention(q,k,v,mode=...,backend='sdpa',attn_mask=None)` expects tensors shaped `(batch, heads, tokens, head_dim)`, returns fused attention output. Supported modes: `reference`, `low_precision`, `restore_value_output`, `restore_query`.

**Important:** real model may contain multiple fusion blocks/directions or projections. Member 1 documents exactly which block(s) were instrumented. The scaffold's tensor-level fusion wrapper is not a complete AVFF adapter.

## 5. Real model adapter

Concrete subclass of `ModelAdapter`, implementing `predict_from_cached_features(features, mode) -> Tensor` with fake-class probability in `[0,1]`. Input includes post-encoder tokens and any additional model-specific metadata needed for alignment/masks. The class is deliberately abstract until checkpoint selection.

## 6. Benchmark configuration — `configs/benchmark.example.json` (PROPOSED, Member 3)

Validated by `contracts/schemas/benchmark_config.schema.json` (new file; shared-directory change that needs review by Members 1 and 2). The template is an *example*: it contains no tensor dimensions, no hardware assumptions and no results. `null` means **unset**; the runner must refuse a real run while `checkpoint_path`, `cache_manifest_path` or (for observed shapes) `operator_shapes_file` are `null`, or while `experiment_id` begins with `UNSET_`.

| Option | Meaning |
|---|---|
| `config_version` | Version of this configuration layout (`"0.1"`). |
| `experiment_id` | Label copied into every record. Example value is `UNSET_EXPERIMENT_ID` and must be replaced. |
| `device` | Requested device: `auto`, `cpu`, `cuda`, `cuda:N` or `mps`. The example uses `cpu` because it exists everywhere; CUDA/MPS are never assumed. |
| `on_device_unavailable` | `error` (default) stops if the requested device is missing; `fallback_to_cpu` substitutes CPU. The actual device is always recorded. |
| `backends` | Attention implementations to time: `sdpa`, `dense_oracle`, `custom_tiled`. |
| `configs` | Precision configurations to benchmark (the four agreed identifiers). |
| `batch_size`, `warmup`, `repeats` | Batch size; untimed warm-up calls; timed repetitions. |
| `measure_scopes` | Scopes repeated `repeats` times: `operator`, `cached_feature_model`. |
| `preprocessing_scope` | `record_once_separately` times raw-media preprocessing once and reports it as its own `preprocessing_once` series; `disabled` skips it. |
| `operator_shapes_source` / `operator_shapes_file` | Where operator tensor shapes come from (`observed_tensor_shapes` = taken from the real model; `user_supplied`; `synthetic_smoke`) and the file holding them (`null` = unset). |
| `input_boundary` | Declares what each scope consumes and a mandatory disclosure string. Cached-model latency excludes raw-media decode and encoder extraction. |
| `checkpoint_path`, `cache_manifest_path` | Real-model inputs (`null` = unset); hashed into the records when `hash_checkpoint` / `hash_input_cache` are on. |
| `synchronization` | `enabled` master switch; `after_warmup` synchronizes after warm-up; `around_each_repeat` synchronizes before and after every timed call. No-op on CPU. |
| `peak_memory` | `cuda_allocator_peak_if_available_else_null` records CUDA peak allocated bytes, else `null`; `disabled` records `null`. |
| `peak_memory_reset_before_measurement` | Reset CUDA peak statistics before the timed loop. |
| `percentile_method` | `linear`, `lower` or `nearest_rank` (definitions in the schema). |
| `conversion_overhead` | `measure`: time precision conversion separately; `included_in_latency`: whether it is already inside p50/p95. |
| `output` | `directory` for benchmark JSONL; `format` (`jsonl`); `timing_samples_storage` (`inline` or `separate_file`); `overwrite` (`false` = refuse to clobber). |
| `reproducibility` | `seed`; `torch_num_threads` (`null` = PyTorch default); record the git commit and environment report (`environment_report_path`); hash checkpoint and input cache; `snapshot_config` stores the resolved config with the run. |

## 7. Environment report — `scripts/check_environment.py` (Member 3)

`python scripts/check_environment.py [--output PATH] [--quiet]` writes `artifacts/runs/environment_report.json` (ignored by git) and prints a text summary. It exits non-zero only if the report cannot be produced or written. It records Python/OS/CPU/RAM, PyTorch/CUDA/MPS availability, available devices, synchronization availability, and one small FP32/FP16/BF16/SDPA tensor operation per device with status `succeeded | failed | unsupported | not_tested`. These checks are **not** performance measurements and do **not** establish that the research model supports a precision on a device. `psutil` is optional (physical core count and RAM); without it RAM falls back to OS APIs and physical cores are `null`. No INT8 hardware test exists because INT8 here is simulated fake quantization.
