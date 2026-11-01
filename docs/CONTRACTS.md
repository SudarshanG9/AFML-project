# Shared interfaces: version 0.1

JSON Schemas are in `contracts/schemas/`. Member 3 coordinates changes; Members 1 and 2 approve effects on inference/evaluation.

## 1. Manifest from Member 2

One JSONL row per clip with: `clip_id`, `dataset`, `source_id` (source-video family), `identity_id` (person identity), `split` in `{train,val,test}`, `label` in `{0,1}`, `manipulation_type` in `{real,audio_fake,video_fake,both_fake}`, and `feature_path`.

**Do not** infer one person equals one original source. Keep both groups and enforce disjointness across val/test/train. Dataset-specific parent-video/identity relationships may need a stronger connected-group partition strategy.

`feature_path` points to a .pt dictionary with `audio_tokens` and `video_tokens`, post frozen encoders. Model adapter defines expected token shapes and optional masks. Never store real dataset content in git.

## 2. Prediction from Member 1 to Member 2

One JSONL row per `clip_id` × `config` with `run_id`, `clip_id`, `split`, `config`, `score_fake` (probability 0–1), and `checkpoint_sha256`. Ground-truth labels are not part of prediction rows. The evaluator obtains labels only through the manifest.

## 3. Benchmark from Member 3

One JSONL row per benchmark series with `run_id`, `config`, `device`, `backend`, `scope`, `batch_size`, `repeats`, `p50_ms`, `p95_ms`, and `peak_memory_bytes` (`null` if unavailable). An explicit `scope` prevents comparing raw-media processing with cached inference.

## 4. Cross-modal attention API

`fusion_attention(q,k,v,mode=...,backend='sdpa',attn_mask=None)` expects tensors shaped `(batch, heads, tokens, head_dim)`, returns fused attention output. Supported modes: `reference`, `low_precision`, `restore_value_output`, `restore_query`.

**Important:** real model may contain multiple fusion blocks/directions or projections. Member 1 documents exactly which block(s) were instrumented. The scaffold's tensor-level fusion wrapper is not a complete AVFF adapter.

## 5. Real model adapter

Concrete subclass of `ModelAdapter`, implementing `predict_from_cached_features(features, mode) -> Tensor` with fake-class probability in `[0,1]`. Input includes post-encoder tokens and any additional model-specific metadata needed for alignment/masks. The class is deliberately abstract until checkpoint selection.
