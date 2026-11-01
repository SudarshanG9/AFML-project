# Architecture and integration flow

```text
OFFLINE ONCE
Member 2: licensed AV clips -> decode / synchronize / encode -> cache of frozen
          audio_tokens + video_tokens + manifest and provenance

REPEATABLE PRECISION SWEEP
Member 3 runner -> Member 2 cache loader -> Member 1 model adapter
    -> [cross-modal Q/K/V attention in four precision modes] -> frozen detection head
    -> prediction JSONL -> Member 2 stratified metrics + cluster bootstrap
    -> Member 3 operator/model benchmarks + run provenance -> final report & CLI
```

## The strict boundary

- **Member 2** owns feature preparation and label custody. Raw clips are never decoded during the four precision runs.
- **Member 1** owns model-specific encoder-token expectations, fusion implementation and one consistent binary detector head. For the pretrained-checkpoint path, everything remains frozen.
- **Member 3** is the integration owner: orchestration, device synchronization and timing, reproducibility, smoke test and CLI.
- **Shared contracts** live outside the member folders. Changes require agreement before merge.

## Research hypotheses

- **H1:** estimate `B * H * Nq * Nk * bytes_per_element` from actual recorded Q/K tensor shapes; document decision to use SDPA versus a custom tiled kernel.
- **H2:** compare reference versus simulated INT8 correctness, with real vs each of three fake categories, on source/identity-disjoint data.
- **H3:** compare targeted V/output restoration with query-path control using paired predictions and measured incremental costs.

## Scope safeguards

- **Restoration:** in the first documented implementation, `restore_value_output` uses original high-precision V *and* skips the final fake-quantized fusion output. `restore_query` uses original high-precision Q while keeping K/V/output fake-quantized. They are not automatically cost-matched; quantify the difference or modify the control design *before* final validation freeze.
- **Oracle:** dense attention is for small correctness checks. SDPA is the performance implementation where supported.
- **Numerics:** simulated INT8 uses floating tensors with rounding. It does not benchmark a deployed INT8 kernel.
- **Timing:** operator is fusion only; cached-feature model is fusion + head (+ any layers after cache); preprocessing is separately timed once.
- **Negative outcomes:** no significant degradation and unsuccessful restoration are acceptable results.
