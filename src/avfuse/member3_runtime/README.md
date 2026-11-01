# Member 3 — Hardware benchmarks + integration + demonstration

**Owns:** experiment configuration, orchestration, device detection, synchronized
operator/model timing, supported peak memory, run hashes, final CLI/demo and CI.

**Implemented in starter:** generic synchronization-aware benchmark callable,
precision-mode orchestration, provenance primitives, synthetic smoke CLI.

**Requires real implementation:** checkpoint-backed CLI, full four-config output
writer, final run config storage, separate measured preprocessing cost, model/device
backend verification and statistically stable repeatable benchmarks.
