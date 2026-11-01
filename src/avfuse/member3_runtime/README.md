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
benchmark config schema/template, and tests under `tests/member3/`. Benchmark
engine work (Segment B) is not yet implemented.
