# Member 3 final implementation audit (Segments A-G)

Audited on branch `member3-fg` (base `f66af16`; nothing committed). Numbers below were measured in this session unless
marked otherwise.

## Implementation status

| Seg | Scope | Code | Tests (collected) | Notes |
|---|---|---|---|---|
| A | environment + schemas | `environment.py`, `contracts/schemas/*` | 20 + 58 (schema) | CPU environment only in this audit |
| B | benchmark engine | `benchmark.py` | 65 | no CPU fallback; CUDA memory untested without GPU |
| C | operator / stage profiling | `operator_benchmark.py`, `stage_profiler.py` | 77 + 48 | synthetic tensors; SDPA operator scope separate from model scope |
| D | provenance + artifacts | `provenance.py`, `artifacts.py` | 32 + 102 | real checkpoint/cache hashing awaits real files |
| E | orchestration | `feature_loader.py`, `runner.py`, `predictions.py`, `model_benchmark.py`, `experiment.py` | 25 + 19 + 17 + 26 | mock adapters; real adapter = `reference` only |
| F | CLI | `cli.py`, `__main__.py` | 39 | exit codes 0-7; sealed-test safeguards; preflight |
| G | CI + docs | `.github/workflows/ci.yml`, `docs/member3/*` | n/a | workflow YAML parsed locally; **not executed on GitHub** |

Whole repository: 580 tests, all passing at Checkpoint 2 (re-measured in the final section below).

## Findings

1. **Real-model capability gap (blocker, Member 1):** `DiMoDifAdapter.predict_from_cached_features` supports only
   `reference`. A strict four-mode run exits 3 by design. No Member 3 code hides this.
2. **Repo-wide Ruff is not clean (pre-existing, other members):** 70 findings outside `member3_runtime`/`tests/member3`.
   CI lints Member 3 paths only. Not modified here.
3. **Python 3.10 compatibility unverified locally:** the package declares `>=3.10`; CI includes 3.10 but has not run.
4. **Test-ledger semantics:** Member 2's `open_test` logs before inference; a later crash still leaves an access entry
   (intentional). The CLI's validation preflight avoids burning an access on an unsupported configuration.
5. **Bug found by Segment F tests and fixed:** plain `FileExistsError` from prediction publishing was misclassified as a
   runtime failure; now a usage error with advice.
6. **Legacy smoke stub bug fixed:** the old `cli.py` printed only the last mode (indentation). Behaviour now prints all four.
7. **`artifacts/member3/` is git-ignored** (`/artifacts/*/*`); checkpoint patches there are local recovery files.
8. **Segment E branch vs. earlier draft:** this audit is based on the integrated `f66af16`, which differs in small ways
   from an earlier standalone Segment E patch (e.g. test benchmarking is rejected inside `execute_experiment`).

## Not verified / not available

- Real DiMoDif inference, real checkpoint hash (`e1e59006...` expectation not exercised against a real file).
- GPU/MPS execution, CUDA peak-memory measurement, real timings of any kind.
- Score orientation on real data (Member 2's readiness check).
- GitHub Actions execution, Python 3.10 run.
- Member 2's reported validation results were not reproduced or reviewed by Member 3.
