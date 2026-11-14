# Member 3 Segments F + G: implementation plan

## Verified baseline (Checkpoint 0)

- Branch worked on: local `member3-fg`, created from `origin/feature/member3-segment-e`.
- Base commit: `f66af16` ("Implement Member 3 Segment E experiment orchestration").
  `902cd7b` (Merge PR #7) is its verified ancestor.
- Working tree at start: clean. Baseline `pytest -q`: **541 passed, 0 skipped** (the "487 passed / 2 skipped"
  quoted in the assignment differs from what this environment produces; this document uses measured numbers only).
- Baseline `ruff check src/avfuse/member3_runtime tests/member3`: passed.
- Baseline `ruff check .` (whole repo): **70 pre-existing errors outside Member 3**. CI therefore lints only
  Member 3 paths (see below). Not fixed here: those files belong to other members.

## Existing implementation discovered

- Segments A-E under `src/avfuse/member3_runtime/`: `environment.py`, `benchmark.py`, `operator_benchmark.py`,
  `stage_profiler.py`, `provenance.py`, `artifacts.py` (`RunContext`, `create_run_context`, `verify_run_integrity`),
  `runner.py` (`run_precision_sweep`), `feature_loader.py`, `predictions.py` (sealed-test gate), `model_benchmark.py`,
  `experiment.py` (`execute_experiment`, `run_dimodif_reference`, `build_dimodif_adapter`).
- `cli.py` is the old synthetic `--smoke` stub (README documents `python -m avfuse.member3_runtime.cli --smoke`).
  It prints only the last mode (indentation bug) and has no tests.
- Member 2 conventions: argparse sub-commands, `--protocol`, `--split {val,test}`, `--unseal-test`, `--purpose`
  (`scripts/m2_evaluate.py`); `SealedTestError`; `ContractError`.
- No `.github/` directory, no CI. `.gitignore` ignores `/artifacts/*/*`, so `artifacts/member3/checkpoints/` is local only.
- Current `DiMoDifAdapter` supports `reference` only; the other three modes raise `NotImplementedError` (Member 1).

## Planned changes (Member 3 only)

| File | Purpose |
|---|---|
| `src/avfuse/member3_runtime/cli.py` | Rewrite: sub-commands `validate-config`, `run`, `predict`, `benchmark`, `inspect`, plus legacy `--smoke` |
| `src/avfuse/member3_runtime/__main__.py` | `python -m avfuse.member3_runtime` |
| `tests/member3/test_cli.py` | Help, invalid args, exit codes, sealing, output contracts, integration with mock model |
| `.github/workflows/ci.yml` | CPU-only CI: Member 3 ruff, full pytest, CLI help smoke |
| `docs/member3/CLI.md`, `ARCHITECTURE.md`, `TROUBLESHOOTING.md` | Usage, runtime architecture, failure handling |
| `docs/member3/FINAL_AUDIT.md`, `RELEASE_CHECKLIST.md` | Segment A-G audit; implementation vs. research-evidence checklist |
| `README.md` (root) | Short pointer to the Member 3 docs/CLI (only if needed) |

## CLI exit codes (planned)

0 success (complete) | 1 runtime failure | 2 invalid usage/config | 3 blocked (unsupported mode) |
4 incomplete (partial/subset; not releasable) | 5 sealed-test access denied | 6 missing input | 7 checkpoint hash mismatch.

## Constraints and assumptions

- No new mandatory dependencies; no dependency or version changes. `pyyaml` is not a dependency, so workflow YAML
  is checked locally with the interpreter that happens to have it, not in tests.
- Real-model runs need Member 1's upstream DiMoDif checkout and checkpoint and Member 2's feature cache (not in CI).
  CLI tests use mock adapters via a patched model factory.
- Held-out test runs reuse `predictions.generate_predictions` -> Member 2 `open_test`; the CLI adds no bypass.

## Known blockers

- Three precision modes unsupported by the real adapter, so a real four-mode run exits 3 until Member 1 delivers.
- No real benchmark, accuracy or held-out evidence is produced or claimed by this work.
