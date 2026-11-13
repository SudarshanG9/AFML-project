# Release-readiness checklist

Two separate questions. **A ticked implementation item does not accept any research claim.**

## A. Implementation completion (software)

- [x] A-D infrastructure present with tests (see FINAL_AUDIT.md)
- [x] Segment E orchestration: loaders, per-mode status, predictions, model benchmark, run integration
- [x] Segment F CLI with documented exit codes and sealed-test safeguards
- [x] CI workflow committed (`.github/workflows/ci.yml`), needs no private data
- [x] Documentation: CLI, architecture, troubleshooting, audit
- [x] Local full test suite passing; Member 3 Ruff clean
- [ ] CI green on GitHub (first run pending)
- [ ] Python 3.10 verified (CI matrix)
- [ ] Repository-wide Ruff cleanup (other members)

## B. Blockers before any real four-mode experiment

- [ ] Member 1: `low_precision`, `restore_value_output`, `restore_query` implemented in the real DiMoDif attention
- [ ] Member 1: dense/SDPA parity evidence for the model-level interventions
- [ ] Member 1: optional conversion-only callable (otherwise `conversion_overhead_ms` stays null)
- [ ] Real checkpoint SHA-256 matches `DIMODIF_LAVDF_SHA256` (`--require-lavdf-hash`)

## C. Research evidence (owned by the team, not produced by this software)

- [ ] Validation four-mode run finalized and `inspect`ed (`complete_four_mode_validation_run: true`)
- [ ] Member 2 readiness/orientation check on the real predictions
- [ ] Validation threshold / restoration decision frozen
- [ ] Benchmarks on target hardware (GPU/MPS) with disclosure that INT8 is simulated
- [ ] Single held-out test run, ledger shows exactly one purposeful access
- [ ] Review of all claims by the owners (1, 2, 3)

## D. Release approval

- [ ] Approver, date and release id: **pending. Not recorded here, none invented.**

A run directory, a passing CI badge or `exit 0` is never a release approval.
