# Segment F + G handoff (final)

**Status: Checkpoint 4 reached. Segments F and G implemented, locally validated. Nothing committed or pushed.**

- Base: `f66af16` (`origin/feature/member3-segment-e`; `902cd7b` is its ancestor). Working tree = base + patch.
- Plan: `SEGMENT_F_G_IMPLEMENTATION_PLAN.md`. Audit: `FINAL_AUDIT.md`. Release checklist: `RELEASE_CHECKLIST.md`.

## Measured results
- `python -m pytest -q`: **580 passed, 0 skipped, 0 failed** (541 baseline + 39 CLI tests).
- `ruff check src/avfuse/member3_runtime tests/member3`: passed. Repo-wide ruff: 70 pre-existing findings in other members' files (untouched).
- `git diff --check`: clean. Final patch applies to a clean `f66af16` worktree (`git apply --check`).
- CI YAML parsed with PyYAML; **workflow not executed on GitHub; Python 3.10 not run locally.**

## Done
CLI (`validate-config`, `run`, `predict`, `benchmark`, `inspect`, `--smoke`), exit codes 0-7, sealed-test safeguards
(+ validation preflight), 39 CLI tests, CI workflow, CLI/architecture/troubleshooting docs, final audit, release checklist.

## Not done / blockers
Real DiMoDif run, GPU/MPS timings, three precision modes (Member 1), GitHub CI first run, repo-wide ruff cleanup,
release approval (pending; none recorded).

## Resume / review
```
git checkout -b member3-fg f66af16
git apply artifacts/member3/checkpoints/member3_FG_final.patch
python -m pytest -q && python -m ruff check src/avfuse/member3_runtime tests/member3
```
Do not repeat completed work; start from the unchecked items in RELEASE_CHECKLIST.md.
