# DeepTrace integration record

## Phase 0: audited baseline

Repository: https://github.com/SudarshanG9/AFML-project
Local checkout: `path-b-review`; base: `5945ae1910153659c322ce7f5123912140fae625`.
The initial main working tree was clean. Remote fetch failed in the sandbox; the requested escalation was denied. Remote freshness is unverified. Work proceeds on `feature/deeptrace-precision-workbench` without merging unrelated branches.

Baseline command (repository root):

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:frontend ../phase4-venv/bin/python -m pytest -q tests frontend/tests
```

Result: **1181 passed, 22 skipped in 52.24s**. Existing skips concern private historical Member 1 fixtures and missing ffmpeg/ffprobe checks. No skipped check establishes real inference.

The original eight renderers cover overview, manifests/cache QA, attention diagnostics, inference/predictions, separate evaluation, benchmarks, run integrity and imports. `app.py` discovers artifacts on every rerun. The original unkeyed form loses hidden widget state and reserves a session run ID before frontend preflight. The wrapper returns unstructured early failures, verifies file existence before `validate-config`, then invokes `run`. `validate-config` reads neither model nor features. The original UI misattributes preflight failures to CLI exit.

Architecture: Streamlit → UI → argument-list subprocess adapter → Member 3 CLI → original Member 1 model/precision interventions + Member 3 NPZ/PT loaders → predictions, benchmarks, provenance → frontend discovery and Member 3 integrity verifier. Member 2 evaluation is a separate CLI, requiring labels and a matching frozen protocol for accepted-threshold reproduction. No inference command computes forensic metrics.

Reusable components: bounded JSON/JSONL reads, schema/group guards, exact saved CSVs with original source hashes, backend feature loaders, benchmark cross-field checks, optional metadata reads and dated integrity receipts. The new primary UI has Experiment, Results and About; technical explorers remain expandable components.

Execution risks: external source/weights/config/cache are absent from the checkout; NPZ uses split/clip paths rather than manifest PT placeholders; successful software validation cannot establish tensor compatibility or full-cohort completion; incomplete benchmark sample verification must remain visible; reports cannot be attached to unrelated predictions. New orchestration stays under `frontend/`. Protected source, scripts, schemas, frozen configs and evidence are compared against the baseline and existing SHA inventory.

Phase 1 introduces the light three-section layout, persistent experiment draft, historical-results distinction and educational limitations. Execution stays disabled until the readiness bridge is integrated.

## Phase 2: readiness and execution

Phase 1 commit `328c33f` pushed successfully; 98 frontend tests passed in 4.01s. Phase 2 adds `workbench_service.py` for asset inspection and readiness coordination, preserving the Member 3 CLI as the runner. All cache paths are checked; missing caches report exact examples (first 20), expected/found/missing counts and selected format. One bounded real-loader cache inspection establishes sample compatibility only. A separately authorized optional one-sample forward uses the existing `predict --max-clips 1` CLI in a temporary directory; it neither reserves the experiment ID nor publishes research scores. Device availability, frozen checkpoint bytes and DFD configuration/source are checked before CLI launch. Corrupt cache loader errors become useful readiness failures.

`backend_status.py` returns structured phase/status/cli_launched fields for preflight, validation, execution, timeout and invalid responses. Execution requires a safe explicit run ID and rechecks assets. No session reservation occurs before execution; backend existing-directory protection remains. UI draft/authorization survives navigation; changing material inputs clears authorization/readiness. File-stat changes invalidate the readiness receipt. Inference completion explicitly leaves label-grounded evaluation absent.

## Phase 3: source-bound results and evaluation

Phase 2 commit `d254b35` pushed successfully; 120 frontend tests passed in 6.37s. Phase 3 renders compact saved performance, manipulation recall with original intervals and workload-specific CPU cost tables. New runs are isolated by run/split/checkpoint with prediction schema, summary hash/count and benchmark sample verification. Synthetic/unspecified origin cannot become detector evidence. Missing modes, metrics and memory remain unavailable. Original reports, failure metadata, explicit integrity receipts and exports are accessible in details. Imported reports are independent, never automatically joined to a run.

The optional Member 2 handoff invokes only `scripts/m2_evaluate.py evaluate --split val --protocol ... --n-boot 1000 --seed 0`. It requires complete validation inference, unchanged labeled validation-only manifest, exact prediction coverage, matching checkpoint/manifest protocol hashes and the original validation-fixed threshold from the release record. It offers no calibration/unseal operation. A historical canonical-manifest protocol will be refused for a different validation-only manifest; the frontend does not rewrite it. Reports are stored outside finalized inference bundles under `artifacts/member3/evaluations/<run>/<unique-id>/`; a receipt binds exact predictions, manifest, checkpoint, protocol and metrics bytes. Only a matching receipt/report is shown as associated evaluation. This receipt is provenance association, not authorship authentication.

Phase 3 frontend verification: **153 passed in 33.31s**. Includes an actual invocation of the existing Member 2 evaluator on eight explicitly synthetic labeled fixture rows (one mode, fixed threshold, 1000 bootstraps), confirming report loading and unchanged inference-bundle integrity. This verifies evaluation wiring, not real pretrained inference or real detector performance. Negative tests cover mixed run/checkpoint/split, corrupt predictions, invalid probability/metric ranges, missing modes/labels, protocol mismatches, evaluator timeout/nonzero/malformed output, receipt corruption and absent evaluation. No licensed model was executed.

## Phase 4: engineering handoff

Repository: `SudarshanG9/AFML-project`; working checkout `path-b-review`; branch `feature/deeptrace-precision-workbench`; base `5945ae1910153659c322ce7f5123912140fae625`. The final phase commit is reported in the delivery message because this document is part of that commit. No main branch update, merge, force-push or history rewrite was performed. Fetch escalation was denied at audit time, so the original remote-main freshness was not established; the feature-branch pushes are independently confirmed.

### Architecture before and after

The original eight-page app coupled configuration, validation and execution into one form. It used the correct backend CLI but consumed session IDs before preflight and could lose hidden widget state. The new three-section app separates durable configuration, asset/software readiness, optional representative forward, explicit validation execution and source-bound result inspection. All model, quantization, evaluation and benchmark calculations still belong to the original Members 1–3 code. The frontend coordinates existing APIs/CLI commands and presents their artifacts. It introduces no web API, new model or evaluator.

### Changed files and ownership

| File | Responsibility and existing backend connection |
|---|---|
| `app.py` | Three-section routing, Experiment default, server-selected repository, refreshed artifact discovery and preserved result selection |
| `.streamlit/config.toml` | Light/purple theme; existing loopback/upload limits retained |
| `avfuse_ui/ui.py` | Minimal setup/preview, four canonical modes, durable draft and material-input authorization reset, truthful workflow stage, readiness/execution controls and short About; reuses existing optional explorers |
| `avfuse_ui/backend_status.py` | Structured preflight/CLI phase attribution, validation-only argument lists, completion/identity response guards, timeout/nonzero distinctions and post-run discovery |
| `avfuse_ui/workbench_service.py` | Actual manifest registry, authorized asset/readiness coordination, one-sample existing-CLI forward option, per-run results association and guarded Member 2 validation-evaluation handoff |
| `avfuse_ui/workbench_results.py` | Compact saved/new/imported result sources; original metrics/strata/cost, sample-aligned score differences, missing data, per-run evaluation outcomes, dated integrity checks and available exports |
| `avfuse_ui/data_access.py` | Additional finite probability and optional metric-block validation; original artifact readers, schemas, loaders and source-bound CSV reads remain authoritative |
| `tests/test_ui.py` | Three-page rendering, saved/imported separation, draft/result-source/authorization persistence, missing-asset failures, enabled execution only after authorization/readiness and incomplete outcomes |
| `tests/test_integration.py` | Existing wrapper tests use real-loader-compatible synthetic caches/config/source while subprocess/model execution stays mocked; success fixtures include exact run identity |
| `tests/test_workbench_service.py` | Missing cache counts/paths for both formats, malformed/unauthorized assets, frozen hash/device, reusable failed IDs, receipt freshness, representative forward CLI and execution error states |
| `tests/test_workbench_results.py` | Authentic saved table/chart structure; run identity/corruption/missing modes; evaluator protocol/label/error handling; exact report receipt association; actual Member 2 synthetic-fixture evaluation |
| `README.md` | Installation, launch, configure → validate → execute → results flow, asset requirements, evaluation protocol constraints, privacy and scientific limitations |
| `INTEGRATION.md` | This audit, phase checkpoints, exact evidence and remaining verification boundaries |

No changes outside `frontend/` appear in `git diff --name-only 5945ae1`. The unchanged existing protected-file inventory test verifies all 359 recorded file hashes. No checkpoint, dataset, prediction fixture or generated evaluation artifact is staged.

### End-to-end use

Open Experiment → choose accepted evidence or new validation inference → select the four modes → select a discovered actual validation manifest or enter authorized asset paths → Check Readiness → optionally authorize a genuine one-sample compatibility forward → explicitly authorize and Run Experiment → open that run's generated results → inspect valid predictions, associated evaluation if actually performed and verified scoped timings → inspect metadata/failures, explicitly verify integrity and export available evidence. A failed preflight leaves the ID reusable. An existing run directory still prevents duplicate execution. Result buttons target the most recent actual run directory; evaluation notifications are scoped to the selected run.

### Exact verification evidence

Local environment uses the existing `../phase4-venv/bin/python` environment (not a newly claimed clean install). All commands below run from the repository root unless stated otherwise.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:frontend ../phase4-venv/bin/python -m pytest -q tests frontend/tests --junitxml=/private/tmp/deeptrace-final-tests.xml
```

**1235 passed, 22 skipped, no failures; 69.30s.** This comprises 1078 backend and 157 frontend tests. The final small result-notification/target-selection refinements were additionally verified by:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:frontend ../phase4-venv/bin/python -m pytest -q frontend/tests/test_ui.py frontend/tests/test_workbench_results.py
```

**47 passed; 23.45s.** Earlier final focused readiness/UI/wrapper checks: **86 passed; 5.66s**.

The same 22 baseline skips remain: 13 historical private-package tests in `test_dayc_colab_pilot`, 1 private execution-archive test in `test_dayc_execution_package`, 5 private recovery-candidate tests in `test_dayc_progress`, and 3 `test_audit_whole` checks requiring ffmpeg/ffprobe. They were not suppressed or reclassified as passes.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:frontend ../phase4-venv/bin/python -m ruff check --select F frontend
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:frontend ../phase4-venv/bin/python -m avfuse.member3_runtime.cli --help
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:frontend ../phase4-venv/bin/python -m avfuse.member3_runtime.cli --smoke
git diff --check
git diff --name-only 5945ae1 -- src contracts scripts configs docs artifacts checkpoints data
```

Correctness lint, help, synthetic diagnostic smoke and whitespace checks passed; the protected-boundary diff is empty. Full-repository style lint and a fresh dependency installation were not claimed. No CI workflow was changed; new hosted CI results are not claimed.

The app was actually launched from `frontend/` on `127.0.0.1:8517`, using `--server.fileWatcherType none` after the sandbox denied the local port and macOS filesystem watcher. The authorized launch ran successfully. Browser screenshots at **1440 × 1000** verified the light lavender sidebar, three-section navigation, setup/preview cards, compact saved table, manipulation chart and workload-specific cost table. Browser interactions verified new-validation asset controls, an intentionally missing manifest producing **“No experiment CLI was launched”**, disabled execution and retained manifest/run ID after About → Experiment navigation. AppTest covers authorization persistence/reset, all three sections, registered manifests, saved/imported empty states and execution state transitions. Browser layout checks do not establish every upload gesture or every responsive breakpoint. The localhost preview was left running with Experiment selected; temporary viewport overrides were reset.

### Commit and push checkpoints

| Phase | Commit | Validation | Remote status |
|---|---|---|---|
| 1 — minimal workbench | `328c33f` | 98 frontend passes | Pushed |
| 2 — execution/readiness | `d254b35` | 120 frontend passes | Pushed |
| 3 — forensic/runtime evidence | `89aba05` | 153 frontend passes | Pushed |
| 4 — final usability, tests and docs | See delivery commit | 1235 passes / 22 skips; final targeted 47 passes | Final delivery records push result |

### Remaining limitations

**Implemented and verified:** saved accepted evidence, three-section UI, durable drafts/authorization, precise missing-asset failures, existing validation CLI command construction and negative states, identity-isolated artifacts, actual backend integrity checks, one existing Member 2 evaluator run on synthetic labeled fixtures and a live browser preview.

**Implemented but not real-asset tested:** full pretrained validation inference, real selected-mode representative forward and associated real benchmarks/evaluation. No configured licensed source/weights/cache set was supplied for an explicitly authorized exact run. The checkout's external DiMoDif/checkpoint location is unpopulated; existing pilot manifests alone do not supply executable assets. Missing requirements are trusted model source, matching frozen LAV-DF weights plus configuration, validation manifest and complete compatible caches.

**External/protocol constraints:** evaluation needs labels, full coverage and an existing protocol bound to the exact validation-only manifest/checkpoint and original fixed threshold. The historical canonical-manifest protocol cannot be silently reused for different manifest bytes. No compatible protocol is manufactured. Larger-than-32-MiB representative caches exceed the current bounded readiness inspection limit. Fresh installation, hosted CI and all upload gestures/breakpoints remain unverified here.

**Outside scope:** raw-media ingestion/preprocessing UI, native INT8 deployment, new quantizers/models/evaluation algorithms, research changes, test-set access, protected-main updates and universal performance/deployment claims.
