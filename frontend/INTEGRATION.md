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
