# Phase 4 — frontend and cleaned backend integration

## Integration status

Completed with documented validation limitations. The implementation is on `integration/afml-frontend`; final release approval is deferred. No new real-input validation or held-out evaluation was executed. The final pushed commit is supplied in the delivery message; this report cannot contain its own commit hash.

## Source branches and selective import

Verified after `git fetch origin --prune`:

| Source | SHA |
|---|---|
| Accepted backend `integration/afml-final-release` | `cf0bfc338bdcacec87f51e3313fc2ccf699a8c9a` |
| Latest frontend `member3/frontend-only-updates` | `6de021adb93142c3f3a2383c7ae8f834101157c0` |
| Older frontend, not used | `c0db8ce5f8dee055cb6b171bb76e87d81bf2ad52` |
| Main, unchanged | `a6c9bf786efe5e5290baf9ec24d5efb7d12fda80` |

An isolated worktree and branch were created directly from the accepted backend. Only `frontend/` and `.github/workflows/ci.yml` were restored from the latest frontend source. No whole-branch merge occurred. No member or cleanup branch was merged again. The frontend-source change to `tests/member3/test_dayB_model_benchmark.py` was already identical to the accepted backend and was not copied.

Reviewable implementation commits:

- `344013a`: exact selective frontend and candidate workflow import.
- `d8db160`: frontend service adaptations and CI guards.
- `9e68cfa`: saved evidence presentation and status corrections.
- `25697d7`: integration/rendering tests and backend identity inventory.
- Subsequent documentation commits record this report and current navigation.

## Architecture and contract verification

`frontend/app.py` binds the server-selected repository and discovers local artifacts. `avfuse_ui/ui.py` presents eight pages and explicit confirmation. `backend_status.py` imports the real interfaces without loading a model and invokes Member 3 by subprocess argument lists. `data_access.py` validates schemas, delegates NPZ/PT inspection and benchmark validation to the existing backend, and reads committed CSV presentations. See [frontend integration details](../../frontend/INTEGRATION.md).

Flow: UI confirmation → frontend path/manifest/identifier guards → Member 3 `validate-config --json` → Member 3 `run --json --split val --require-lavdf-hash` → existing Member 1 adapter → existing prediction/run artifacts → refreshed discovery → separately requested integrity verification. There is no frontend detector implementation.

Checked interfaces include Member 1 `DiMoDifAdapter.forward_with_mode`, strict checkpoint loading and frozen parameters; Member 2 manifest/prediction schemas; shared precision modes; Member 3 parser commands `validate-config`, `run`, `predict`, `benchmark`, `inspect`, and synthetic `--smoke`; NPZ/PT loaders, benchmark cross-field validator, and run integrity verifier. All production source bytes remain unchanged. Original masks, five joint attention layers, projections and detection head retain their accepted semantics.

## Corrections and evidence

| Files | Correction | Tests |
|---|---|---|
| `backend_status.py` | Boolean Git dirty state; actual imports/parser instead of source-string readiness; separate readiness stages | `test_backend_status.py`, `test_integration.py` |
| `backend_status.py`, `ui.py` | Explicit frozen LAV-DF hash, device/modes, validation-only manifest, safe IDs/roots; structured status separate from diagnostics; codes 0–7 and timeout; duplicate session/existing-run guards | Wrapper mocks across all exit codes, malformed JSON, timeout, missing inputs, invalid IDs/modes/device, test rejection |
| `data_access.py`, `ui.py` | Actual NPZ/PT loader validation; 32 MiB compressed/expanded inspection bounds; operator-authorized roots and symlink containment | Wrong keys/rank/width/dtype, empty/nonfinite arrays, PT batch mismatch, traversal/symlink tests |
| `data_access.py`, `ui.py` | Preserve run/split/checkpoint prediction groups; malformed multi-file import blocks comparison | Prediction contract/group tests and rendered malformed import test |
| `ui.py`, `data_access.py` | Saved aggregate CSV and imported evaluation views separated; exact provenance and paired B−A sign retained | Exact CSV equality tests; rendered counts; selected imported metrics remain selected |
| `ui.py`, `data_access.py` | Saved CPU tables separated from imports; no combined headline latency across workloads; actual benchmark sample/cross-field checks | Valid synthetic benchmark, invalid statistics and missing external samples tests |
| `ui.py` | Discovered manifests are not integrity passes; explicit verifier action; partial/failed run diagnostics | Actual finalized synthetic artifact verifies, then corruption is detected |
| `app.py`, `.streamlit/config.toml` | Server-selected repository, refresh button, loopback default and upload cap | AppTest refresh and all-page rendering |
| `.github/workflows/ci.yml` | Python 3.10/3.12 CPU PyTorch; integration branch trigger; all backend tests with conditional skips; explicit frontend tests and CLI smoke | Workflow reviewed; hosted status recorded below |

The source frontend's Artifact import page used an unsupported `st.table` argument; rendered testing exposed it and the page now uses the supported dataframe interface. No backend bug was repaired by modifying frozen code.

## Page-by-page verification

All eight pages were actually rendered using `streamlit.testing.v1.AppTest`, without model assets or inference. These are Streamlit rendering checks, not browser visual approval.

| Page | Observed check |
|---|---|
| Overview | Renders source/interface versus asset states, Git dirty status, artifact inventory and completed-study conclusion |
| Dataset & features | Missing-input view renders; actual loader/manifest contracts exercised with temporary synthetic fixtures |
| Fusion diagnostics | Renders available reports/missing states; theoretical memory and ablation limitations remain explicit |
| Demo & predictions | Empty-asset view renders; mocked wrapper validates exact arguments; malformed mixed import blocks success/comparison |
| Evaluation | Saved tables render all 156 rows (20 overall, 36 manipulation, 80 paired, 20 validation); imported JSON remains separate; malformed metrics show errors |
| Benchmarks | Saved tables render 12 runtime records and 600 timing observations; empty imported view remains separate |
| Runs & system | Missing-run view renders; backend integrity API passes synthetic bundle and detects corruption |
| Artifact import | Renders supported formats and server upload boundary without crashing |

Browser upload interaction, visual layout across screen sizes, and confirmed real-model execution remain manual/deferred checks. Reproducible procedures are in [INTEGRATION.md](../../frontend/INTEGRATION.md).

## Scientific integrity

[PHASE_4_PROTECTED_FILES.json](PHASE_4_PROTECTED_FILES.json) records 360 baseline entries: 359 SHA-256-checked files and the original DiMoDif gitlink. The preservation test passes. Production code, scripts, schemas, research configurations, historical records, accepted result tables and original Phases 1–3 report remain unchanged. The gitlink mode/object is retained. Only current root/docs navigation and new Phase 4 material supplement backend documentation.

Four modes, checkpoint identities, frozen threshold, source groups and accepted cohorts are unchanged. No final result was recomputed. The UI reports that overall precision damage and V/output superiority were not statistically established; a CI crossing zero does not establish equivalence. CPU cached-feature timing is not raw-video latency or native INT8 acceleration. No unavailable memory result becomes zero. Source disjointness is not verified speaker disjointness.

## Test evidence

Local environment: Python 3.13.5, PyTorch 2.11.0, Streamlit 1.45.1, pytest 8.3.4.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:frontend python3 -m pytest -q tests frontend/tests --junitxml=../phase4-test-results.xml
```

**1,160 passed, 22 skipped, 0 failures, 0 errors; 44.16 seconds.** This includes 1,078 backend and 82 frontend passes. These are current execution results, not reused cleanup counts.

Skips: 13 in `tests/member1/test_dayc_colab_pilot.py`, 1 in `test_dayc_execution_package.py`, 5 in `test_dayc_progress.py` depend on unavailable historical private fixtures; 3 in `tests/member2/test_audit_whole.py` require ffmpeg/ffprobe. No broad test exclusions were added.

Additional executed checks:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:frontend python3 -m avfuse.member3_runtime.cli --help
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:frontend python3 -m avfuse.member3_runtime.cli --smoke
python3 -m ruff check src/avfuse/member3_runtime tests/member3
python3 -m ruff check --select F frontend
python3 -m compileall -q frontend
```

All above completed successfully. Synthetic smoke scores are not research outputs. `git diff --check` passes. Full-repository Ruff was also examined: 329 findings, including 52 frontend style findings; it is not claimed clean. CI explicitly scopes full configured lint to Member 3 and correctness (`F`) lint to the frontend, with syntax compilation. It does not suppress a failing command or rewrite protected source for style.

A fresh virtual environment was created and the requested editable/backend and frontend installation was attempted. Download/install failed with **OSError 28: no space left on device**. Only that incomplete task-created virtual environment was removed. Local test evidence therefore comes from the installed environment, not a successful clean installation. CI provides the clean Python 3.10/3.12 installation checks when available.

Hosted CI status: pending initial push; see subsequent verification note and delivery message for actual observed status. No hosted pass is asserted here.

## Acceptance checklist

| Requirement | Evidence/status |
|---|---|
| Correct base/latest frontend | Exact baseline parent and selective import recorded above |
| Backend integrity/Members 1–3 | 359 protected file hashes pass; real interfaces and synthetic tests pass |
| Eight pages | All rendered by AppTest; browser visual approval deferred |
| Saved research results | Exact CSV record equality/provenance checks and rendered counts pass |
| Validation-only controls | Argument-list tests across all statuses; no unseal/protocol/purpose flags assembled |
| Feature/artifact contracts | Actual loaders, benchmark validator and integrity API tested |
| Backend regressions/frontend tests | 1,160 passes; 22 documented existing skips |
| CI | Both Python versions configured; hosted result reported separately |
| Scientific controls | No production/evidence changes, no held-out evaluation |
| Main and source branches | Not modified by this work; only target branch is pushed |
| Final release gate | Explicitly deferred |

## Remaining limitations and Phase 5

| Severity | Exact files/scope | Limitation and next action |
|---|---|---|
| Validation limitation | `.github/workflows/ci.yml` | Local clean install blocked by disk space; verify clean hosted Python 3.10/3.12 jobs before release |
| Validation limitation | `frontend/tests/test_ui.py`, `frontend/INTEGRATION.md` | AppTest does not establish browser visual quality or every upload-widget gesture; perform documented browser checks |
| Authorization boundary | `frontend/avfuse_ui/backend_status.py` | Real-input validation demonstration not executed; supplied real assets were not authorized for execution. Obtain explicit validation-only asset/run authorization |
| Existing private dependencies | Three Member 1 test modules named above | Restore authorized historical fixtures only if required for final provenance review |
| Environment limitation | `tests/member2/test_audit_whole.py` | Install ffmpeg/ffprobe for the three skipped audit checks |
| Style debt | Existing backend files and `frontend/avfuse_ui/*.py` | Full configured Ruff is not clean; review style debt separately without rewriting frozen scientific code |
| Provenance limitation | `frontend/avfuse_ui/ui.py` | Prediction rows lack dataset identity; consult run provenance before any cross-run scientific comparison |

Before merging into main: review hosted results for both Python versions, complete browser checks, optionally perform an explicitly authorized real-validation demonstration, review final provenance and private-fixture requirements, and obtain separate final release approval. Do not reopen accepted held-out evaluation. PRs #8/#10 remain untouched.
