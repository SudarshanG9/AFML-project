# Phase 5 — final pre-merge release-readiness report

## 1. Release candidate identity

Verification date: 2026-11-15. Repository: [SudarshanG9/AFML-project](https://github.com/SudarshanG9/AFML-project).

| Reference | Identity |
| --- | --- |
| Working/push branch | `integration/afml-frontend` |
| Starting frontend commit | `41d9a84b217ae9fe1e88ef22a9ba9914e218fb81` |
| Pre-report implementation commit | `6de4accf09e4d47b9b1eef011e0ddf96a5d576a4` |
| Accepted backend foundation | `cf0bfc338bdcacec87f51e3313fc2ccf699a8c9a` |
| Original main | `a6c9bf786efe5e5290baf9ec24d5efb7d12fda80` |
| Original frontend source | `member3/frontend-only-updates` at `6de021adb93142c3f3a2383c7ae8f834101157c0` |

Remote references were fetched before changes and matched these baselines. The existing clean frontend integration checkout was reused. The accepted backend is an ancestor. No whole-branch merge, history rewrite, main update, or changes to PR #8/#10 were performed. The report's containing commit necessarily follows the implementation SHA; the final delivery records its full SHA and matching completed CI run, avoiding a self-referential report commit cycle.

## 2. Work completed

| Files | Original problem → correction | Validation |
| --- | --- | --- |
| `README.md`, `docs/README.md` | Inherited deferred/untouched frontend status contradicted integration → completed Phases 1–4, current Phase 5, launch commands and pending human merge decision | Link/path review; CLI help/smoke; `test_current_repository_instructions_match_app` |
| `frontend/README.md`, `frontend/INTEGRATION.md` | Current authority and verification interpretation needed clarification → Phase 5 pointer and dated past-check meaning | Link review; documentation regression test |
| `frontend/BACKEND_REVIEW.md` | Historical review pointed only to Phase 4 → header now points to current integration guide and Phase 5; historical body retained | Diff review |
| `frontend/app.py`, `frontend/avfuse_ui/ui.py` | Instructions referenced a nonexistent sidebar root field; upload wording obscured server transfer → use `AVFUSE_REPO_ROOT`, explain Streamlit server handling | Documentation regression; all-page AppTest |
| `frontend/avfuse_ui/backend_status.py`, `frontend/tests/test_backend_status.py` | Internal “ready” names could imply pretrained execution; parser failure could leave import success true → interface-availability names and failure resets | `test_interface_failure_never_means_runnable`, existing readiness tests |
| `frontend/avfuse_ui/run_inspection.py`, `frontend/avfuse_ui/ui.py` | Unhandled optional metadata and early return blocked inspection → bounded, filename-specific errors; independent stage-profile section; explicit manifest/finalization/integrity states | `test_run_states_preserve_stage_explorer`, `test_unreadable_optional_metadata_keeps_page_usable`, `test_optional_metadata_no_replacement` |
| Same run-inspection files | Stale verification could appear current → dated explicit verification receipts, cheap filesystem change invalidation, Overview past-check counts | `test_explicit_integrity_action_and_past_receipt`, `test_receipt_detects_changes_without_rehashing`; browser corruption exercise |
| `frontend/avfuse_ui/ui.py` | Environment command did not refresh discovery; malformed stage entries could break display → rediscover after completion, reject malformed entries visibly | `test_discovery_refreshes_all_runtime_artifacts`, `test_malformed_stage_entry_does_not_crash` |
| `frontend/avfuse_ui/ui.py` | Toolbar obscured top text; empty messages could confuse unselected with absent artifacts → top padding and explicit “selected” wording | Final 103-test frontend rerun; final spacing requires human visual recheck |
| `frontend/tests/test_run_inspection.py` | Missing focused regression coverage → 21 additional cases covering states, unreadability, receipts, refresh, readiness and documentation | Combined and targeted suites below |

Documentation review covered local Markdown destinations, launch/module/script paths, actual CLI help, feature contracts and external prerequisites. Historical reproducibility and research chapters were preserved. No current launch instruction requests a fresh held-out evaluation.

## 3. Architecture verification

The supported path is Streamlit UI → frontend service adapter → structured Member 3 CLI subprocess → Member 1 frozen pretrained detector using cached encoder features → run/prediction artifacts → Member 2 evaluation/reporting contracts → frontend visualization. Importability and parser construction do not prove model assets are verified, execution is authorized, inference completed, or artifacts passed integrity checks. Overview does not load a model.

Saved historical tables in `docs/results/` are explicitly separate from imported/new reports. Parsing an uploaded artifact does not promote it to accepted scientific evidence. The completed study is not reopened by the application.

## 4. Eight-page verification

All eight pages passed real Streamlit AppTest rendering without model assets or automatic inference. Browser inspection used local Streamlit with harmless fixtures; browser interaction coverage was partial, not equivalent to the automated results.

| Page | Automated result and inputs | Browser result / limitation |
| --- | --- | --- |
| Overview | Passed; interface discovery, unavailable assets, no model execution | Cards/readiness inspected; toolbar overlap found and padding adjusted; final spacing recheck pending |
| Dataset & features | Passed; manifest JSONL, NPZ/PT contracts, empty/malformed inputs | Empty upload controls and historical audit table inspected; actual file-picker upload not completed |
| Fusion diagnostics | Passed; missing diagnostic artifacts render empty states | Empty diagnostic sections inspected; no actual model diagnostics generated |
| Demo & predictions | Passed; validation-only structured arguments, prediction grouping, error codes and timeout | Form controls observed in DOM/accessibility state: LAV-DF, four modes, fixed hash, CPU/device and unchecked confirmation; actual submit interaction not confirmed |
| Evaluation | Passed; saved and imported results separated, malformed metrics visible | Saved tables and original ROC figures inspected |
| Benchmarks | Passed; record/sample contracts and missing-sample rejection | Saved 12 runtime records and 600 timing observations inspected |
| Runs & system | Passed; empty, incomplete, failed, finalized, corrupt, malformed/missing optional metadata and explicit verification | Synthetic run discovered/refreshed; explicit integrity pass; file corrupted; stale receipt warning and filename error; explicit integrity failure, no page crash |
| Artifact import | Passed; safe root resolution, JSONL/NPZ/PT and benchmark validation | Import table/controls inspected; upload gesture remains unverified |

Regression tests cover valid/invalid manifests, duplicate predictions, mixed identities, missing modes, NPZ keys/dimensions, PT layouts, malformed metrics and missing benchmark samples. Saved row counts are 20 test-overall, 36 manipulation, 80 paired and 20 validation-overall; exact parsed records are compared against original CSV values. Paired differences retain `mode_b - mode_a`. Display formatting does not overwrite source precision.

Run selection never hashes automatically. A missing check says “Not verified.” A structurally recognized manifest is finalization metadata, not proof of scientific success or integrity. Optional files remain untouched on parse failure. Receipt timestamps are past observations; filesystem stamps detect ordinary changes without rereading large artifacts and are not a continuous cryptographic guarantee.

Browser screenshots were retained locally outside Git for saved evaluation and synthetic corruption. Desktop and narrower views were inspected partially, but sidebar/navigation automation and the upload picker were unreliable. No complete light-theme or all-page narrow-screen acceptance was achieved. The synthetic run was moved out of the checkout after testing; no research outputs were created.

Required human checklist: launch using `frontend/README.md`; visit all eight pages at desktop and narrow widths; check top spacing, sidebar navigation, tables/charts, warning visibility and supported themes; upload harmless valid and malformed JSONL plus NPZ/PT fixtures; confirm actionable errors; exercise the unchecked/checked confirmation behavior without authorizing real assets; refresh artifacts and repeat explicit verification after modifying a synthetic file. Record browser/version, viewport and outcomes before removing the conditional gate.

## 5. Software test results

Local environment: Python 3.13.5, pytest 8.3.4, Streamlit 1.45.1, torch 2.11.0. Dependencies were already installed. A new clean local install was not attempted after the prior environment attempt exhausted disk space; hosted CI supplies fresh environments.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:frontend \
  python3 -m pytest -q -rs tests frontend/tests --junitxml=../phase5-test-results.xml
```

Result: **1,181 passed, 22 skipped, 0 failed, 0 errors**, 52.69 seconds (1,078 backend and 103 frontend passes). After the final spacing/empty-message changes, the complete frontend suite passed again: **103 passed**, 7.51 seconds. No backend file changed between these runs.

Every skip belongs to an existing environment/private-fixture category:

| Test file | Count | Exact missing prerequisite category |
| --- | ---: | --- |
| `tests/member1/test_dayc_colab_pilot.py` | 13 | Private reference predictions, C0 acknowledgement and G5 execution-context evidence |
| `tests/member1/test_dayc_execution_package.py` | 1 | Private `AFML_DayC_C8_2_CPU_Pilot_Mapped_v3_Verified.zip` execution archive |
| `tests/member1/test_dayc_progress.py` | 5 | Private recovery-engineering candidate source `dayc_recovery.py` |
| `tests/member2/test_audit_whole.py` | 3 | `ffmpeg`/`ffprobe` absent locally; real-file features, silence shortcut and script end-to-end cases |

Skipped cases were not reported as passes. Private fixtures were not imported. The local JUnit/log output records individual case names and reasons outside Git.

All six commands below exited **0**, with `PYTHONPATH=src:frontend`:

```bash
python3 -m avfuse.member3_runtime.cli --help
python3 -m avfuse.member3_runtime.cli run --help
python3 -m avfuse.member3_runtime.cli predict --help
python3 -m avfuse.member3_runtime.cli benchmark --help
python3 -m avfuse.member3_runtime.cli inspect --help
python3 -m avfuse.member3_runtime.cli --smoke
```

The smoke test is synthetic infrastructure evidence only. Required source checks passed:

```bash
python3 -m ruff check src/avfuse/member3_runtime tests/member3
python3 -m ruff check --select F frontend
python3 -m compileall -q frontend/app.py frontend/avfuse_ui
git diff --check
```

Full-repository style lint was not used as an acceptance gate or rerun in Phase 5; pre-existing out-of-scope findings documented in Phase 4 remain separate. Frozen scientific code was not reformatted.

## 6. CI evidence

[Branch workflow history](https://github.com/SudarshanG9/AFML-project/actions?query=branch%3Aintegration%2Fafml-frontend) identifies the final report-containing commit. `.github/workflows/ci.yml` is unchanged from Phase 4 and triggers on this branch with Python **3.10** and **3.12**, fresh CPU-compatible dependencies, Member 3 lint, backend tests, frontend tests, frontend correctness lint, compilation and CLI smoke.

At report-authoring time the report-containing commit is not yet pushed: both final-commit jobs are **pending verification**, not claimed green here. The delivery message will supply the exact final SHA, direct run URL, and separate completed outcomes for 3.10 and 3.12 after pushing and waiting. A failed/unavailable final run prevents a passed engineering gate regardless of this report's local results. [Phase 4 run 43714392883](https://github.com/SudarshanG9/AFML-project/actions/runs/43714392883) is historical only and is not used to certify Phase 5.

## 7. Research integrity

The protected-file test passed before and after corrections: **1 passed** on each run (final 0.09 seconds). All **359 protected file SHA-256 identities** match the unchanged Phase 4 inventory. Tree comparison to `cf0bfc338bdcacec87f51e3313fc2ccf699a8c9a` shows no changes in `src/`, `scripts/`, `configs/`, `contracts/` or `docs/results/`. The original Phase 1–3 and Phase 4 reports and protected inventory are unchanged from the starting integration commit. Research figures, evidence, release/provenance records and original checkpoint references remain unchanged.

The `third_party/dimodif` gitlink remains mode `160000`, revision `830fe1ba4a5bfb53b809331fdcbcce0a977cbd36`. No upstream backend source was replaced with older frontend-branch code.

The frozen checkpoint SHA-256 remains `e1e59006d1bb6ad36a3859af5486a6649ed41bba23258b7c41e24f873be7948e`; threshold remains **0.6627789735794068**. Modes remain `reference`, `low_precision`, `restore_value_output`, `restore_query`. Joint audiovisual self-attention and simulated floating-point quantization are unchanged. Overall precision damage and overall V/output superiority were not statistically established; intervals crossing zero do not prove equivalence. Source-disjoint groups do not guarantee speaker independence. CPU cached-feature timing is neither raw-video latency nor native INT8/GPU acceleration. No new scientific calculation or held-out evaluation was executed.

## 8. Security and local asset boundaries

Uploads are handled by the Streamlit server; remote hosting entails network transfer to that server. Parsing does not automatically publish to GitHub or third parties. Authorized roots and symlink checks remain enforced; safe loaders and format/size contracts remain covered by tests. No arbitrary checkpoint upload or raw-video inference is introduced.

Execution uses structured subprocess arguments with validation-only manifests, the supported LAV-DF checkpoint and frozen hash, explicit device, no silent CPU fallback, no test-unseal flags, duplicate-run rejection and explicit confirmation. Backend codes 0–7 and timeout remain distinguishable in tests. Page rendering does not execute inference. Asset verification, authorization, inference and integrity are distinct states.

Only focused source, current documentation, tests and this report are committed. Synthetic browser fixtures, screenshots, logs and asset inspection outputs remain outside Git. No media, checkpoint, per-clip research output or private dataset is included.

## 9. Reproducibility status

A clean checkout can install the project/frontend dependencies, run synthetic tests and smoke, launch the dashboard and view saved aggregate evidence. Actual detector execution additionally requires the external pinned DiMoDif checkout, checkpoint/configuration, validation manifest, complete compatible feature cache, supported device and explicit authorization.

Read-only inspection of the authorized local handoff found these files and computed their SHA-256 identities without loading a model:

| Asset | Bytes | SHA-256 |
| --- | ---: | --- |
| `third_party/dimodif/src/models.py` | 7,187 | `e74b5e3b5fcee24353f9940dddd48d9ac1982ced725664db2518f63a6d853778` |
| LAV-DF `.pth` checkpoint | 4,566,082 | `e1e59006d1bb6ad36a3859af5486a6649ed41bba23258b7c41e24f873be7948e` |
| Matching model configuration JSON | 5,464 | `bea128dc7dad6b72bc0487dcc45d03ae3322e48fd597de1a5cd83ad036f30651` |
| `data/manifests/manifest_whole_val.jsonl` | 418,047 | `22d1d9e02403cfdbb05f2116da16bc4015f540547a4ac00bf6b7127c119e359c` |

A complete authorized feature cache was not established; historical synthetic fixture files are not a substitute. No exact validation run was separately authorized. CPU software smoke succeeded; this is not a real-model device validation.

**Real-model validation demonstration not executed; authorization or required external assets were unavailable.**

The missing `.gitmodules` mapping is pre-existing. The explicit external clone/checkout process in [the reproducibility guide](../09_REPRODUCIBILITY_AND_USER_GUIDE.md) remains the setup path; preserve pinned revision `830fe1ba4a5bfb53b809331fdcbcce0a977cbd36` and do not assume `git submodule update` alone installs it. Full original-study reproduction was not performed.

## 10. Remaining issues

| Classification | Location / scope | Required next action |
| --- | --- | --- |
| P0 | None identified by executed checks | Any later preservation or CI failure blocks review acceptance |
| P1 — important release verification blocker | `frontend/app.py`, `frontend/avfuse_ui/ui.py`; browser upload/confirmation/navigation/responsive/theme behavior and final top spacing | Human execute the concrete checklist in section 4; automated rendering does not close this gap |
| External dependency/authorization limitation | External feature cache, checkpoint/source setup and exact validation authorization | Project lead supply/verify permitted assets and authorize a specific validation-only demonstration if required; never reopen held-out study |
| External dependency limitation | `third_party/dimodif`, missing `.gitmodules` | Follow documented explicit external checkout at the pinned revision |
| Environment limitation | Local clean install, ffmpeg/ffprobe, private historical fixtures | Fresh hosted CI verifies installation; skips remain disclosed, do not invent private fixtures |
| P2 — nonblocking improvement | Out-of-scope pre-existing lint; browser automation stability | Address separately without cosmetic edits to frozen scientific code |

## 11. Final release decision

**CONDITIONAL — NEEDS HUMAN VERIFICATION**

Executed software tests, CLI checks and preservation checks pass. Remaining browser interaction/visual verification is important and incomplete, and no authorized real-asset demonstration was executed. Final-commit hosted CI must additionally finish successfully and be identified in the delivery. The branch can be reviewed by the project lead with these explicit conditions; this report does not authorize merging or claim unconditional deployment readiness. `main` remains reserved for the project lead's decision.
