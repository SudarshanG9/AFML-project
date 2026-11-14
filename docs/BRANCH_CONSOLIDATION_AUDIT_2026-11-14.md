# AFML project branch consolidation audit

Repository: SudarshanG9/AFML-project  
Audit date: 2026-11-14  
Default branch at audit: main  
Scope: all seven remote branches, relevant pull requests, backend source deltas, frontend integration and CI evidence.

## 1. Remote branch inventory

| Branch | Finding | Consolidation treatment |
|---|---|---|
| main | Current default. Already includes the Member 2 data branch and earlier Member 3 runtime branch. | Primary base. |
| frontend | Contains the Streamlit dashboard commit but is behind current main. | Preserve its frontend directory; do not let stale backend snapshots overwrite main. |
| member1 | Contains later Day B/Day C Member 1 implementation, scripts, configs, tests, reports and owner/release records. | Include unique source/docs/config/test files; resolve overlapping runtime files in favor of the later integrated Day C branch. |
| feature/member2-data | Already merged to main (PR #9). | No duplicate merge; retain current main. |
| feature/member3-runtime | Earlier Member 3 runtime PR #2 is merged to main. | No duplicate merge; retain current main. |
| feature/member3-segment-e | Ancestor/base of the later Day C integration branch. | Its runtime work is included through the newer Day C branch. |
| feature/member3-dayc-integration-review | Later Member 3 A–G CLI/runtime integration plus Member 1 four-mode adapter/intervention changes, docs, tests and CI. Draft PR #8 remains open against feature/member3-segment-e. | Select later code for overlapping paths; include distinct code, docs and tests. |

PR #9 (Member 2) is merged; PR #2 (earlier Member 3 runtime) is merged; PR #8 remains a draft integration PR. PR #8's Python 3.10 and 3.12 checks completed successfully on commit 46b385a770daf96320fd0d8455e44cb180f44624. Those checks are not a substitute for CI on the consolidated commit.

## 2. Conflict resolution

The consolidation is based on current main, not the stale frontend base:
- Retain current main versions for code already merged from Member 2 and the earlier Member 3 runtime.
- Bring in unique Member 1 Day B/Day C additions (code, tests, docs, configs and release metadata).
- For modified paths shared by member1 and feature/member3-dayc-integration-review, use the later Day C integration branch's blob. This includes the adapter/attention intervention, shared Member 2 evaluation/paired changes and Member 3 CLI/runner changes.
- Retain frontend/ from the frontend branch but do not copy its older pyproject.toml, schemas or backend files over current main.
- Exclude .DS_Store, __pycache__, .pyc, generated egg-info, ignored data/cache payloads, model checkpoints and local run outputs. Existing tracked data/artifacts in main remain unchanged.

## 3. Four-mode adapter source discrepancy

An older statement in docs/member3/FINAL_AUDIT.md describes the adapter as reference-only. The current integrated code contains DiMoDifAdapter.forward_with_mode(), calls intervene_dimodif_attention(), and dispatches non-reference modes through that intervention in predict_from_cached_features(). The four mode names are reference, low_precision, restore_value_output and restore_query. The implementation and tests are newer than that audit statement; the historical file is retained unchanged to preserve its checkpoint context.

The Member 3 CLI provides validate-config, run, predict, benchmark, inspect and synthetic --smoke. Real inference uses cached encoder features, not raw video/audio. Explicit loader formats are dimodif_npz (NPZ keys video_features/audio_features) and shared_pt (tensor keys video_tokens/audio_tokens).

## 4. Scientific and release boundaries

This consolidation is software integration, not a declaration that research is finished:
- Verify the external DiMoDif checkout, checkpoint hash and local feature cache on the machine performing the run.
- Cached-feature inference is not raw-video end-to-end. Raw-media decoding, face/mouth processing and encoder extraction remain separate preprocessing.
- Simulated INT8 remains floating-point fake quantization, not an integer kernel, energy result or guaranteed hardware speedup.
- Operator-only, cached-feature-model and preprocessing measurements are separate scopes. CPU evidence is not GPU/MPS evidence; null memory means unmeasured.
- Member 3's release acknowledgement has scope qualifications. It does not grant C9/C10 freeze approval or held-out test access.
- Held-out test execution remains subject to Member 2's frozen protocol/unseal/purpose controls and owner approvals. The dashboard execution form runs validation only.
- CI success verifies software behavior, not detector accuracy, causal claims, speedup or research release approval.

## 5. Frontend integration changes

The dashboard readiness detector checks the four-mode intervention files and Member 3 runtime CLI components. The Demo page validates local paths/configuration and invokes the backend CLI for validation-only cached-feature inference. Artifact discovery covers nested manifest locations and the Member 3 artifacts/member3/runs/ layout. The feature inspector recognizes both backend loader formats.

No raw media or checkpoint upload is introduced. Dataset clips, feature caches and upstream checkpoint data remain local.
