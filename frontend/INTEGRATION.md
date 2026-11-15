# Frontend integration contracts

The branch starts at cleaned backend `cf0bfc338bdcacec87f51e3313fc2ccf699a8c9a`. Only `frontend/` and the candidate CI workflow were imported from `6de021adb93142c3f3a2383c7ae8f834101157c0`; no source-branch merge or stale backend import occurred. The frontend-source Member 3 benchmark test already matched the cleaned backend.

## Service flow

1. Overview imports existing package interfaces and parses commands without constructing a model.
2. A confirmed Demo request passes through server-authorized path, manifest, cache-existence, mode, device, identifier and frozen-checkpoint guards.
3. The service invokes `python -m avfuse.member3_runtime.cli validate-config ... --json` and then `run ... --json`. Both commands use `--split val`; no unseal controls are exposed.
4. The existing Member 1 adapter performs inference under Member 3 orchestration. Exit codes 0–7 retain their backend meanings: complete, failure, usage, blocked, incomplete, sealed denial, missing input, checkpoint mismatch. Timeout is separately reported as 124.
5. Discovery refreshes run directories and prediction paths. Integrity verification is a separate explicit backend action.

The six prediction writer fields are retained. The existing schema's optional fields are not falsely described as forbidden; labels and ground truth remain rejected. Canonical mode names come from `avfuse.shared.contracts.PRECISION_MODES`.

## Adapter corrections and tests

| Area | Implementation | Evidence |
|---|---|---|
| Git/source status | `backend_status.py` Boolean dirty state; actual imports/parser; readiness stages | `test_backend_status.py`, `test_integration.py` |
| Execution | Validation-only arguments, explicit frozen checkpoint/device, safe IDs/roots, structured JSON, existing-run refusal | `test_integration.py` subprocess mocks |
| Feature inspection | `data_access.py` delegates to actual NPZ/PT loaders, bounded archives and safe weights-only load | `test_data_access.py`, `test_integration.py` |
| Published tables | Exact CSV reads retaining source hashes and paired B−A sign | `test_integration.py`, rendered row counts in `test_ui.py` |
| Imports | Schema and backend cross-field benchmark checks; prediction identity isolation | `test_integration.py` |
| Protected backend | SHA-256 inventory from the accepted baseline | `test_backend_preservation.py` |
| UI | All eight pages and empty imported-result views render via AppTest | `test_ui.py` |

## Reproducible manual checks

AppTest cannot simulate all browser file-upload controls or assess visual layout. Launch locally, visit each page, and verify:

- Import one valid and one malformed JSONL together: show the filename error and disable combined comparison.
- Import a structured metrics JSON in Imported mode: show its provenance without saved table replacement.
- Upload an invalid NPZ (token keys instead of feature keys): reject loader compatibility.
- Select an incomplete run, inspect its failure summary, and explicitly verify a finalized synthetic run before and after corrupting a copy of an artifact.
- Refresh discovery after a new synthetic fixture appears.
- Confirm an unavailable device never invokes a fallback. Do not submit real inference without separate authorization for the assets.

Real-input validation demonstration is not executed by this integration. The user has not authorized execution against supplied real assets. The [Phase 5 release-readiness report](../docs/integration/PHASE_5_RELEASE_READINESS_REPORT.md) records current browser checks, final provenance review and any remaining limitations. Real-input demonstration requires separate exact-asset authorization; merge approval remains with the project lead. Neither main nor the accepted backend branch is changed.
