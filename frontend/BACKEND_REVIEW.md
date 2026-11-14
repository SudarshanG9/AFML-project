# Backend integration review for the AVFuse-CausalQ dashboard

Repository: SudarshanG9/AFML-project  
Review date: 2026-11-14  
Branches inspected: main, frontend, member1, feature/member2-data, feature/member3-runtime, feature/member3-segment-e, feature/member3-dayc-integration-review.

The dashboard is a local Streamlit interface over the repository's real Python packages and local artifacts. It does not contain a second detector implementation, upload raw media to a server, or fabricate predictions and measurements.

## Consolidated architecture

| Owner | Integrated backend | Dashboard use |
|---|---|---|
| Member 1 | src/avfuse/member1_fusion/ | DiMoDif checkpoint adapter, attention interventions, precision modes, attention and masking diagnostics |
| Member 2 | src/avfuse/member2_data/ | manifest/cache contracts, split and shortcut audits, prediction checks, sealed-test controls, evaluation and paired analysis |
| Member 3 | src/avfuse/member3_runtime/ | explicit feature loaders, cached-feature inference, CLI, benchmarks, stage profiling, run provenance and integrity |
| Frontend | frontend/ | artifact inspection, environment check, validation-only real cached-feature CLI workflow and synthetic smoke control |

## Integrated backend behavior

### Member 1

The DiMoDif adapter retains the original pretrained model and checkpoints. forward_reference() performs the unmodified baseline. forward_with_mode() temporarily applies intervene_dimodif_attention() to the pretrained transformer's existing attention layers, and predict_from_cached_features() dispatches non-reference modes through that intervention. The intervention calls the shared fusion_attention() implementation and restores original forward methods after use. The agreed modes are reference, low_precision, restore_value_output and restore_query.

Quantization remains floating-point fake quantization; it is not integer tensor storage, an INT8 kernel, energy evidence or a guaranteed hardware speedup.

### Member 2

The package provides manifest and prediction contracts, cache helpers, dataset/split construction, data-quality and shortcut checks, prediction readiness checks, metrics, paired contrasts and sealed-test controls. Evaluation labels are joined from the manifest rather than copied into prediction rows. Test scoring remains governed by the explicit protocol/unseal/purpose workflow.

### Member 3

The CLI exposes validate-config, run, predict, benchmark, inspect and the synthetic --smoke diagnostic. The real path loads an explicitly named checkpoint and caller-supplied whole-video manifest/feature root. It does not silently select a checkpoint or fall back to CPU.

The feature loader requires an explicit format:
- dimodif_npz: <feature-root>/<split>/<clip_id>/features.npz, containing video_features and audio_features; these map to the adapter's video_tokens and audio_tokens inputs.
- shared_pt: the manifest .pt feature path, containing video_tokens and audio_tokens.

The CLI writes prediction and benchmark artifacts into a run folder and records provenance. It measures cached-feature work, not raw-video end-to-end latency. Operator, cached-feature-model and one-time preprocessing scopes must remain separate.

## Dashboard behavior

- Overview inspects the checked-out source tree and reports implementation presence separately from local run artifacts.
- Dataset & features validates manifests and source/identity split groups; .pt and .npz payloads are inspected without model inference.
- Fusion diagnostics reads Member 1 ablation, attention-boundary, theoretical attention-memory and padding/mask reports.
- Demo & predictions validates inputs and invokes the backend CLI for validation-only cached-feature inference. It runs validate-config before run, does not expose the held-out test split, and accepts no raw media or arbitrary checkpoints.
- Evaluation displays Member 2 structured metrics and labels Markdown validation summaries as documentation, not a fresh run.
- Benchmarks discovers JSONL from dedicated benchmark folders and finalized run folders.
- Runs & system reads Member 3 run manifests, environment reports, stage profiles and integrity information.
- Artifact import explains supported files and privacy boundaries.

The app does not transmit local data to GitHub or another service. Uploaded JSON/JSONL and feature archives are parsed in the Streamlit session, not copied into the repository.

## Branch audit and documentation discrepancy

main already includes the Member 2 data branch and earlier Member 3 runtime branch. feature/member3-segment-e is an ancestor of the later feature/member3-dayc-integration-review branch. The consolidated branch also contains later Member 1 Day B/Day C code, scripts, configs, tests and release records, plus the frontend.

An older statement in docs/member3/FINAL_AUDIT.md describes a checkpoint where the adapter was reference-only. The current integrated adapter contains the intervention dispatch described above, and PR #8's Python 3.10 and 3.12 checks passed. The historical audit is retained unchanged; the later code and tests supersede that specific implementation-status statement. See docs/BRANCH_CONSOLIDATION_AUDIT_2026-11-14.md.

## Remaining research and release gates

1. Verify the external DiMoDif checkout, checkpoint hash and local feature cache on the machine performing the run.
2. Reproduce and retain validation evidence with immutable inputs and run provenance.
3. Keep precision/threshold policy validation-only and obtain applicable C9/C10 approvals before held-out test access.
4. Perform the single authorized held-out evaluation only after protocol and owner approvals are in place, using Member 2's frozen procedure.
5. Report hardware and memory evidence only for devices/scopes actually exercised. CPU evidence is not GPU evidence; null memory means unmeasured.
6. Raw-media decoding, face/mouth processing and encoder computation remain upstream preprocessing. The dashboard/CLI operate on cached encoder features.

The Member 3 release acknowledgement has scope qualifications and does not grant C9/C10 freeze approval or held-out test access. Passing CI verifies software behavior, not detector accuracy, causal claims, hardware speedup or research release.
