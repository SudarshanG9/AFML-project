# Day C C0 baseline inventory and research review

Status: **BLOCKED / C0 HUMAN RESEARCH REVIEW REQUIRED**. C0 source preparation is complete; original Day 8 has not started. This document is not an approval or a policy freeze.

## Authorized local root and instruction scope

The user explicitly selected `/Users/sudarshan/Documents/GitHub/AFML-project`. All Day C creations, test fixtures, logs and edits are confined to that clone. The user's request authorizes preservation, filename normalization and Member 1 Day C implementation. The Day C Markdown supplies the sequential execution contract and research gates. The original PDF supplies original Days 8/9/10 responsibilities; previous Day B G8/G9 technical gates do not complete those days. There were no Git index changes, commits, branch changes, remote calls, downloads or account operations.

## Baseline and ownership

- Branch: `member1`; HEAD: `4af1615c10cd35f34081f99be42f1913e0727ea6`, exactly the runbook snapshot.
- Pre-existing `git status --short`: ` ? third_party/dimodif`. The third-party submodule had pre-existing untracked content and was left untouched.
- Core precision/attention/adapter/intervention hashes match the accepted Day B sources. The local LAV-DF checkpoint, upstream config and original model source match accepted runtime provenance.
- Read-only comparison `git diff --stat main...HEAD` reports 78 changed files, 10,270 insertions and 24 deletions; the reverse main-side comparison is empty. These are existing branch differences, not Day C edits. No remote state or unperformed ancestry calculation is asserted. See `branch_comparison.json`.
- **234 enumerated pre-existing files are byte-identical after C0**, including Member 1 production code, Member 2/3 sources, contracts, existing tests/docs/configs, predictions/reports/benchmarks, checkpoint/config/model source and `.git/index`. See `preservation.json` and `baseline_before.json` for exact scope.
- The older "Current incomplete sections" paragraph in `docs/EXPERIMENT_PROTOCOL.md` is stale relative to the implemented adapter and accepted evidence. It was not rewritten; no shared documentation exception was inferred.

## Filename normalization

The production source filenames and all three canonical full-validation prediction paths are already normalized. Their original aliases have identical bytes and are retained because historical experiment records refer to them. No old file needed renaming, moving, overwriting or deletion. The missing canonical `artifacts/predictions/restore_query_val.jsonl` remains absent. Existing pilot query outputs are not the full Day 8 result.

| Historical filename | Canonical filename | Verified action |
|---|---|---|
| `artifacts/predictions/dayB_reference_val.jsonl` | `artifacts/predictions/reference_val.jsonl` | identical; both preserved |
| `artifacts/predictions/dayB_low_precision_val.jsonl` | `artifacts/predictions/low_precision_val.jsonl` | identical; both preserved |
| `artifacts/predictions/dayB_restore_value_output_val.jsonl` | `artifacts/predictions/restore_value_output_val.jsonl` | identical; both preserved |

All new Day C files use the exact ownership paths from the runbook; no code has been duplicated from exported bundles.

## Inspected local input paths and SHA-256

Hashes identify **file bytes**, not a full feature-cache digest. The full canonical manifest was hashed for provenance only; its rows and held-out feature paths were not inspected. Validation-only schema/group checks use unchanged Member 2 functions.

| Input | Project-relative existing path | SHA-256 |
|---|---|---|
| `calibration` | `artifacts/reports/dayB_validation_calibration.json` | `358c037242f6980fed2a8d0ff0ef830f17b4789676b391c8f6dff245842b37dd` |
| `canonical_manifest` | `data/manifests/manifest_whole.jsonl` | `8339b2476a71e62c6c6fca7d937ba4cc7a704b09d26f08506c20d1f10de8f3dd` |
| `checkpoint` | `third_party/dimodif/ckpt/dfd/dfd_lavdf_reduceonplateau_64_4_1_5_True.pth` | `e1e59006d1bb6ad36a3859af5486a6649ed41bba23258b7c41e24f873be7948e` |
| `dayb_audit` | `artifacts/dayB/final_review/DayB_FINAL_CPU_20261112T031627Z_91e5f723/DAYB_FINAL_AUDIT.json` | `7a55e3c62d957d4eae3f0df67e58668fde89fe8536342fd561495bd73b94cd68` |
| `dayb_context` | `artifacts/dayB/final_review/DayB_FINAL_CPU_20261112T031627Z_91e5f723/received_evidence/provenance/g5/execution_context.json` | `c03488cb787b1230bd86274dc7a36d39983574c7e2bc996c931dcfc8a835bd7c` |
| `environment_reference` | `configs/dayB/environment.reference.json` | `460b0786144d2e3d9f70e1f1e9e2ae0d988ff2bfe88a34e36375c6057432b18f` |
| `model_config` | `third_party/dimodif/ckpt/dfd/dfd_lavdf_reduceonplateau_64_4_1_5_True.json` | `bea128dc7dad6b72bc0487dcc45d03ae3322e48fd597de1a5cd83ad036f30651` |
| `precision_modes` | `configs/dayB/precision_modes.json` | `8c2be706b0998ccbf9f9198da2e8d1dcbeffff289937308cfd6a5f4ce7363a23` |
| `predictions_low_precision` | `artifacts/predictions/low_precision_val.jsonl` | `79bb4e462254beba2812bbf2ce56ad2f8cbe3c15fa128a3db5542d843978c636` |
| `predictions_reference` | `artifacts/predictions/reference_val.jsonl` | `e146432023638635b53dc517f3b4b06ecd87a6599efe3709ed14b145c7458ff3` |
| `predictions_restore_value_output` | `artifacts/predictions/restore_value_output_val.jsonl` | `2deb811328045c1ca0f04ea87db9c693fb776d84079e375ae6c42dee456441ea` |
| `ready_val_manifest` | `artifacts/dayB/final_review/DayB_FINAL_CPU_20261112T031627Z_91e5f723/received_evidence/provenance/g4/dayB_val_ready.jsonl` | `c79ad908517878ebf0f3aa0e0dac053a01d9d5c4eb264583db26096985890481` |
| `source_attention` | `src/avfuse/member1_fusion/attention.py` | `8ef4f1f85c48963d65dc99ec415ab34a170d493fe51336635d221c396fee8578` |
| `source_dayB_inference` | `scripts/dayB_inference.py` | `a0fd7257206bcc7c7c3b08c45684fb1ddba287d6ad018cdc4bfe4819c403399a` |
| `source_dayB_model_benchmark` | `scripts/dayB_model_benchmark.py` | `5dc20853e976ceadb2f3020ab4782eac371c48f183d6a43afb1520f917d4781c` |
| `source_dimodif_adapter` | `src/avfuse/member1_fusion/dimodif_adapter.py` | `256e5620a00f8c4cb846c30100d1870d2c4c9db5c59f7f6a0d1d25eb8a130ae0` |
| `source_dimodif_intervention` | `src/avfuse/member1_fusion/dimodif_intervention.py` | `3a0fe003ca3081884f5b38b362145df43a8d3cd3cdd9bee4a0db05e69bb131cf` |
| `source_precision` | `src/avfuse/member1_fusion/precision.py` | `a0e54283deb806e5541b80f162354c22ed85f48582917e603eed10690a997cd9` |
| `upstream_source` | `third_party/dimodif/src/models.py` | `e74b5e3b5fcee24353f9940dddd48d9ac1982ced725664db2518f63a6d853778` |
| `validation_manifest` | `data/manifests/manifest_whole_val.jsonl` | `22d1d9e02403cfdbb05f2116da16bc4015f540547a4ac00bf6b7127c119e359c` |

## Completed C0 implementation

- `configs/dayC/member1/experiment_spec.json`
- `src/avfuse/member1_fusion/dayc_guards.py`
- `scripts/dayC_member1/__init__.py`
- `scripts/dayC_member1/preflight.py`
- `tests/member1/test_dayc_guards.py`
- `docs/member1/dayC/00_BASELINE_INVENTORY.md`

- C0.3: exact scientific experiment definitions, four existing modes and 20 inspected baseline-input hashes. No verdict, approval, frozen policy or assumed held-out category count.
- C0.4: repository containment; traversal/symlink checks; streaming hashes; exclusive atomic JSON/JSONL writes with sibling temporaries and hard-link commit; nonfinite rejection; scope-bound human ACK validation; phase/split guards; exact frozen-input verification.
- C0.5: baseline preflight reuses unchanged Member 2 manifest/prediction/group validators. It checks all 1,200 validation IDs and four 300-item categories, checkpoint/provenance, canonical score files, calibration, query output absence, local feature-path resolution and runtime differences. It never opens feature contents or constructs a detector. Later phase CLI values deliberately return BLOCKED until their sequential implementations and human gates are available.
- C0.6: this inventory, a separate immutable review context, pending C0 request and full created-file/hash inventory.

## Tests and invocation evidence

Existing baseline Member 1 suite: **48 passed**, 0 failed, 0 skipped (`baseline_member1.xml`). Final Member 1 regression plus C0 synthetic tests: **103 passed**, 0 failed, 0 skipped (`final_member1.xml`). The command is recorded in `final_tests.json`; fixtures and temporary outputs are inside this evidence directory. The full team suite was not rerun because the change is confined to C0 Member 1 orchestration.

The first metadata-only preflight invocation failed while serializing a Path into JSON. It ran no inference and finalized no report. Its log is retained as `baseline_preflight.log`. Path serialization was fixed, a regression test added, and the final-name dangling-symlink case was also covered. Subsequent reports have new names; nothing was overwritten or hidden. These were source verification checks, not research experiment retries.

Final preflight: **35 checks PASS; runtime comparison BLOCKED**, exit code 2 as documented for readiness blockers. It confirms three full 1,200-row prediction files, identical validation cohort/checkpoint, correct six-field score schema, no labels in predictions and exact accepted hashes. The score/category check uses validation metadata only.

- Final report: `artifacts/dayC/member1/preflight/C0_20261112T053940Z_6f9b69d7/baseline_preflight_final.json`
- Final report SHA-256: `195da7e92267ab7b7cee09ef56f5b1b70cf8b38971acd30b302887c40315a2f8`
- Specification SHA-256: `36e45ffbe48a452090854a7fa0939371acb86578d7c973f6f4da22c53d6496ab`
- Test logs and JUnit: same evidence directory, `final_member1.log`, `final_member1.xml`.

## Existing scientific findings requiring lead acknowledgement

These numbers are accepted prior Day B results, not new Day C measurements:

- Validation BCE: reference 0.060848822; low precision 0.060805678; V/output restored 0.060849594.
- Paired damage (lowp minus reference): -4.3144353457e-05; 95% source-group CI [-0.000624812977708, +0.000464924289069].
- Paired recovery (lowp minus V/output): -4.39166091455e-05; CI [-0.000476094793791, +0.00042285118253].
- No established positive precision damage or restoration rescue; this is not an equivalence claim. Share recovered is undefined for nonpositive damage.
- Typical prior matched CPU p50: lowp 118.984093 ms; V/output 168.305224 ms. These are floating-point fake INT8 measurements, not integer-kernel acceleration. CPU peak memory was not measured.
- Prior scientific acknowledgement remains pending in the audited artifacts. Directory selection is not a scientific ACK. No signed approval has been generated.

Source: `docs/dayB/RESULTS.md` and `artifacts/dayB/final_review/DayB_FINAL_CPU_20261112T031627Z_91e5f723/DAYB_FINAL_AUDIT.{md,json}`.

## Actual blockers and unresolved research limitations

1. Historical ready manifest: exactly 1,200 validation clips, 300 per type, accepted hash. **0/1,200 feature paths resolve inside this project**; every supplied path is an external Colab path. No external feature was opened. The canonical manifest was not repurposed and the local pilot was not substituted. A user-managed, authorized full local cache and correctly reconciled ready manifest are needed before real Day 8 entry.
2. Local CPU runtime: macOS arm64, Python 3.13.0, PyTorch 2.10.0; accepted Day B CPU runtime: Linux x86_64, Python 3.13.18, PyTorch 2.14.1+cpu. Installed package versions and default thread counts also differ; full differences are in the report. No global environment was changed. Reestablish an approved comparable environment, or obtain an explicit research protocol decision before any inference.
3. C0 acknowledgement is missing. This request does not finalize the provisional threshold, approve a changed runtime or authorize Day 8 full-validation inference, policy freeze or held-out testing.
4. Day 8 still needs the full query-control run, direct four-mode source-group validation analysis and genuine matching four-mode short/typical/long cost evidence. Member 2 owns analysis/labels; Member 3 owns benchmark methodology and release. No teammate implementation or result has been impersonated.
5. Preserve validation checkpoint-selection bias, unresolved training overlap and speaker-disjointness, silence/duration shortcut concerns, cumulative five-layer intervention limits, and previously disclosed historical test-label-derived summary exposure. No new held-out labels/statistics were inspected here; test precision-policy selection must remain validation-only.

## Required human decision and next step

Review the baseline, preserved Day B findings, source-only C0 work and blockers. To acknowledge this evidence, supply exactly:

`ACK DAYC C0: ACCEPT DAYB EVIDENCE AND LOCAL BASELINE`

The pending request references a separate review-context digest; it is **not** an approval. The user's actual response must be retained verbatim in a new approval record when supplied. The next numbered phase is **C8.1**, only after C0 acknowledgement **and** satisfaction of Day 8 local-cache and approved-environment entry conditions. C8A remains a separate stop before the full 1,200-clip query experiment.
