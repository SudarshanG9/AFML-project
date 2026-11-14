# Day9 policy decision — source implemented, research decision pending

Status: **B3 SOURCE COMPLETE / JOINT B2+B3 AUDIT PENDING / NO POLICY FROZEN**.

The user explicitly requested B3 source as one block before joint B2/B3 review. Actual B2 statistics remain pending the self-contained job in `COLAB_B2_B3_COMBINED_AUDIT.md`. H2/H3 verdicts and CIs must come from its real output; no conditional example from the runbook is adopted as a decision. The local proposal at `/Users/sudarshan/Documents/GitHub/AFML-project/artifacts/dayC/member1/day9/B3_SOURCE_20261112T161540549058Z/JOINT_UNAPPROVED_PROPOSAL/policy_proposal.json` deliberately has `selected_interpretation=null`, `H2=null`, `H3=null` and `ready_for_freeze=false`.

## Implemented decision rules

D = BCE(lowp) − BCE(reference). Positive harm needs D>0 and paired CI lower bound>0. Rv/Rq are lowp minus restored BCE. H3_direct is BCE(query) − BCE(value/output); direct paired evidence is required. A positive restoration point alone or one arm significant/another not does not establish differential superiority. The machine proposal requires coherent manipulation-specific direction and positive-harm/recovery/direct evidence for the original H3 claim; the lead reviews that interpretation before any freeze. If D<=0, share recovered is undefined and no known damage is claimed rescued. CI overlap with zero never establishes equivalence.

All four configurations remain in the eventual frozen held-out experiment regardless of the interpretive policy: reference, low_precision, restore_value_output, restore_query. The code has no sample, skip-error, threshold-retune, overwrite, resume, mode-selection or test-unseal bypass on its production score CLI. No training stage or custom integer kernel exists.

## Actual available cost facts

Verified typical p50 ms: reference 118.4543895; low_precision 161.434481; restore_value_output 119.91815; restore_query 167.4740195. Typical E_value = −41.516331 ms; E_query = 6.0395385 ms. Value/output E is also nonpositive on short and long. Therefore the planned comparison per strictly positive added millisecond is undefined; this is retained as a valid negative/unavailable normalization outcome. It is not repaired by a new timing run, changed denominator or unsupported equal-cost claim. Raw short/typical/long samples and p95 are preserved in the B2 return audit. Timing uncertainty is not automatically in BCE CIs. Fake INT8 may cost more than FP32, and cached-model timing excludes raw preprocessing.

## Real inputs still required before a freeze-ready proposal

Member2 must provide an actual external acknowledgement with: release ID/source/time; exact original canonical/calibration/checkpoint hashes; the final validation-fixed threshold; original source/identity group definition and analysis-stratum method; 1000 replicates/seed0; all four modes and probability-of-fake orientation; signed ready-test file path/hash/full count as **opaque metadata only**; the sealed one-time evaluation purpose; and the known limitations/test-summary exposure disclosure. An unsigned JSON or existing provisional threshold is not an approval.

Member3 must acknowledge the exact runtime/device/backend and three approved workload identities, all four modes, actual new benchmark evidence hashes, batch1, 10/50, CPU synchronization, complete raw samples, fake-precision conversion overhead, separate operator/cached-model scopes, CPU memory null/not_measured and separate raw preprocessing costs. Existing measured records do not forge that acknowledgement.

The actual input registry must cover original weights/config/model source, four boundary definitions/all-five-layer semantics, schemas and complete scoring/guard/runtime-helper source, all four validation outputs, calibration, full metric/paired/per-type/cost evidence, owner releases, and the opaque ready-test manifest identity. A fresh successful source-test report is bound to its actual JUnit/log/source hashes. Any changed code after freeze is a violation. Frozen configs and anchor use exclusive creation; a partial failed freeze is retained and cannot authorize testing.

`freeze_day9 --action propose` creates new draft files only. `--action finalize` requires real C8B acceptance of the actual B2 packet and an actual C9 ACK bound to the exact fully ready proposal, plus every real owner/data/code dependency. It never creates those ACKs. C10 authorization is separately bound to policy/run/anchor digests. The source-preparation instruction is not any of these scientific approvals.

## Open limitations, never omitted

Frozen pretrained DiMoDif uses joint AV self-attention; all-layer interventions are cumulative and deeper states differ. Fake symmetric per-tensor INT8 is floating-point emulation, not an integer deployment speedup. Validation checkpoint-selection bias remains. Source-video grouping does not establish independent speaker separation. Checkpoint-training overlap is unverified. Trailing-silence shortcut warning remains (about 0.682 validation AUROC and an earlier inspected test summary about 0.660); a matched-control result is not invented. Historical test-label summary statistics were inspected; no claim of completely untouched test-derived information is made. No test-model feedback may select/tune this policy. The lead must explicitly review proceeding with those disclosures. A fresh holdout would require a separately approved protocol change.

No ready-test rows/features, model test scores, C9 final configs or actual C10 ledger were accessed/created during B3 source preparation. Synthetic test fixtures are isolated and explicitly labeled TEST_FIXTURE. Actual policy selection, owner endorsements and held-out runtime/cache arrangement remain pending.


## Latest status: joint evidence audited; real owner releases and lead decision pending

The initial source-only/pending-statistics description above is retained history. Actual B2 statistics have now returned and been audited. H2 positive damage and the originally motivated H3 restoration advantage are not supported; direct V/output-vs-query comparison is inconclusive, with all main BCE intervals spanning zero. No equivalence claim is made. A new reference-preferred **review draft**, with real validation findings and no final approval, is at `/Users/sudarshan/Documents/GitHub/AFML-project/artifacts/dayC/member1/day9/JOINT_RETURN_AUDIT_20261112T204538343795Z/REFERENCE_PREFERRED_REVIEW_DRAFT/policy_proposal.json`. The actual remaining local freeze dependencies are owner2 release, owner3 release, opaque ready-test manifest identity and lead acceptance/policy approval. The complete joint report is `/Users/sudarshan/Documents/GitHub/AFML-project/artifacts/dayC/member1/day9/JOINT_RETURN_AUDIT_20261112T204538343795Z/B2_B3_JOINT_REVIEW.md`. All four eventual test contrasts remain unchanged; no freeze/test action has occurred.


## Member 2 release intake — original JSON pending verification

The latest lead-supplied resume request reports release `M2-REL-20261113-01`, expected file `m2_handoff/m2_release.json`, expected SHA-256 `6ec2a65c90b3d91b93522fe8abbf09f914ad71499863e3c6ea32ff126b58d454`. The authentic JSON was not found in the clone or prior project folder; Downloads could not be searched due to OS access denial. These details are reported assertions, not verified owner evidence. No owner acknowledgement has been manufactured.

The reported approved cohort is the original canonical manifest's `split == "test"` subset, 2,000 clips / 500 per manipulation category. Do not create or demand a separate test-only manifest. The original canonical bytes can be hashed as an opaque file; no test rows, labels, IDs or features may be parsed before C10. The reported 2,000 matching folders alone do not establish file identity. Authenticate the release and available provenance first.

The reported scientific corrections are zero validation/test source overlap against the complete LAV-DF training split; speaker overlap remains unverified; validation trailing-silence control showed no detector performance drop; duration was not controlled. Preserve historical caveats above but do not promote these corrections to authenticated findings until the original release is verified. Historical test-summary exposure remains disclosed. Accepted validation numbers, threshold and four precision definitions are unchanged.

The existing source preparation assumes a separate ready-test input. After authentic release verification, reconcile the canonical test-split contract with the Member1-owned freeze/runner guards using synthetic tests before binding new source hashes. Do not generate a substitute manifest or weaken cohort/seal checks. Held-out Drive resolution and runtime must be explicitly specified before C9; current validation-only mapping is insufficient. This is pending implementation reconciliation, not a freeze-ready proposal.

Member 3's genuine final release is still missing. It must identify its source/release, approve the CPU/runtime/backend and four configurations, exact short/typical/long workload/input and benchmark hashes, raw samples, batch 1 / warmup 10 / repeat 50, fake-INT8 overhead, cached-model/operator/preprocessing scopes, CPU memory limits, variability and approved Day10 runtime arrangement.

C9 and C10 remain unapproved. No test data was opened or attempt claimed. Intake evidence: `artifacts/dayC/member1/day9/OWNER_RELEASE_INTAKE_20261113T052338064126Z/release_intake.json`.


## Authentic Member 2 release verified

Release `M2-REL-20261113-01` is now preserved at `docs/member1/dayC/releases/M2-REL-20261113-01/m2_release.json`; exact SHA-256 `6ec2a65c90b3d91b93522fe8abbf09f914ad71499863e3c6ea32ff126b58d454`. This supersedes the earlier missing-original-file intake status. All provenance hashes and existing manifest identities match. The released cohort is the canonical test split, reported as 2,000 clips; no separate manifest will be invented. No test rows were parsed.

Member2 corrects source-overlap limitations: its supplied audit reports zero validation/test source overlap against the full LAV-DF train split; speaker-level overlap remains unverified. Its acknowledged validation silence control reports no detector drop for the three earlier validation configurations; no query-specific silence control is claimed. Duration remains uncontrolled. These are owner findings; no local repetition was performed. Test-summary exposure remains disclosed. The supplied coverage report does not establish complete per-file test tensor identity or full-array verification for all files (2,800/3,200 checked).

Release verification clears the missing Member2 artifact dependency, but production integration is still pending: the legacy validator assumes exact shorthand strings and a separate ready-test file, whereas the authentic release specifies descriptive strings, a boolean no-test-predictions flag and the original canonical test-split selector. Preserve original release bytes and extend explicit contracts with synthetic tests before freezing; do not forge a normalized owner release or bypass guards. Member3 and the exact approved held-out runtime/path arrangement remain pending. C9/C10 are not authorized.


## Member2 source integration complete — Member3 remains the external dependency

The original release is now accepted by the production Member1 validator without rewriting its bytes. Canonical test-split selection, explicit frozen read-only Drive resolution, C10-only complete cache integrity inspection and five-file final handoff are implemented. No separate ready-test manifest is needed; the historical registry/CLI name is a hash/path alias of the canonical file. The original scoring adapter, quantization math, owner implementations and accepted validation outputs remain unchanged.

Execution/provenance scope and the remaining historical cache-identity limitation are detailed in `C10_CANONICAL_EXECUTION_PLAN.md`. The cache policy is a proposal for C9 review, not actual authorization. Real runtime availability and cache identity have not been inspected remotely; no test access or attempt occurred. Latest full suite: **405 passed**, with logs and fresh source bindings under `artifacts/dayC/member1/day9/M2_INTEGRATION_FINAL_20261113T054042774678Z`.
