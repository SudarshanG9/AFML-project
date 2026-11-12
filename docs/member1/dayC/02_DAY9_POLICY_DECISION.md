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
