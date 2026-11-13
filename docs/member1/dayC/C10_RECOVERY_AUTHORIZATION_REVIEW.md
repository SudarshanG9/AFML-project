# C10 recovery authorization review

Status: DECISION PREPARATION ONLY. No recovery approval, code change, package rebuild, new attempt, remote access or inference is performed by this review.

## Incident and preserved identities

Original run: `C10_20261113T191638908596Z`. Its original ZIP, internal hash inventory, saved run records, approval and two-event ledger were checked against the preserved originals. All match. The ledger contains one opened attempt followed by one failure; every mode has zero predictions. Canonical metadata access occurred during that authorized attempt; do not describe the test as never opened.

Original archive SHA-256: `6c5e6aa061d772895ad006f160940d28784ff1440474f7cd88e2794715c0a074`.

Failure: `features.npz` unavailable at execution time at `/content/afml_drive/MyDrive/AFML_Project/dimodif_emb/test/0068b4934f169b13/features.npz`. This classifies the observed failure, not its historical cause. It does not prove data deletion, corrupt features, a wrong mount or a transient Drive failure.

Original C9 policy SHA-256: `4951455481405602e835902da04478afa10eb9707935e6208aed6a813a0401bf`.
Original test-run SHA-256: `4ddc0696c91c2595b6592c3a4db5f3303f8899fbe6883bfcf07c6e244969d1e1`.
Original anchor SHA-256: `5a6c2b8b9ac141ef958ed2aeea4e16122a521fe05fc2df8165e3f63c80c689a5`.
Original execution ZIP SHA-256: `1c230a84b1c860dee3b6b305eda49db219475da1317c924c6b612a61a3b84d2d`.
Original lock SHA-256: `33b152e449c7a0d797b75c763d657b2d1f180e19fe6527a457fb3690690e3fc7`.
Original two-event ledger SHA-256: `1d12976032b58787cde9ee6bd33a06ae9e9d7aae5b8c57e9b7a97c60fb59b1c1`.

Preserved incident directory: `artifacts/dayC/member1/day10/C10_20261113T191638908596Z/`. Existing frozen configs, sources, receipt, lock, ledger, original logs and archive remain unchanged.

## New evidence and limits

The lead's supplied recovery message reports a separate read-only Colab audit: original canonical cohort 2,000; complete feature/metadata pairs 2,000; missing feature files 0; missing metadata files 0. The originally unavailable file is now present at the exact frozen path, 897,566 bytes, with adjacent `meta.json` 588 bytes. These are user-reported observations; no raw audit output or independently inspected remote cache is claimed here. No further cache audit is requested.

This establishes reported current existence/coverage, not historical availability, the cause of the failure, file-content validity, or original-byte equivalence. Historical per-test-file hashes remain unavailable. Existing full-array/metadata checks and acquisition hashing still have to occur under an authorized recovery execution; existence checks do not replace them.

## Source findings: why a new run ID is insufficient

- `dayc_heldout.claim_attempt` exclusively creates the single project-level `test_attempt.lock` and `test_attempt_ledger.jsonl`. The paths do not vary with run ID.
- `dayc_freeze.verify_release` rejects any prior attempt/output through `prior_test_artifacts`; a failed attempt is not an exception. Its dry-run `AUTHORIZED_METADATA_ONLY` does not mean another launch is permitted.
- `colab_day10.run_authorized` exclusively copies its approval to the fixed `approvals/C10_ack.json`; that already exists after the first launch.
- The output directory supports distinct run names, but `run_day10_test` exposes no approved recovery, force, resume or retry option.
- `GATE_ACKS` defines C9/C10/C10R and earlier gates, not an incident-recovery gate. The current code cannot authenticate a newly invented recovery ACK.
- `build_day10_package` also calls complete-proposal validation, which rejects prior test artifacts. Simply rebuilding the old package is not a supported recovery mechanism.

Deleting or moving the lock, removing the receipt, changing the run ID, recreating an empty workspace or monkeypatching those guards would evade the incident record. None is proposed or authorized.

## Recommended minimum legitimate change proposal

Preserve the original scientific C9 decision, checkpoint, four modes, threshold, canonical cohort, CPU semantics and cache mapping. Preserve every original C9/C10 artifact byte-for-byte as the historical parent release. Current reported availability needs no path rewrite, feature regeneration, new research or validation rerun.

The existing executable contract cannot simply be reused: it binds the original guard source and permits only the original one-attempt workflow. A separately approved, versioned operational recovery amendment is needed. It must provide:

1. A recovery-aware authorization/verification path that requires this exact parent failure/archive/ledger/lock, zero scoring, the unchanged scientific inputs and an explicitly approved allowance of exactly one additional attempt. It must reject successful, unknown, already-recovered or mismatched parents; never merely skip `prior_test_artifacts`.
2. A distinct recovery identity, receipt path, exclusive lock, ledger and output namespace. Verify and reference the immutable parent ledger; record recovery events in a separate append-only ledger. No original ledger rewrite. Repeated or concurrent launch of that approved recovery is rejected.
3. Hash-bound original parent context and new operational source/test/execution identities. Reuse the existing numerical scorer and saved-output/schema checks; do not create another model implementation. Recovery auditing and review packaging must return BOTH parent and recovery histories, and verify that only the specifically authorized parent failure precedes the single recovery attempt.
4. A recovery-specific package-builder/preflight path that explicitly validates the retained parent incident rather than requiring an empty history. No arbitrary extra attempt allowance or generic force option.
5. Synthetic tests for invalid/stale/missing approval, parent tampering, nonzero/successful parent, wrong hashes, concurrent/repeated attempts and recovery failure; keep numerical tests/evidence unchanged unless affected. Update source proof only for the actual authorized changes.

The amendment must never count the failed attempt as if it did not occur. It authorizes at most one additional attempt because no model scores were generated, rather than asserting the experiment had only one attempted launch.

## C9 validity, new identities and approvals

**Scientific C9 remains valid and unchanged as the parent decision.** Its old receipt does NOT approve changed guard sources or the successor execution contract. If the proposed source/operational changes are selected, a refreshed lead C9 approval of the concrete versioned execution amendment is required under the existing context-bound governance. This is approval of changed operational identities, not a request to retune or revisit the scientific policy. Do not overwrite the original frozen policy/run/anchor or silently inherit their old approval for changed sources.

**A new execution package and fresh frozen C10 context are required for this proposed recovery implementation.** The new package needs its own manifest and archive hashes. Do not assign them before implementation, tests and approved versioned freezing. The original ZIP and C10 context stay unchanged as incident evidence and are not reused as permission to launch.

The new lead C10 approval must expressly authorize ONE additional incident-linked full-cohort/four-mode attempt and bind the actual successor frozen context, package and manifest hashes. Its receipt must preserve the true acknowledgement source and timestamp and bind the parent incident identity, hashes and recovery allowance. Exact recovery receipt field/schema changes must first be implemented and reviewed; there is no currently accepted magic recovery phrase. This document creates no approval receipt or new governance gate.

Approval sequence recommended for lead review:

1. Explicitly authorize the narrowly scoped operational recovery implementation described here, with no inference or remote cache access. This is the minimum next decision; it is not C9/C10 approval of artifacts that do not yet exist.
2. After implementation and affected tests, review the actual versioned operational amendment and its digest using the existing C9 approval process, preserving the parent scientific policy and all prior artifacts.
3. After that approved successor freeze/package and user-managed metadata prepare, issue a NEW C10 approval bound to the actual recovery context/package hashes and incident-specific one-additional-attempt scope. Do not recycle the original receipt.

If the lead chooses a different governance mechanism, define and approve it explicitly before implementation. No current source path grants an exception merely because cache files are present again.

## Manual steps only after those separate approvals

User preserves the original Colab project/lock/ledger, stages the approved versioned recovery package without replacing the parent, and performs the approved metadata prepare. After the NEW C10 authorization, the recovery guard verifies the parent incident and claims its own exclusive attempt BEFORE any recovery test-row/feature reads, then validates the entire original cohort and all 2,000 actual feature/metadata pairs before model calls. Record acquisition hashes and confirm stability throughout scoring/audit. Run all four modes, complete saved-output audits and return one archive containing the incident/recovery linkage and histories. No autonomous Colab/Drive access by the agent.

## Stop, rollback and no-retry rules

Before a new claim, failures leave the original release usable for inspection only; do not delete any staged failure evidence. After a recovery claim, any failure remains permanently recorded and stops without an automatic third attempt. Rollback means stop and retain evidence, never restore a pre-attempt ledger/lock or present the old receipt as unused. Keep no post-test tuning, no feature reconstruction and no C10R success acceptance until genuine complete outputs pass audit.

## Minimum next lead action

Review and explicitly authorize (or reject) the narrow recovery implementation scope above. That authorizes local operational code and synthetic tests only; it does not permit inference, another attempt or modification of the original freeze/ledger. No new cache audit, credentials or experiment is requested at this boundary.
