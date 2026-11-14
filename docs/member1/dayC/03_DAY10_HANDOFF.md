# Day 10 Member 1 handoff — prepared, not executed

Status: PREPARED_NOT_EXECUTED. C8B validation evidence is accepted. The genuine owner releases and sealed ready-test manifest identity are missing; C9 and C10 have not been supplied. See `B4_B5_EXECUTION_READINESS.md` for the consolidated dependency request and exact current evidence paths.

The implemented runner, release verifier, saved-output auditor and handoff packager have synthetic test coverage. No production frozen policy/run/anchor, test attempt, held-out score file or immutable Member 2/3 delivery exists yet. No test performance result is asserted.

After the authorized single run, append a dated execution record with the real approved runtime, full cohort count, all four mode names, each output's exact path and SHA-256, schema/ID/checkpoint completeness results, frozen policy/code/anchor identities, bound C9/C10 records, one-time lock and ledger, saved-output audit, handoff archive identity and read-only Git status. Preserve this preparation record. In a partial failure, retain the attempt and failure evidence; do not rerun or publish an apparently complete handoff.

Member 2 owns labeled test evaluation and fixed-threshold reporting. Member 3 owns final benchmark and cost reporting. Packaging files locally does not establish delivery or owner acceptance. Request `ACK DAYC C10R: ACCEPT MEMBER1 DAY10 HANDOFF` only after the actual complete audited handoff exists.


## Requested final payload — preparation only

After successful C10-authorized execution, supply `reference_test.jsonl`, `low_precision_test.jsonl`, `restore_value_output_test.jsonl`, `restore_query_test.jsonl`, and reuse accepted `artifacts/predictions/restore_query_val.jsonl` unchanged (1,200 rows; SHA-256 `855a90bf1a72c59043ac3e80f04ec15df16d10ea0b98056ff10c56d751d5dbf1`). Do not regenerate the validation output.

The lead reports 2,000 unique canonical-test IDs per test file, `split="test"`, probability-of-fake scores, frozen checkpoint SHA-256 and no ground-truth labels. Bind the count and cohort selector only after authentic Member2 release verification. Include schema/cohort validation, SHA-256 inventory, runtime/run provenance and seal/attempt evidence. Saved-output audit and packaging must verify these requirements before an actual handoff is declared complete. No test file has been generated during preparation.


## 2026-11-14T07:29:40.084277+00:00 — completed and C10R accepted

The historical preparation status above is superseded by the actual accepted run `C10_RECOVERY_R2_C10_20261113T191638908596Z`. All four modes completed 2,000 canonical test clips each; the verified returned archive and Member 2/3 handoff are preserved at `artifacts/dayC/member1/additional_recovery/C10_RECOVERY_R2_C10_20261113T191638908596Z/returned_evidence`. C10R acknowledgement: `artifacts/dayC/member1/approvals/C10R_ack_20261114T072940084277Z.json`. Review SHA-256: `4b452b2eadb280604f9daaf606300fef95b78b8f63d1280412cb250e19e97e60`. Handoff SHA-256: `97e603cb3883023ba87cb2fec518e69ce8a5a3438cf79827db72be56bbee4f0c`. Three total attempts, including both preserved failures; no further attempt authorized. CPU frozen runtime and unchanged model-state identity are bound in the accepted archive.

Prediction SHA-256 inventory: `{"low_precision": "bbbfed7fb54f05ae7b2ddb5046c156f0a3fe176d270a9d46eae49a50bc3ca833", "reference": "392ba75b21ca84819cc167776e829658c88dccca0c4e19806def5e4a194f20a0", "restore_query": "3744f5573a5e5896dceda5ac6ba642cec3684352c79b78befb938ae232ed28fc", "restore_value_output": "bc156b3cf058a8818364cad7ea942e25082f306aeecdf768e179d8342ba1dbfa"}`. The accepted query-validation predictions are unchanged. Remote acquisition hashes are preserved; historical per-file test-cache hashes were unavailable and remote cache bytes were not independently rehashed locally. Member 2 labeled evaluation and Member 3 final reporting remain owner tasks; this receipt does not establish delivery or owner acceptance. No test metrics have been computed by Member 1.
