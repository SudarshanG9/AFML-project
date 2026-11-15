# AFML Member 3 — final owner-acknowledged release handoff

**Release:** M3-REL-20261113-01  
**Owner:** Mirunjai, authorised Member 3 owner  
**Owner acknowledgement recorded:** 2026-11-13T23:51:00+05:30 (Asia/Kolkata)  
**Member 1 owner validator:** PASS  
**C9/C10 approval:** NOT GRANTED

## Contents

- `m3_release.json` — authoritative owner-acknowledged Member 3 release.
- `acknowledgement_provenance.json` — exact written owner statement and actual timezone-aware recording timestamp.
- `verification/member1_validator_execution.json` — actual execution of the existing Member 1 `validate_owner_release()` function against this release.
- `packages/AFML_MEMBER3_PRIVATE_HANDOFF_20261113.zip` — original evidence support archive, preserved byte-for-byte.
- `packages/AFML_MEMBER3_FINAL_RELEASE_CANDIDATE_REVIEW_SANITIZED.zip` — earlier review candidate package with its stale nested checksum inventory explicitly renamed and marked historical; a fresh internal inventory is provided.
- `review/AI_TECHNICAL_REVIEW_DECISION.md` — prior independent AI technical review, retained as separate provenance.
- `provenance/source_archive_hashes.json` — hashes linking source archives to this final package.

## Validation

The Member 1 validator was executed after recording the owner acknowledgement. The expected runtime was sourced from the approved B2 execution context, approved workloads from the approved workload file, and the three benchmark hashes were recomputed directly from the raw benchmark JSONL files in the evidence archive.

Checkpoint SHA-256: `e1e59006d1bb6ad36a3859af5486a6649ed41bba23258b7c41e24f873be7948e`  
Layer-0 QKV SHA-256: `dc126a957281617a771bef29d1baf4e5b798b6cc54f79f3e808ac24c06507a4f`

## Evidence and scope

The scientific results, approved workloads, runtime, checkpoint identity, benchmark hashes, operator evidence, and preprocessing qualifications were preserved. No benchmark or experiment was rerun. No new three-anchor reproduction is claimed or required for this evidence acceptance.

The exact stale historical inventory is retained only under the name `SHA256SUMS.historical-stale.txt`, accompanied by a notice. It is not used as the current archive inventory.

## Authority boundary

This is the actual written acknowledgement of the Member 3 owner, Mirunjai. It is not a digital-signature claim. It does not grant C9/C10 approval or held-out test access.
