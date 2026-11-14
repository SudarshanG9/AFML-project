# Day B project files

All agent-generated outputs through G6 package preparation are now copied into this repository. Original experiment outputs, archives and prior audit text are preserved byte-for-byte. Historical reports can contain the old workspace paths; use the project inventory to map those paths to the verified project copies.

- G6 source: [run_g6.py](../../scripts/dayB_g6/run_g6.py), [operator_g6.py](../../scripts/dayB_g6/operator_g6.py)
- G6 tests: [test_dayB_g6_guards.py](../../tests/integration/test_dayB_g6_guards.py)
- Colab instructions: [COLAB_G6_INSTRUCTIONS.md](COLAB_G6_INSTRUCTIONS.md)
- Upload this original archive: [AFML_G6_CPU_Addon.zip](../../artifacts/dayB/local_deliverables/g6_colab/AFML_G6_CPU_Addon.zip)
- Ready reference predictions: [reference_val.jsonl](../../artifacts/predictions/reference_val.jsonl)
- Original reference filename: [dayB_reference_val.jsonl](../../artifacts/predictions/dayB_reference_val.jsonl)
- Reference metrics: [metrics.json](../../artifacts/reports/dayB_reference_val/metrics.json)
- Exact provisional calibration: [dayB_validation_calibration.json](../../artifacts/reports/dayB_validation_calibration.json)
- Reference benchmarks: [artifacts/benchmarks](../../artifacts/benchmarks/)
- All historical audits: [audit](../../artifacts/dayB/local_deliverables/audit/)
- G4/G5/G6 code packages, extracted snapshots, handoffs and PDF extracts: [local_deliverables](../../artifacts/dayB/local_deliverables/)
- Full source-to-project path and SHA-256 inventory: [PROJECT_FILE_INVENTORY.json](../../artifacts/dayB/PROJECT_FILE_INVENTORY.json)

The canonical G6 source files above locate their frozen payload under artifacts/dayB/local_deliverables/g6_colab/AFML_G6_CPU_Addon. The upload archive remains unchanged and has its original internal g6/ layout. Copies of tests inside historical distribution snapshots/ZIPs are retained to preserve package hashes; the repository test file lives under tests/integration.

No G6 research inference has been executed. The existing CPU Colab runtime, approved G5 package and G5 run are still required for user-run G6 execution. Nothing was staged, committed or pushed by this operation. Generated artifacts remain covered by the repository's existing .gitignore.

## G6 received review

- [G6 audit: technical PASS, awaiting acknowledgement](../../artifacts/dayB/g6_colab_review/G6_CPU_20261111T194903Z_6a2bf3c1/G6_AUDIT.md)
- [Original returned evidence, logs and hashes](../../artifacts/dayB/g6_colab_review/G6_CPU_20261111T194903Z_6a2bf3c1/received_evidence/)
- [Member 2 low-precision predictions](../../artifacts/predictions/low_precision_val.jsonl)
- [Review inventory](../../artifacts/dayB/g6_colab_review/G6_CPU_20261111T194903Z_6a2bf3c1/LOCAL_REVIEW_FILE_INVENTORY.json)

No established positive quantization damage; no G7/G8 execution. Preserve the exact G5 threshold and the open research limitations.

## G7 initial preparation — superseded by approved release below

- [Scientific wording and separate matched-upstream design](G7_RESEARCH_REVIEW.md)
- [G7 Colab instructions — pending separate scope](COLAB_G7_INSTRUCTIONS.md)
- [Standard upstream script](../../scripts/dayB_upstream_audit.py)
- [Separate matched-upstream script](../../scripts/dayB_matched_upstream_diagnostic.py)
- [Unit tests](../../tests/member1/test_dayB_matched_upstream_diagnostic.py)
- [Implementation status and hashes](../../artifacts/dayB/g7_colab/G7_IMPLEMENTATION_STATUS.json)

The initial pending-scope archive is preserved. The user subsequently approved the separate diagnostic; use the approved release below. Real-model evidence is still outstanding. No G8 execution.

## G7 approved release

The user approved the typical validation clip and one restored target layer per arm, with every non-target layer including the suffix kept low precision. The current source, approval record and instructions reflect that decision.

- [Approved G7 upload ZIP](../../artifacts/dayB/g7_colab/AFML_G7_CPU_Addon_APPROVED.zip)
- [Exact checksum-verified Colab cell](G7_APPROVED_QUICKSTART.md)
- [Full approved instructions](COLAB_G7_INSTRUCTIONS.md)
- [Approved diagnostic design](G7_RESEARCH_REVIEW.md)
- [Release checksum, size and scope](../../artifacts/dayB/g7_colab/G7_APPROVED_PACKAGE_DELIVERY.json)
- [Verified project files](../../artifacts/dayB/g7_colab/G7_APPROVED_IMPLEMENTATION_STATUS.json)

24 tests pass. Manually return G7_review.zip for audit. Stop at G7; no G8 execution, commits or pushes.

## G7 passed; combined G8/G9 release

The supplied G7 review passed; its conditional user acknowledgement is now recorded. The user explicitly authorized G8 and G9 as one batch with one final verification, superseding the earlier intermediate G8 acknowledgement stop. Final scientific conclusions remain for the single final review.

- [G7 audit](../../artifacts/dayB/g7_colab_review/G7_CPU_20261111T202044Z_ce6a0493/G7_AUDIT.md)
- [One-shot G8/G9 upload archive](../../artifacts/dayB/final_colab/AFML_DayB_G8_G9_CPU.zip)
- [Exact checksum-verified Colab cell](DAYB_FINAL_QUICKSTART.md)
- [Full one-shot execution instructions](COLAB_G8_G9_ONE_SHOT.md)
- [Runner source](../../scripts/dayB_finish/run_g8_g9.py)
- [Guard tests](../../tests/integration/test_dayB_finish_guards.py)
- [Release checksum, contents and test counts](../../artifacts/dayB/final_colab/DAYB_FINAL_PACKAGE_DELIVERY.json)
- [Verified project release inventory](../../artifacts/dayB/final_colab/PROJECT_RELEASE_INVENTORY.json)

516 local tests passed with 3 existing ffmpeg/ffprobe-dependent skips. The explicitly named runbook suite passed all 18 tests. The research batch was subsequently executed and returned; see the final audit below. Day 8 query control, final policy freeze and held-out test remain outside this batch.

## Returned G8/G9 final audit — technical PASS

All three full validation files have exactly 1,200 approved IDs and 300 clips per category. All 66 outer evidence hashes and the nested handoff hashes match. Metrics and all source-cluster CIs independently reproduce. No established positive quantization damage or V/output rescue was found; final human scientific acknowledgement remains pending.

- [Final human-readable audit](../../artifacts/dayB/final_review/DayB_FINAL_CPU_20261112T031627Z_91e5f723/DAYB_FINAL_AUDIT.md)
- [Machine-readable checks and statistics](../../artifacts/dayB/final_review/DayB_FINAL_CPU_20261112T031627Z_91e5f723/DAYB_FINAL_AUDIT.json)
- [Verified Member 2 handoff ZIP](../../artifacts/dayB/handoffs/DayB_FINAL_CPU_20261112T031627Z_91e5f723/Member2_DayB_handoff.zip)
- [Restored validation predictions](../../artifacts/predictions/restore_value_output_val.jsonl)
- [Full three-mode metrics](../../artifacts/reports/dayB_three_mode_val/metrics.json)
- [Paired recovery](../../artifacts/reports/dayB_restoration_paired_val/paired.json)
- [Review file inventory](../../artifacts/dayB/final_review/DayB_FINAL_CPU_20261112T031627Z_91e5f723/LOCAL_FINAL_REVIEW_INVENTORY.json)

The requested G8/G9 batch has run and its technical audit passes. No further Colab run or evidence upload is required for these Day B deliverables. Final G9 scientific acknowledgement is outstanding. Day 8 equal-cost query control, final policy freeze and held-out test remain unevaluated and unauthorized by this final review. No Git commit/push or new model inference was performed by the auditing agent.

## GitHub source organization and Member 2 handoff

- [Current project overview](../../README.md)
- [GitHub file layout and existing-index exclusions](GITHUB_FILES.md)
- [Portable Day B source/reproduction guide](README.md)
- [Versionable results summary](RESULTS.md)
- [Versionable precision definitions](../../configs/dayB/precision_modes.json)
- [Packaging requirements](../../requirements/dayB-colab-cpu.txt)
- [Readiness verification and exact pre-existing tracked-payload list](../../artifacts/dayB/github_preparation/GITHUB_READINESS.json)

Missing G4/G5 helper sources are now in scripts/dayB_colab; frozen package manifests/configs are in configs/dayB; packaging tests are in tests/integration. Both frozen packages rebuilt successfully with exact file contents. Three new packaging tests passed. Source placement is complete; the pre-existing Git index still includes pilot raw/tensor data and OS metadata and needs owner review before upload. No git staging/commit/push was performed. Raw artifacts stay local.
