# Documentation authority and integrated application scope

Frontend integration now exists on `integration/afml-frontend`: [application guide](../frontend/README.md) and [Phase 4 verification report](integration/PHASE_4_FRONTEND_REPORT.md). It adapts the latest frontend to the accepted backend. Final release approval and merging into `main` remain pending.


Phases 1–3 (backend cleanup/consolidation) and Phase 4 (frontend integration) are completed. Phase 5 is the current final quality gate, recorded in the [release-readiness report](integration/PHASE_5_RELEASE_READINESS_REPORT.md). Merge into `main` remains pending project-lead authorization. Earlier phase reports retain their original scope and deferred-work statements as historical evidence.

## Current project explanation

Start with the [root README](../README.md), then read the numbered research chapters:

1. [Research foundations](01_RESEARCH_FOUNDATIONS.md)
2. [Literature review and novelty](02_LITERATURE_REVIEW_AND_NOVELTY.md)
3. [Research decisions and pivots](03_RESEARCH_DECISIONS_AND_PIVOTS.md)
4. [Final system architecture](04_FINAL_SYSTEM_ARCHITECTURE.md)
5. [Dataset and methodology](05_DATASET_AND_EXPERIMENTAL_METHODOLOGY.md)
6. [Experiments and results](06_EXPERIMENTS_AND_RESULTS.md)
7. [Failures, limitations and hypotheses](07_FAILURES_LIMITATIONS_AND_HYPOTHESES.md)
8. [Conclusions and future work](08_FINAL_CONCLUSIONS_AND_FUTURE_WORK.md)
9. [Reproducibility and user guide](09_REPRODUCIBILITY_AND_USER_GUIDE.md)
10. [Evidence and references](10_EVIDENCE_AND_REFERENCES.md)

The current component guides explain [Member 1](../src/avfuse/member1_fusion/README.md), [Member 2](../src/avfuse/member2_data/README.md) and [Member 3](../src/avfuse/member3_runtime/README.md). [Saved result tables](results/README.md) retain original aggregate values and their source hashes. The [Phases 1–3 report](integration/PHASES_1_TO_3_REPORT.md) records backend consolidation; the [Phase 4 report](integration/PHASE_4_FRONTEND_REPORT.md) records frontend integration; the [Phase 5 report](integration/PHASE_5_RELEASE_READINESS_REPORT.md) is the current pre-merge authority.

## Historical records and supersession

| Retained source | How to read it |
| --- | --- |
| [ARCHITECTURE.md](ARCHITECTURE.md), [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) | original scaffold/protocol documents; incomplete-status paragraphs describe earlier implementation stages, superseded for current status by Chapters 4–6 and the current component guides |
| [PLAN_12_DAYS.md](PLAN_12_DAYS.md) | original schedule and intended gates, not proof that every proposed feature was completed |
| [Day B records](dayB/README.md), [Member 1 Day C records](member1/dayC/00_BASELINE_INVENTORY.md) | dated development, acceptance and recovery narratives; use Chapter 3 to reconcile earlier pending states with later acceptances |
| [Member 2 original final report](member2/results/TEST_RESULTS.md) and original figures | authentic retained results; Chapter 6 and the source-bound CSVs provide the qualified final explanation |
| Frozen `configs/dayB/`, `configs/dayC/`, release records and original handoff locators | immutable identities and scientific/access controls; do not rewrite them to describe this integration |
| [Shared foundation record](cleanup/INTEGRATED_BASE.md) | explains how the original single-parent foundation was composed; its statement that no Git merge occurred refers to that earlier construction, not the reviewed lineage merge performed for this branch |
| [Earlier cleanup report](../FINAL_CLEANUP_AND_REVIEW_REPORT.md), naming report, [cleanup validation](cleanup/FINAL_VALIDATION.json) | records the completed cleanup workstreams and their earlier checks; its suggested future merge procedure is superseded for this integration by the current Phases 1–3 report |
| Earlier Member 3 audit/segment notes | historically accurate implementation stages; reference-only support descriptions precede the current four-mode adapter/runtime integration |

Historical text and hashes are preserved, including qualifications and contradictions resolved by later records. The integration does not rewrite an earlier note to make a later capability appear to have existed sooner. Current integration test counts are separate from the inherited cleanup validation counts.

The runtime requires cached encoder features and the separately supplied pretrained DiMoDif dependency. This branch provides no arbitrary raw-media upload/inference service. Licensed media, checkpoint binaries, complete caches, sealed labels and private prediction archives remain external. The known upstream gitlink without a `.gitmodules` mapping is retained; Chapter 9 provides the explicit upstream revision and acquisition instructions.
