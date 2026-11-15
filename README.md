# AVFuse-CausalQ

Frontend integration now exists on `integration/afml-frontend`: [application guide](frontend/README.md) and [Phase 4 verification report](docs/integration/PHASE_4_FRONTEND_REPORT.md). It adapts the latest frontend to the accepted backend. Final release approval and merging into `main` remain pending.


**A controlled study of numerical precision inside a frozen audiovisual deepfake detector.**

**Integration scope:** this branch completes backend Phases 1–3. Frontend integration and full final-release validation remain deferred. See the [integration report](docs/integration/PHASES_1_TO_3_REPORT.md) and [documentation authority index](docs/README.md).

Can reducing precision in audiovisual fusion change forensic decisions, and can retaining precision in the value/output path repair more damage than retaining query precision at comparable measured cost? AVFuse-CausalQ investigates that question using pretrained **DiMoDif**, whole **LAV-DF** videos, four frozen precision configurations, source-grouped statistics and CPU cached-feature benchmarks.

The motivation is practical: efficient inference is useful only if the detector retains the evidence needed to recognize audio-only, video-only and joint manipulations. The contribution is a documented precision-sensitivity experiment with authentic saved results and provenance. It is not a newly trained detector, a new attention/quantization algorithm or a demonstrated deployment speedup.

## What was built

Offline auto-AVSR encoders produce cached audio/video features. The frozen DiMoDif adapter aligns and pads them, instruments all five joint self-attention layers and preserves the original detection head. The runtime writes predictions without labels; the evaluator later compares them with ground truth.

| Mode | Activation boundaries rounded to simulated symmetric INT8 |
| --- | --- |
| `reference` | none |
| `low_precision` | Q, K, V and attention output |
| `restore_value_output` | Q and K |
| `restore_query` | K, V and attention output |

All modes execute in floating point. Weights, softmax, projections and other model paths retain FP32. This is fake-quant numerical sensitivity, not native INT8 storage or kernels. Restoration is cumulative across layers and V/output is bundled.

| Owner package | Responsibility |
| --- | --- |
| [`member1_fusion`](src/avfuse/member1_fusion/) | pretrained adapter, attention, four interventions and frozen execution safeguards |
| [`member2_data`](src/avfuse/member2_data/) | data/feature preparation, prediction contracts, source-grouped metrics and paired comparisons |
| [`member3_runtime`](src/avfuse/member3_runtime/) | CLI, cached-feature orchestration, benchmark infrastructure and artifact integrity |

## Authentic final results

The canonical cohort contains 1,200 validation and 2,000 test videos. Test has 500 authentic and 500 of each fake category. One reference-validation threshold, **0.6627789735794068**, applies to every mode.

| Test mode | AUROC | Average precision | BCE | Recall | Real FPR |
| --- | --- | --- | --- | --- | --- |
| reference | 0.998044000 | 0.999352103 | 0.067792265 | 0.992 | 0.042 |
| low precision | 0.998033333 | 0.999348271 | 0.068121211 | 0.992 | 0.042 |
| V/output restored | 0.998037333 | 0.999349608 | 0.068176098 | 0.992 | 0.042 |
| Q restored | 0.998040000 | 0.999350411 | 0.067693347 | 0.992 | 0.042 |

The paired BCE damage estimate (low−reference) is +0.000328946, 95% CI [−0.000237371,+0.000962465]. V/output−Q BCE is +0.000482751 [−0.000164867,+0.001185028]. Neither establishes the proposed overall degradation/restoration advantage. Equal displayed rates do not prove statistical equivalence or general precision robustness. [Full intervals, per-manipulation results and original figures](docs/06_EXPERIMENTS_AND_RESULTS.md) explain the evidence.

Accepted typical CPU model p50 latencies are 118.454/161.434/119.918/167.474 ms in the table's mode order. These include applicable fake-quant conversions on resident cached features. They exclude raw-media preparation and do not demonstrate native INT8 acceleration, reliable equal restoration costs or measured peak-memory savings.

## Start here

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m avfuse.member3_runtime.cli --help
python -m avfuse.member3_runtime.cli --smoke
```

The smoke command is a synthetic wiring check. Real validation inference requires a separately supplied DiMoDif checkout/checkpoint, validation manifest and compatible feature cache. The runtime consumes **cached encoder features**, not arbitrary raw video. See the [step-by-step user guide](docs/09_REPRODUCIBILITY_AND_USER_GUIDE.md) for asset hashes, supported commands and validation-only demonstrations.

## Read the research

| Chapter | Purpose |
| --- | --- |
| [1. Foundations](docs/01_RESEARCH_FOUNDATIONS.md) | original question, concepts and H1/H2/H3 |
| [2. Literature and novelty](docs/02_LITERATURE_REVIEW_AND_NOVELTY.md) | technical prior art and provisional contribution |
| [3. Decisions and pivots](docs/03_RESEARCH_DECISIONS_AND_PIVOTS.md) | AVFF/FakeAVCeleb intention through DiMoDif/LAV-DF and C10R recovery |
| [4. Architecture](docs/04_FINAL_SYSTEM_ARCHITECTURE.md) | tensor flow, equations, interfaces and frontend boundary |
| [5. Methodology](docs/05_DATASET_AND_EXPERIMENTAL_METHODOLOGY.md) | cohorts, extraction, freeze, bootstrap and cost protocol |
| [6. Experiments and results](docs/06_EXPERIMENTS_AND_RESULTS.md) | authentic numerical evidence and interpretation |
| [7. Failures and hypotheses](docs/07_FAILURES_LIMITATIONS_AND_HYPOTHESES.md) | phase-specific limits and qualified decisions |
| [8. Conclusions](docs/08_FINAL_CONCLUSIONS_AND_FUTURE_WORK.md) | direct answer and future experiments |
| [9. Reproducibility](docs/09_REPRODUCIBILITY_AND_USER_GUIDE.md) | install, assets, supported CLI and common errors |
| [10. Evidence and references](docs/10_EVIDENCE_AND_REFERENCES.md) | original artifact hashes, verification scope and bibliography |

[Result CSVs](docs/results/README.md) retain unrounded saved aggregates. Historical `docs/dayB`, `docs/member1/dayC`, release configs and recovery scripts remain under original identifiers because accepted evidence refers to them. Their pending-status paragraphs describe earlier stages; the numbered chapters provide the reconciled final account.

The evidence is limited to one checkpoint/dataset, simulated precision, source rather than verified speaker separation, cumulative interventions and a partial silence control. Validation checkpoint-selection exposure, historical test-summary exposure, execution interruptions and the Member 3 acknowledgement inconsistency are disclosed. Private research references, licensed media, complete caches and sealed labels are not newly published. The frontend is untouched. [Cleanup report](FINAL_CLEANUP_AND_REVIEW_REPORT.md) and [rename audit](RENAMING_AND_COMPATIBILITY_REPORT.md) describe the five review branches; the project lead performs every final merge manually.
