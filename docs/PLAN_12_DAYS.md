# 12-day implementation and test plan (+2 documentation days)

| Day | Member 1 | Member 2 | Member 3 | Gate |
|---|---|---|---|---|
| 1 | select/verify checkpoint, license, provenance | dataset labels and license | device/backend + schemas | checkpoint and dataset locked or explicit fallback |
| 2 | both-modal ablations, locate QKV and dimensions | identity/source-disjoint split | mock orchestration | fusion boundary confirmed |
| 3 | attention memory estimate, tiling go/no-go | decode + cache | stage profile | written IO-aware decision |
| 4 | dense oracle, SDPA parity | finish cache | operator microbench | numeric tests pass |
| 5 | simulate symmetric INT8 at fusion only | reference metrics | model-level FP32 benchmark | FP32 predictions integrated |
| 6 | produce lowp predictions | category-stratified metrics | benchmark lowp | H2 preliminary evidence |
| 7 | restore V/output, preserving FP values | paired validation harness | add conversions to timing | first restoration prototype |
| 8 | query control | paired validation comparison | benchmark both | H3 validation evidence |
| 9 | choose/freeze policy | lock validation threshold/protocol | freeze configurations | sealed test remains untouched |
| 10 | four final test config runs | stratified metrics + cluster bootstrap CIs | final benchmark | final numbers |
| 11 | numerical correctness review | leakage and shortcut re-audit | repeatability/hashes | reproducibility gate |
| 12 | research claims signoff | final figures | CLI and archive | code freeze |
| 13–14 | collaborate | collaborate | collaborate | research report/documentation |

**If gate fails:** reduce dataset size during development but do not present underpowered test results as conclusive; if no usable joint model is found, state explicitly that frozen-encoder + small trained fusion-head evidence applies to a different model. Do not silently skip tests or reuse test labels for tuning.
