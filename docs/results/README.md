# Saved-result presentation tables

These CSVs are presentation extracts from original aggregate evaluation/benchmark outputs, not newly evaluated predictions. Values retain source floating-point precision. Each row names its original artifact SHA-256. No labels, media paths, features or per-clip detector scores are published here. The timing-sample table contains the 600 existing benchmark observations, not new timings.

* `test_overall_metrics.csv`: all five overall metrics and confidence intervals for four modes.
* `test_manipulation_metrics.csv`: original AUROC/AP/recall blocks for each manipulation and mode.
* `test_paired_comparisons.csv`: all 80 saved paired results, with explicit subset/metric/a/b/sign.
* `validation_overall_metrics.csv`: corresponding saved validation values, original member hash `a5a268f0fb827c9dbd53a01edc6413816d576f90ac769d134256fcf61f7a8013`.
* `cpu_model_runtime.csv`, `cpu_model_timing_samples.csv`: accepted three-workload/four-mode CPU measurements and their 600 samples.

The CSVs do not substitute for original archives. [Chapter 10](../10_EVIDENCE_AND_REFERENCES.md) maps their source hashes, owners and limitations. The current cleanup verifies hashes and existing arithmetic only; it does not reopen test evaluation.
