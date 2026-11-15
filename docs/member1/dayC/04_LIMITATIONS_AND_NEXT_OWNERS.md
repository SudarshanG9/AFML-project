# Limitations and remaining ownership

Status: validation findings available; held-out execution and final owner handoff pending. This record does not assert completion of Day 10 or Days 11–12.

The detector is the frozen pretrained DiMoDif model. Its joint AV self-attention is not a new cross-attention architecture. Quantization is simulated symmetric per-tensor INT8 with floating-point quantize/dequantize operations; no integer kernel, energy saving or guaranteed hardware speedup was established.

Main restoration is cumulative across all five layers, so deeper activations differ. The separate matched-upstream one-clip diagnostic supports local intervention inspection, not population causal attribution. Source-video disjointness does not prove speaker-identity disjointness. Member2’s supplied source audit reports zero validation/test source overlap against the complete LAV-DF training split. Speaker-level overlap remains unverified. Its acknowledged validation silence control found no detector drop for the three earlier configurations; no query-specific control is established, and duration remains uncontrolled.

The approved 1,200-clip validation study does not support positive H2 harm or the motivated H3 restoration claim; the direct query/value comparison is inconclusive. Confidence intervals containing zero do not establish equivalence. The recovered-damage fraction is undefined when the measured damage denominator is nonpositive. Recovery per positive additional millisecond is undefined for the observed nonpositive value-restoration overhead; do not retime to obtain a favorable denominator.

CPU cached-feature timing includes the measured numerical conversion overhead. CPU peak memory was not measured. Raw preprocessing and face-landmark costs are separate and must not be presented as part of an end-to-end latency result. These measurements do not establish GPU or integer-kernel behavior.

Earlier test-label-derived shortcut summary statistics were inspected. Report that history transparently; do not claim pristine zero test-derived exposure. Precision policy and threshold decisions must remain validation-only. No test-model inference has been performed in this Day C continuation. A negative scientific result must not trigger test tuning or an automatic second attempt.

Before freeze, Member 2 must release the final fixed threshold, evaluation/bootstrap procedure, sealed ready-test identity and prior-summary disclosure. Member 3 must release benchmark artifacts, CPU runtime and measurement scopes. Their unsigned local review drafts are not releases.

After the one authorized complete held-out run, Member 2 must perform the originally planned labeled evaluation with the frozen validation-selected threshold and uncertainty procedure. Member 3 must complete the originally planned benchmark/cost reporting using the frozen measurement methodology. Member 1 supplies the four immutable score outputs, full provenance/attempt ledger and limitations. Original Days 11–12 analysis, reporting and additional ownership deliverables remain downstream work, with no implied authorization here to tune after test access or modify another owner's code. Resolve any extra requested experiments through their applicable research review.


The original Member2 final release is now verified and integrated. Statements above requiring that release describe earlier pending work; Member3 final release remains outstanding. The canonical test split replaces the earlier separate-ready-file assumption. See `C10_CANONICAL_EXECUTION_PLAN.md` for the proposed cache resolution and the distinction between C10-time acquisition identities and unavailable original per-test-file hashes.
