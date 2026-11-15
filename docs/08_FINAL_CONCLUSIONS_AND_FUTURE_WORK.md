# 8. Final conclusions and future work

## Answer to the original question

For the frozen DiMoDif LAV-DF detector and selected source-grouped cohort, reducing the resolution of Q/K/V and pre-projection attention output through dynamic symmetric fake INT8 produced small score/metric changes. The saved validation and held-out evidence does **not** establish overall forensic degradation or an overall advantage from restoring V/output instead of Q. Reliable equal restoration cost was not demonstrated. The original question's proposed equal-cost recovery advantage therefore receives no positive support from this completed study.

That answer has three distinct parts. First, the numerical intervention was real and defined: selected activations were rounded in all five joint self-attention layers while pretrained parameters and other operations remained floating point. Second, the hoped-for label-grounded damage/recovery pattern was not established. Third, the floating simulation did not implement the hardware acceleration required to turn a precision policy into a deployment gain. These must not be collapsed into a broad assertion that quantization is harmless or that restoration can never work.

## What was hypothesized, implemented and observed

The plan expected that precision reduction could disturb subtle forensic evidence and that the information-carrying V/output path would deserve precision more than Q. It also expected existing optimized attention to remove the need for a custom kernel. The implemented system retained a dense oracle/SDPA path, a strict pretrained adapter, four frozen modes, whole-video feature preparation, source-aware statistics and scoped runtime/provenance tools. It used DiMoDif/LAV-DF instead of the initially preferred AVFF/FakeAVCeleb pair.

The held-out BCE damage estimate is +0.000328945787, 95% CI [−0.000237371472,+0.000962465411]. V/output-minus-Q BCE is +0.000482751028, CI [−0.000164867252,+0.001185027814]. The direct preference is not supported, and neither interval provides an equivalence result. Identical aggregate recall 0.992 and real FPR 0.042 at the fixed threshold coexist with nonidentical ranking/loss values. The manipulation analysis finds video-only fakes descriptively hardest, but does not establish robust manipulation-dependent precision damage.

The accepted typical CPU p50 costs are about 118.454 ms reference, 161.434 ms low precision, 119.918 ms V/output and 167.474 ms Q. These are resident cached-feature model measurements including applicable conversions. They cannot establish native INT8 acceleration, measured memory reduction or a raw-media response time. The reference silence control retains strong ranking after matching one shortcut in a subset, while leaving broader shortcut questions open.

## Contribution and what remains unknown

The contribution is an explicit experimental comparison with authentic saved outputs, source-grouped paired uncertainty, measured but carefully bounded costs, and a retained failure/recovery/approval chain. This makes it possible to inspect where a proposed optimization lacked evidence without mistaking absent support for incomplete implementation. The project also preserves a stable three-member code boundary: model/intervention, data/evaluation and runtime/provenance.

It does not introduce audiovisual detection, cross-modal attention, quantization, mixed precision, selective restoration or IO-aware computation. The narrower experimental contribution is provisional relative to the reviewed literature. The study does not determine whether small effects arise from dynamic scales, redundant features, residual paths, high baseline separability or another mechanism. It does not evaluate native integer accumulation, quantized weights, multiple detectors, speaker-disjoint generalization, cross-dataset transfer, localization or equal-budget adaptive precision.

## Scientifically useful next experiments

**Separate numerical and hardware effects.** A native INT8 implementation should specify weight/activation ranges, accumulation, calibration, granularity and supported operators. Validate parity and measure its own label-grounded error before comparing real device latency, transfers, preprocessing, peak memory and energy where a reliable instrument exists. Floating fake-quant costs should remain a separate baseline.

**Improve causal isolation.** On a newly authorized development cohort, hold upstream activations fixed and vary one boundary or layer at a time. Separate V from output and add Q/K comparisons. Record scales and hidden-state changes. Such a design can determine whether an apparent configuration effect comes from one operation or accumulated changes. It is future work, not a description of the accepted all-layer experiment.

**Make cost fairness an experimental condition.** Repeated/interleaved measurements, matched thread/device settings and uncertainty on timing can test whether two restoration policies are genuinely comparable. If costs remain unequal, define a budgeted selection rule before examining held-out outcomes. Avoid ratios with near-zero, negative or unreliable extra costs.

**Broaden the evidence deliberately.** Other pretrained checkpoints, finer/coarser precision levels and independent datasets could reveal effects hidden by this strong in-dataset baseline. Verify licenses, preprocessing, source/speaker identity and checkpoint training overlap first. A cross-dataset test answers a different and stronger question than repeating the same accepted test cohort.

**Specify statistical targets in advance.** Choose a primary endpoint, a smallest practically important effect, multiplicity policy and sample-size/power plan. If the question is equivalence, specify equivalence margins and conduct the appropriate design. More bootstrap replicates alone do not create more independent sources or resolve checkpoint-selection bias.

**Strengthen validity controls.** Evaluate duration, leading/trailing silence, codec and speaker effects under predefined matched or perturbed conditions. Test every precision mode if making a mode-wide shortcut claim. Preserve generated-media authenticity tests separately from artificial synchronization perturbations; a shifted real clip is not automatically a deepfake.

Any such extension requires a new protocol and new permissible data access. The accepted C9/C10R record remains a finished historical experiment. Its value is the bounded answer it actually provides, including the absence of the hoped-for restoration advantage.
