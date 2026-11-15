# 7. Failures, limitations and hypothesis decisions

## Why a completed study can contain failed expectations

The plan's desired result was selective restoration that repaired precision-induced forensic damage at a favorable measured cost. Completion means implementing and honestly testing that question, not obtaining its preferred answer. An unavailable checkpoint, interrupted execution and statistically unsupported hypothesis are different kinds of difficulty: respectively a feasibility constraint, an operational failure and a scientific result. This chapter keeps them separate.

## Model-selection and feasibility phase

AVFF was the initial architectural preference, but the supplied proposal did not verify an official runnable checkpoint. A small newly trained fusion head was a fallback; using it would have changed the scientific target and introduced training-data effects. DiMoDif instead supplied a pretrained joint model and inspectable implementation. That resolved the immediate executable-model problem, while narrowing the result to a different detector. There is no evidence that the team reproduced AVFF or exhaustively ruled out every available AV checkpoint.

The original custom-kernel expectation also met a larger real padded shape than the proposal illustrated. The 22.046 MiB score estimate and observed masks warranted careful correctness work, but did not demonstrate that a new tiled implementation was needed. Choosing dense/SDPA avoided an additional unverified kernel. It left a limitation: the project did not empirically prove an IO bottleneck absent or quantify measured peak-memory savings. H1 is therefore primarily an engineering-scope decision supported by the inspection, not a universal systems result.

## Data-preparation and validity phase

An initial short-window pipeline could exclude a localized fake edit, leaving a misleading whole-video label. Moving to whole videos resolved that particular window/label mismatch for the selected DiMoDif inputs. The earlier 9,143-window and 1,992-test counts remain historical and must not be blended into the final 3,200-video/2,000-test cohort.

Source grouping resolved a narrower leakage concern than the plan's optimistic speaker/identity language suggested. The metadata does not verify speakers; setting `identity_id = source_id` does not create identity information. Zero selected-source overlap with LAV-DF training and zero validation/test crossing groups reduce recording leakage, but do not rule out a shared speaker across recordings or every upstream pretraining exposure. The checkpoint's use of the development split for selection additionally limits independent interpretation of validation results.

The full-array audit was interrupted after 2,800 of 3,200 arrays. Directory and metadata coverage continued to support all 3,200, and accepted inference eventually completed, but there is no complete independently verified historical original per-feature byte baseline. File hashes acquired later establish acquisition identity, not proof that every tensor equals its earliest extraction. The large source metadata and media were not fully re-audited during this cleanup. These limitations remain even though they do not invalidate the accepted saved-output arithmetic.

## Precision-intervention phase

The scheme tests activation rounding rather than deployed integer arithmetic. Weights, residual paths, normalization, softmax and computation remain floating point. Dynamic per-tensor scaling may tolerate this checkpoint's ranges better than a static or hardware quantizer. Consequently a small observed effect cannot establish robustness to native INT8, other granularities, weight quantization, reduced accumulators, lower bits or other checkpoints.

All five layers are intervened on cumulatively. Retaining V/output preserves values computed from the current mode's hidden state, which may already differ from reference. V and output are bundled, and Q changes another boundary. The comparison estimates effects of these configuration choices within a fixed system; it cannot independently attribute an effect to one tensor, one layer, or cross-modal semantics alone. Stronger causal isolation would require a separately approved future design with fixed upstream activations and individual boundary/layer ablations.

The anticipated validation damage did not gain statistical support. Without established positive damage, a “percentage recovered” is unstable or meaningless. The team retained the direct loss contrasts and froze an honest reference-preferred interpretation. It did not redefine improvement as closeness to FP32, replace the loss endpoint with a favorable diagnostic, or tune a new precision policy on final test scores.

## Execution and provenance phase

The parent C10 attempt failed on a missing cache input; R1 was interrupted; R2 completed all four test outputs. Failure records, consumed locks and distinct authorizations were retained rather than silently overwritten. That resolves the missing-output problem for the accepted return but leaves a more complex execution history than the planned single uninterrupted pass. “No completed predictions” in failed ledgers does not mean no transient processing occurred. An R3 package revision must not be miscounted as an additional scientific attempt.

Original C9/C10R approvals and release acknowledgements are retained attestations, not digital signatures. The Member 3 release names Mirunjai as the human owner, while a nested validator qualification says GPT-6; the acknowledgement provenance records the inconsistency. It should remain visible without inventing a replacement acknowledgement or presenting an automated actor field as a signed human review.

The final Member 2 package was unavailable to an earlier audit and later directly recovered from the supplied evidence copy. That later verification supersedes the missing-evidence conclusion. All eleven payload checksums match, including final metrics, control, ledger and figures. The exact literal evaluation shell command is still absent from the run log: a README example is not proof of the exact invocation. Neither hashes nor conventional append-only ledgers prove the absence of undocumented access outside the retained system.

Historical test-label summaries were inspected before freeze and explicitly disclosed. Member 1's inference-attempt history is separate from Member 2's two-entry final evaluation/control ledger. The defensible claim is no test-score-based precision-policy selection in the accepted chain, with disclosed earlier test-derived information. A completely untouched-test narrative is unsupported.

## Statistical and benchmark phase

The held-out cohort contains four balanced categories, but only one dataset and checkpoint. High baseline separation and modest source counts leave sensitivity to subtle precision effects limited. A confidence interval crossing zero means the data do not establish the specified direction under this method; it does not prove zero effect. The study specifies no equivalence margin or adequately powered equivalence test.

Eighty nominal paired intervals include correlated metrics and manipulation subsets sharing authentic videos. The one audio-fake BCE interval favoring Q over V/output is exploratory without multiplicity correction. No robust manipulation-by-precision interaction or broad Q superiority follows. Source bootstrap accounts for within-source dependence as encoded, not unidentified speaker dependence or all checkpoint-selection uncertainty.

The accepted benchmark measures resident cached-feature CPU model calls. Fake quantization adds conversion work without integer matrix multiplication, and the recorded costs are unequal/noisy. V/output-minus-low overhead is negative in the typical record, while Q-minus-low is positive. This prevents an honest equal-cost or reliable positive-cost-normalized restoration claim. Timing uncertainty is not included in BCE confidence intervals.

CPU peak memory is unmeasured, accelerator support is partly tested with mocks, and separate offline CPU/T4 stage medians cannot be summed into a measured raw-media latency percentile. The workload names refer to different valid input lengths under fixed padding; they are not an attention-length scaling experiment. These qualifications preserve the value of measured CPU costs without turning them into deployment evidence.

The trailing-silence matching control weakens one shortcut explanation in 912 selected test videos. It does not prove no silence use, does not control duration or every artifact, and does not establish the same result for all precision modes. A precision-stable shortcut could still be present.

## Final hypothesis assessment

| Hypothesis or subclaim | Category | Evidence and remaining boundary |
| --- | --- | --- |
| H1: custom tiling is not justified for this project's scope | Supported as a scope decision | real shapes, theoretical storage, preserved masks, parity and existing SDPA support the no-go; absence of every memory bottleneck is inconclusive |
| H2: this simulated low precision degrades overall detection | Not supported | validation and test paired overall BCE intervals cross zero; other overall metrics do not establish degradation |
| H2: degradation differs reliably by manipulation | Inconclusive | different baseline difficulty and tiny subset changes do not establish an interaction; one unadjusted comparison is insufficient |
| H3: V/output restores more label-grounded quality than Q | Not supported | validation direct contrast and final overall V/output-minus-Q interval cross zero; point estimates do not favor the proposed preference |
| H3: superiority at equal measured cost | Outside experimental scope of the achieved comparison | reliable comparable restoration costs were not established |
| Native INT8 speed/memory improvement | Outside experimental scope | floating simulation, no deployed integer kernel, CPU memory unmeasured |
| Universal precision robustness/equivalence | Outside experimental scope | one checkpoint/cohort and no equivalence design |

The original registry said H2/H3 could be weakened by no meaningful difference or a successful control. Statistical reporting adds the necessary distinction between absence of demonstrated support and proof that a hypothesis is universally false. The present study rejects the proposed positive claim for this evidence; it does not rule out every future restoration benefit.

The negative result is useful because it blocks an unsupported recommendation to spend engineering effort on a restoration policy that the selected detector did not demonstrate a need for. It also supplies an auditable starting point for stronger follow-up work, with the failed expectations, unresolved validity limits and executable boundaries documented together.
