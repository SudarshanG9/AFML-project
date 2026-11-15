# 1. Research foundations

## The question the project set out to answer

The starting point is the lead's **AVFuse-CausalQ — 12-Day Execution Plan**, supplied as `AFML Final.pdf`. Its page 1 asks:

> Under a source-disjoint, manipulation-stratified evaluation, does reducing numerical precision inside AV cross-modal fusion measurably change forensic detection decisions (by attack type: audio-fake / video-fake / both-fake), and can restoring precision in one specific, well-motivated part of the fusion computation recover more of that loss than an equal-cost, naively-chosen restoration — at a cost we actually measure on real hardware?

This is a question about the consequences of changing a detector's arithmetic. It does not ask the team to invent audiovisual detection or to build a better attention kernel. It asks whether a particular engineering simplification changes the detector's correctness, and whether spending precision in a chosen location repairs that change. The phrase “equal-cost” is a condition to demonstrate, not an assumption that follows from naming two configurations “restored.” The original PDF itself makes that qualification in its architecture audit. Its hypotheses are restated below as research expectations; their outcomes belong in [Chapter 6](06_EXPERIMENTS_AND_RESULTS.md) and [Chapter 7](07_FAILURES_LIMITATIONS_AND_HYPOTHESES.md).

The original PDF is a private reference input, not redistributed in this repository. Its hash and inspected pages are recorded in [Chapter 10](10_EVIDENCE_AND_REFERENCES.md). Ordinary hyphens replace extraction ligatures in the quotation. The repository's [12-day plan](PLAN_12_DAYS.md) provides a shorter historical implementation schedule.

## What an audiovisual detector reasons about

A video can contain an authentic face with generated speech, a manipulated face with authentic speech, or manipulations in both streams. A purely visual detector may miss the first case because the pixels are genuine. A purely audio detector has the corresponding problem when only the image is manipulated. A joint model combines representations from both streams, potentially using evidence within each stream and their relationship.

A representation is a sequence of numerical features produced by an encoder. Instead of carrying every pixel and audio sample into a transformer, a visual encoder can express mouth movements as feature vectors, while an audio encoder expresses speech as another sequence. These vectors are useful only with their original preprocessing and temporal conventions. A file containing arbitrary image embeddings is not interchangeable with the speech-related features expected by a specific pretrained detector.

Joint reasoning also does not mean that every fake must be out of synchronization. A lip-synchronization generator is explicitly designed to make mouth motion match speech. Conversely, authentic media can contain transmission lag. The literature therefore motivates content-aware correspondence and manipulation-specific evaluation rather than treating low synchrony as the definition of forgery. [Cai et al.](https://arxiv.org/abs/2404.19171) give a relevant correlation-learning approach; its relation to this study is examined in [Chapter 2](02_LITERATURE_REVIEW_AND_NOVELTY.md).

## Attention and numerical precision

Transformer attention lets each token combine information from other tokens. A query describes what the current token seeks, a key describes what another token offers for matching, and a value contains the information transferred when the match receives weight. In joint audiovisual self-attention these tokens can come from either modality, subject to the pretrained model's mask. This is a broader computation than one separately implemented audio-to-video attention direction.

The matching scores pass through softmax, producing weights that mix values. Small numerical perturbations may change the matches, the transferred content, or both. Later residual connections, normalizations and classification layers can suppress or amplify those changes. Attention is therefore a useful intervention boundary, but an attention-map difference alone does not demonstrate worse forensic performance. The measured outcome must eventually be checked against labels.

FP32 stores numbers with much finer resolution than an 8-bit quantized grid. Quantization maps many nearby real values to the same representable level. With symmetric dynamic scaling, the largest absolute activation sets the scale and positive and negative levels are balanced. This is different from codec compression of the input video: one changes internal arithmetic values, while the other changes the evidence entering the detector.

Fake quantization rounds a floating tensor to such a grid and then represents the rounded values in floating point again. It can test the effect of limited numerical resolution without providing integer storage or integer matrix multiplication. Native INT8 deployment additionally requires appropriate kernels, accumulation rules, scales and hardware support. The final implementation's exact formula and boundaries appear in [Chapter 4](04_FINAL_SYSTEM_ARCHITECTURE.md).

A forensic decision can be sensitive even when overall accuracy barely changes. Scores may move while remaining on the same side of a threshold; rankings can change among already correctly classified clips; an error can become more confident. AUROC measures ranking, recall measures detected fakes at a fixed threshold, and binary cross-entropy measures probability quality against labels. Looking at all of them prevents one rounded accuracy number from hiding changes relevant to the question.

## Selective restoration and its control

The proposal expected V and the attention output to be important because they carry the content passed to the next computation. If rounding damages that content, retaining higher precision at these boundaries might reduce label-grounded loss. Restoring Q is a comparison intervention: it improves the representation used to choose attention matches while leaving values and output rounded.

The comparison has two caveats even before data are observed. V plus output restores two boundaries, whereas Q restores one; tensor operations and conversion costs need not match. In addition, intervening in every layer means each mode develops its own upstream hidden states. Restoring V in layer five keeps the V computed from that mode's current input, not the V that an independent FP32 trajectory would have produced. These experiments compare well-defined configurations, but do not isolate the independent causal importance of V, output, Q or any single layer.

## IO-aware computation and the three hypotheses

IO means movement of data between memory levels. A dense attention implementation materializes a matrix whose storage grows with the product of query and key lengths. IO-aware methods reduce expensive memory traffic through tiling and fusion. [FlashAttention](https://arxiv.org/abs/2205.14135) established this approach; choosing an existing operator requires no claim of having invented it. Whether a new custom kernel is justified depends on actual shapes, hardware and bottlenecks, rather than the word “transformer.”

The proposal's page-18 registry defines three expectations:

| Hypothesis | Expectation and rationale | Test and evidence that would weaken it |
| --- | --- | --- |
| H1 | At real token counts, dense fusion-score storage will be too small to justify a custom tiled kernel. Existing optimized attention should make a separate kernel an unnecessary implementation burden. | Inspect actual padded shapes and masks, estimate dense storage and inspect operator/model cost. A demonstrated memory bottleneck requiring another implementation would weaken the engineering expectation. A size estimate alone cannot prove a runtime bottleneck absent. |
| H2 | Simulated low-precision fusion will measurably degrade detection, with effects differing across manipulation types. Quantization can disturb subtle forensic cues and cross-modal relationships. | Compare matched reference/low-precision scores against labels, overall and by manipulation. Directional paired uncertainty supporting worse low-precision loss would support degradation. A category-specific claim additionally needs evidence of differing effects, not merely different baseline difficulty. |
| H3 | Restoring V/output will recover more validation loss than restoring Q at equal cost. The information-carrying path was the motivated target. | Establish damage first, measure each restoration's label-grounded loss reduction, compare those reductions with paired intervals, and verify comparable reliable costs. Similar or better Q outcomes weaken the preference; unequal costs leave the equal-budget part unresolved. |

For BCE loss, write damage as \(D=L_{low}-L_{ref}\), restoration recovery as \(R_m=L_{low}-L_m\), and the proposed preference as \(R_V-R_Q=L_Q-L_V\). Positive values respectively mean damage, recovery, and a V/output advantage. These signs matter: closeness to FP32 outputs is not automatically recovery, because the reference itself can make mistakes.

The proposal explicitly permits negative findings as a completed project outcome. A credible investigation must preserve the question even if the hoped-for optimization proves unnecessary. That principle connects the foundations here to the decisions, methods and final evidence in the following chapters.
