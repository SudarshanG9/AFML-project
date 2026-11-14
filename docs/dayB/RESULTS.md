# Day B validation results

G8/G9 technical audit passed; final human scientific acknowledgement is pending. Three configurations were evaluated on the same 1,200 approved validation clips, exactly 300 each of real/audio_fake/video_fake/both_fake. The fourth query-control configuration is defined but its Day 8 full-validation experiment remains outstanding.

| Mode | AUROC | AUPRC | BCE | Fake detections / 900 | Real false positives / 300 |
|---|---:|---:|---:|---:|---:|
| reference | 0.998370370 | 0.999459452 | 0.060848822 | 894 | 15 |
| low_precision | 0.998381481 | 0.999463378 | 0.060805678 | 894 | 16 |
| restore_value_output | 0.998374074 | 0.999460798 | 0.060849594 | 894 | 15 |

Paired BCE damage (lowp minus reference): -4.3144353457e-05; 95% CI [-0.000624812977708, +0.000464924289069].

Paired BCE recovery (lowp minus restored): -4.39166091455e-05; 95% CI [-0.000476094793791, +0.00042285118253].

Neither interval excludes zero. No positive quantization damage or V/output restoration rescue is established. This does not prove equivalence or improvement. Share recovered is undefined because estimated damage is nonpositive. All bootstrap analyses use 1,000 source-group resamples, seed 0.

Fixed provisional validation threshold: 0.6627789735794068. This is not the Day 9 policy freeze.

The matched typical CPU benchmark observed lowp p50 118.984093 ms and restored p50 168.305224 ms, an added 49.321131 ms in this run. Timings vary by workload/session; no stable integer-kernel speedup is established. These are floating-point fake-INT8 runs. CPU memory was not measured. Model-on-cached-features and isolated-attention timing scopes exclude raw-media extraction.

The main restoration is cumulative across five joint AV self-attention layers; deeper inputs can change. A separate one-clip diagnostic verified matched lowp upstream and quantized QK at each target when only that layer was restored. This does not establish population forensic accuracy.

Open limitations: validation/dev checkpoint-selection bias; source groups are not verified speaker groups; training overlap not independently verified; trailing-silence and other shortcut risks outstanding. High absolute validation scores do not resolve these limitations.

**Day 8 equal-cost control comparison not yet evaluated.** Query-control validation, final policy freeze and all four held-out test runs remain later work. No held-out feature access, Git commits or pushes were performed by the agent.

The audited full predictions, confidence intervals, raw benchmark samples, provenance and Member 2 handoff are retained locally under `artifacts/dayB/` and corresponding prediction/report/benchmark folders. They are intentionally excluded from Git by the repository policy. Reference scientific summary identifiers are in `configs/dayB/`; local full evidence is indexed by `docs/dayB/PROJECT_FILES.md`.
