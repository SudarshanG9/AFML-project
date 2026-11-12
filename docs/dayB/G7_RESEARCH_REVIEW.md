# G7 preparation and scientific scope review

Status: implementation/tests ready; real-model G7 evidence outstanding. **G7 is not PASS or closed.** G6 has been acknowledged. The user has requested a separate diagnostic with matched low-precision upstream at every layer. The user explicitly approved the typical validation clip, one restored target per arm, and all non-target layers including the suffix kept low precision.

## Mathematics retained

The existing symmetric per-tensor fake INT8 quantize/dequantize is unchanged. All execution remains floating point. `restore_value_output` fake-quantizes Q and K while retaining V and bypassing the attention-output fake quantizer. The original FP32 packed QKV projection, output projection, weights, masks, residuals, normalization, feed-forward layers and head remain unchanged.

“Attention output” in the precision trace is the output of the attention operator before the unchanged FP32 output projection. The separate diagnostic additionally measures the target output after that projection and labels that boundary explicitly.

## Main whole-model interpretation

Recommended wording: “The main restoration configuration applies a cumulative V/output precision intervention across all five attention layers while keeping Q/K fake-quantized. Layer 0 has matched upstream input. At deeper layers, earlier restorations can alter the hidden states reaching the layer; the whole-model contrast therefore does not hold every layer's upstream activations constant.”

The standard upstream audit measures this per layer. A later input mismatch is a scientific observation, not an implementation failure. Unexpected layer-0 mismatch is a failure. The pilot trace must show Q/K changed and V/attention_output unchanged at all 20 defined boundaries. One clip cannot establish improved forensic accuracy.

## Separate matched-upstream diagnostic — approved scope

Use the already approved typical validation clip `7cea6ad75b4ebf41`, no labels, the accepted checkpoint and existing CPU runtime. For each target layer 0–4:

1. Baseline: all five layers low precision.
2. Diagnostic arm: restore V/output only at the target layer; keep both prefix and suffix layers low precision.
3. Verify prefix and target inputs are bit-exact, projected Q/K are bit-exact, and fake-quantized Q/K at the target are bit-exact between arms.
4. Verify only the target changes precision mode, and its V and attention output are unquantized.
5. Report target-output difference after the FP32 output projection separately from the downstream final-score change.
6. Verify unchanged state, restored forward methods and removed diagnostic hooks. Check the diagnostic lowp baseline against the original adapter and the accepted G6 score for this clip.

Recommended wording if real evidence passes: “For this fixed model and validation input, each layer-specific comparison has identical low-precision upstream at its target. Its target-output difference isolates the local precision intervention under those inputs. Its final-score difference measures the downstream response to that single-layer intervention with suffix precision modes fixed low precision; suffix activations are allowed to change. These conditional diagnostic effects are separate from the cumulative five-layer experiment and do not establish population-level forensic gains.”

The user explicitly approved this stronger diagnostic scope on 2026-11-12. `scope_decision.json` records APPROVED and the exact user acknowledgement. No stronger diagnostic has yet been run, and no main configuration has been replaced.

## Local validation

24 tests passed: the runbook upstream toy test, three separate matched-upstream tests, and the existing required intervention/precision/mode-math tests. The new tests cover per-target matched inputs/QK, exactly one restored layer, preservation, and cleanup after failures/invalid targets. These are toy unit tests, not real DiMoDif results.

Real five-layer upstream tables, full-model trace and matched-target effects remain outstanding. No numerical table or causal result is fabricated. G6 found no established positive quantization damage; G7 cannot assume or invent a rescue.

## Preserved research limitations

Checkpoint-selection bias on dev/validation; source groups without verified speaker identities; training overlap not independently verified; trailing-silence and other shortcut risks. Simulated INT8 is not an integer kernel. No held-out features may be opened.

## Execution boundary

The standard audit, pilot trace and separately approved matched-upstream subphase may now run within G7 on the existing CPU Colab runtime. G8 full-validation restoration, any query run, and held-out testing are not authorized by this package. Return real G7 evidence for review and explicit ACK G7 before G8.
