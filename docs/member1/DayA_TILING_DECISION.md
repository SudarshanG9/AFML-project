# Day A — IO-Aware Attention Tiling Decision

## 1. Objective

Determine whether implementing a custom IO-aware tiled attention kernel is justified for the pretrained DiMoDif detector used in AVFuse-CausalQ.

## 2. Observed Model Configuration

- Model: DiMoDif
- Attention mechanism: Joint audio-visual self-attention
- Transformer layers: 5
- Number of attention heads: 4
- Batch size: 1
- Query/key sequence length: 1,202
- Attention head dimension: 16
- Numerical format: FP32

## 3. Analytical Memory Findings

The dense attention-score matrix has dimensions `[1, 4, 1202, 1202]`.

- Dense score storage per layer: **22.046 MiB**
- Combined Q/K/V storage per layer: **0.880 MiB**
- Calculation: `B × H × Nq × Nk × bytes_per_element`

These values represent theoretical tensor storage, not measured runtime memory or peak GPU allocation.

## 4. Padding and Masking Findings

The audited sample contains 235 valid positions per modality, with 365 zero-padded positions per modality.

The model uses a structural attention mask without a separate key-padding mask at the inspected attention boundary.

The mask allows some interactions between valid queries and padded keys. The CLS token can attend to all 730 padded audio-video positions.

The full attention sequence remains 1,202 positions despite the shorter valid input length.

## 5. IO-Aware Engineering Decision

**Decision: NO-GO for implementing a custom attention kernel within the AVFuse-CausalQ implementation schedule.**

Rationale:

1. PyTorch SDPA provides an existing attention interface capable of using optimized execution paths where supported.
2. The current theoretical memory analysis does not establish a bottleneck that requires a new kernel.
3. Designing a custom kernel would introduce additional correctness, numerical stability, masking and hardware compatibility requirements.
4. The primary research objective concerns precision-sensitive forensic predictions, not designing a faster attention algorithm.
5. A dense reference implementation and SDPA allow the project to study numerical interventions without introducing another unverified computational implementation.

## 6. Selected Approach

- Dense scaled dot-product attention: Mathematical correctness oracle
- PyTorch SDPA: Preferred operator for precision experiments and benchmarking
- Custom tiled kernel: Not implemented

This decision does not claim that SDPA is always optimal or that custom kernels could not improve performance.

## 7. Outstanding Performance Verification

The decision must be reviewed against Member 3's operator-level benchmarking results, including:

- Dense versus SDPA latency
- Actual hardware and backend configuration
- Peak allocated memory, where supported
- Attention's contribution to overall runtime

No runtime speedup or measured memory saving is claimed until those results are available.

## 8. Impact on Future Experiments

All four precision configurations will preserve the original pretrained Q/K/V projections, attention-mask semantics and detection head.

The selected implementation allows the project to isolate numerical precision interventions while retaining a verified FP32 reference.