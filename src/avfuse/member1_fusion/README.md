# Member 1 — Model + fusion + precision research

**Owns:** checkpoint/model adapter, true Q/K/V interception in actual joint AV detector,
both-modality ablations, IO-aware score-matrix estimate, dense attention correctness
oracle vs SDPA, symmetric simulated INT8, all four precision policies, prediction
export for Member 2. Preserve the original high-precision tensors for restoration.

**Present code:** generic attention wrapper, quantization policy, abstract model
adapter, and a diagnostic model for smoke testing only.

**Not implemented:** a trained real AVFF or other detector integration. Do not report
diagnostic synthetic scores as evidence. First real checkpoint gate: Day 1–2.
