# AVFuse-CausalQ

**IO-Aware, Precision-Controlled Audio-Visual Fusion for Deepfake Detection**

> **Status:** repository scaffold and unit-testable research primitives, **NOT** a trained or validated deepfake detector. Before interpreting results, select a real checkpoint, verify its cross-modal fusion boundary, obtain licensed labeled clips, and complete end-to-end integration.

## Research question

Under source-disjoint, manipulation-stratified evaluation, does simulated INT8-style precision reduction inside a joint audio-visual detector's cross-modal attention change forensic correctness by attack type (audio-fake, video-fake, both-fake)? Can selective value/output-path restoration recover more loss than a query-path restoration control relative to *measured* added cost?

## Three member-owned components

| Member | Code path | Owns | Produces |
|---|---|---|---|
| **Member 1 / Lead** | `src/avfuse/member1_fusion/` | checkpoint adapter, attention oracle/SDPA, numerical interventions, ablations | four model configurations, predictions |
| **Member 2** | `src/avfuse/member2_data/` | licensed data, source/identity-disjoint splits, caching, forensic evaluation | `manifest.jsonl`, tensor cache, stratified metrics |
| **Member 3** | `src/avfuse/member3_runtime/` | timing, peak memory, repeatability, orchestration, CLI | benchmark records, environment details, demo |

Shared schemas are in `contracts/schemas/`; reference decisions and integration gates are in `docs/`.

## Install and run the scaffold

```bash
python3 -m venv .venv
source .venv/bin/activate           # on Windows: .venv\Scripts\activate
python -m pip install -e '.[dev]'
python scripts/check_environment.py
pytest -q
python -m avfuse.member3_runtime.cli --smoke
```

The **smoke CLI intentionally uses synthetic tensors and a randomly initialized toy fusion head**. Its fake-score outputs are not scientific results. It only confirms module wiring.

## Research outputs (planned; not yet generated)

```text
data/cache/<clip_id>.pt          # encoder tokens or an explicitly documented feature boundary
artifacts/predictions/*.jsonl   # one line per clip per configuration
artifacts/benchmarks/*.jsonl    # one line per configuration/device/scope
artifacts/reports/              # validation/test statistics and intervals
artifacts/runs/                 # frozen run config + hashes + software/device info
```

Data, personally sensitive clips, model weights and generated results are excluded from git by default. See `data/README.md`.

## Ground rules

1. **Model:** lock a real, reproducible checkpoint and audit its training-set overlap with evaluation clips before experiments.
2. **No fabricated results:** do not treat the synthetic smoke test as forensic evaluation.
3. **Cache boundary:** when frozen encoder outputs are cached, model-level latency means **cached-feature fusion + head**, not raw-video inference. Separately report raw-media preprocessing and feature-encoding costs.
4. **Precision:** `fake_quant_int8` numerically emulates symmetric quantization while tensors remain floating point. No claim of real 8-bit kernel speedups.
5. **Restoration control:** value **and output** restoration may not cost the same as query-only restoration. Compare empirical added cost; never assume equal work.
6. **Data discipline:** choose the threshold and any precision policy on validation data. Run held-out test scoring only after freeze.
7. **Evaluation:** for each attack-type AUROC/AUPRC, compare that fake class **against real clips**, not a single-positive-class subset.
8. **PRs:** changes to `contracts/`, `configs/`, and orchestration need review from all affected owners.

See `docs/ARCHITECTURE.md`, `docs/OWNERSHIP.md`, `docs/CONTRACTS.md`, and `docs/PLAN_12_DAYS.md`.

Member 3 runtime CLI, architecture, troubleshooting, audit and release checklist: see `docs/member3/` (start with `docs/member3/CLI.md`).
