# AVFuse-CausalQ

**IO-Aware, Precision-Controlled Audio-Visual Fusion for Deepfake Detection**

> **Implementation status:** Member 1's four-mode DiMoDif attention intervention, Member 2's data/evaluation tooling, and Member 3's cached-feature inference/runtime CLI are integrated. This does not mean the held-out research evaluation or final scientific release is complete.

## Research question

Under source-disjoint, manipulation-stratified evaluation, does simulated INT8-style precision reduction inside a joint audio-visual detector's self-attention change forensic correctness by attack type (audio-fake, video-fake, both-fake)? Can selective value/output-path restoration recover more loss than a query-path restoration control relative to measured added cost?

## Three owner components

| Owner | Code | Responsibility |
|---|---|---|
| Member 1 / Lead | src/avfuse/member1_fusion/ | Pretrained DiMoDif adapter, attention interventions, precision modes and diagnostics |
| Member 2 | src/avfuse/member2_data/ | Dataset manifests, feature cache contracts, split/shortcut audits, prediction checks and evaluation |
| Member 3 | src/avfuse/member3_runtime/ | Feature loading, inference orchestration, CLI, timing, benchmarking, provenance and run integrity |

Shared schemas are in contracts/schemas/. Member 3 command documentation is in docs/member3/CLI.md; the integrated dashboard is in frontend/README.md.

## Install and test

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -e ".[dev]"
    python -m pytest -q
    python -m pip install -r frontend/requirements.txt
    python -m pytest -q frontend/tests
    streamlit run frontend/app.py

## Real cached-feature execution

The real Member 3 commands are validate-config, run, predict, benchmark and inspect. Choose the checkpoint explicitly and provide the external DiMoDif checkout, whole-video manifest and feature root.

Feature loader formats:
- dimodif_npz: <feature-root>/<split>/<clip_id>/features.npz with video_features and audio_features.
- shared_pt: manifest feature_path to a tensor dictionary containing video_tokens and audio_tokens.

This runtime operates on cached encoder features; it is not raw-video end-to-end inference. The synthetic --smoke CLI is a wiring diagnostic only.

## Research integrity boundaries

1. Do not treat fake quantization as a real integer kernel, INT8 speedup or energy result. Tensors remain floating point.
2. Keep isolated operator, cached-feature model and preprocessing measurements separate.
3. Do not invent unavailable memory values; null means not measured.
4. Freeze thresholds and precision policy using validation only. Held-out test remains subject to Member 2 protocol/unseal/purpose controls and C9/C10 owner approvals.
5. Report source/speaker identity disjointness only to the level established by dataset metadata.
6. CI success demonstrates software tests, not forensic accuracy, causal conclusions or a research release.

See docs/BRANCH_CONSOLIDATION_AUDIT_2026-11-14.md for the branch inventory and merge-resolution record.
