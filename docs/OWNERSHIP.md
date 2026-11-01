# Team ownership and handoff agreements

| Workstream | Member 1 | Member 2 | Member 3 |
|---|---|---|---|
| Model/checkpoint and QKV boundary | **Owns** | checks labels compatibility | checks device |
| Raw data access/decoder/feature cache | receives output | **Owns** | times offline stages |
| Cross-modal attention/reference and precision modes | **Owns** | evaluates outputs | benchmarks |
| Accuracy / source-group confidence intervals | sends scores | **Owns** | logs run provenance |
| Runtime, CLI, orchestrator, CI | supplies callable model | supplies test examples | **Owns** |
| Shared schemas / final experiment freeze | reviews | reviews | coordinates |

## Required handoffs

- **End of Day 1**: Member 1 checkpoint ID/checksum/encoder-boundary; Member 2 dataset/label availability; Member 3 JSON schema + device check.
- **End of Day 2**: Member 1 Q/K/V shape report and modality ablation; Member 2 source/identity split audit; Member 3 synthetic end-to-end smoke run.
- **End of Day 4**: Member 1 oracle/SDPA parity; Member 2 cached real-feature manifest; Member 3 operator microbenchmark.
- **End of Day 5**: first integrated FP32 predictions and timings; **hard blocker if missing**.
- **End of Day 6**: low-precision predictions + initial stratified metrics.
- **End of Day 8**: restoration/control predictions + observed timing/cost.
- **Day 9**: freeze policy, detection thresholds, checkpoint, split and protocol using validation only.
- **Days 10–12**: held-out test once, verification, documentation of outcomes, CLI.

## Branches and pull requests

Use `main` as protected integration branch. Suggested branches: `feature/member1-fusion`, `feature/member2-data`, `feature/member3-runtime`. Prefer short feature branches and frequent merges, not three long-lived independent codebases. Add commit/PR labels `[M1]`, `[M2]`, `[M3]`, `[shared]` for transparent contributions. Update this table with actual names after the team decides, without assuming GitHub usernames.

A change under `contracts/` should be reviewed by all three; a change under a member's folder should be approved by that owner. Keep test fixtures synthetic and small.
