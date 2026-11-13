# Member 3 CLI

```
python -m avfuse.member3_runtime.cli <command> [options]      # or: python -m avfuse.member3_runtime
```

Everything here is **cached-feature** inference (frozen encoder tokens in, `score_fake` out). It is not raw-video
end-to-end latency, computes no accuracy, and the CLI never edits labels, thresholds or Member 1/2 code.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
python -m avfuse.member3_runtime.cli --smoke     # synthetic wiring check only
```

## Commands

| Command | What it does |
|---|---|
| `validate-config` | Checks arguments, benchmark config (schema), manifest, feature root, checkpoint files. Loads no model, reads no feature. |
| `run` | Predictions + cached-feature benchmark inside one Segment D run directory; finalizes and verifies only when complete. |
| `predict` | Writes only a validated prediction JSONL (no run directory). |
| `benchmark` | Cached-feature benchmark on one **validation** sample (no test options exist). |
| `inspect RUN_DIR` | Recomputes artifact hashes and summarizes status; read-only. |
| `--smoke` | Legacy toy-model wiring check. Not a result. |

Shared options: `--upstream-root`, `--checkpoint-name {lavdf,avdeepfake1m}` (always explicit), `--manifest`
(whole-video `manifest_whole.jsonl`), `--feature-root` (`<root>/<split>/<video_id>/features.npz`),
`--feature-format {dimodif_npz,shared_pt}`, `--device {cpu,cuda[:N],mps}` (no `auto`, no CPU fallback),
`--modes` (default all four), `--allow-partial`, `--max-clips`, `--expected-checkpoint-sha256` /
`--require-lavdf-hash`, `--json`. Run options: `--runs-root` (default `artifacts/member3/runs`, git-ignored),
`--run-id`, `--experiment-id`, `--overwrite`, `--no-hash-cache`, `--warmup`, `--repeats`, `--percentile-method`,
`--no-memory`, `--benchmark-clip-id`.

## Realistic examples

```bash
# 1. Check everything first (validation split; nothing is loaded)
python -m avfuse.member3_runtime.cli validate-config \
  --upstream-root ~/DiMoDif --checkpoint-name lavdf \
  --manifest data/manifest_whole.jsonl --feature-root /path/to/features

# 2. Reference-only development run on validation (what the current adapter supports)
python -m avfuse.member3_runtime.cli run \
  --upstream-root ~/DiMoDif --checkpoint-name lavdf --require-lavdf-hash \
  --manifest data/manifest_whole.jsonl --feature-root /path/to/features \
  --modes reference --device cpu --experiment-id day5-reference-val

# 3. Strict four-mode run (exits 3 until Member 1 integrates all modes)
python -m avfuse.member3_runtime.cli run ... --device mps

# 4. Review a run
python -m avfuse.member3_runtime.cli inspect artifacts/member3/runs/<run_id>

# 5. Final held-out test pass, only after the validation decision is frozen
python -m avfuse.member3_runtime.cli run ... --split test \
  --protocol artifacts/protocol.json --unseal-test --purpose "final held-out evaluation"
```

Validation runs hash `<feature-root>/val` only (so provenance never reads test features). Hashing 1,200 videos takes
time; `--no-hash-cache` skips it and the run then records no cache hash.

## Exit codes

| Code | Meaning | Examples |
|---|---|---|
| 0 | Success, run complete and finalized | all requested modes completed on the whole split |
| 1 | Runtime failure | model error, invalid scores, integrity check failed, `failure_summary.json` written |
| 2 | Invalid usage/configuration | bad flags, `--device auto`, test flags on val, run id already exists |
| 3 | Blocked: unsupported mode in a strict run | `low_precision` etc. raise `NotImplementedError`; nothing published |
| 4 | Incomplete: not releasable | `--allow-partial`, `--max-clips`, unfinalized run |
| 5 | Sealed test access denied | missing `--protocol`/`--unseal-test`/`--purpose` |
| 6 | Missing input | manifest, feature root, checkpoint, upstream root, run dir |
| 7 | Checkpoint hash mismatch | `--expected-checkpoint-sha256` / `--require-lavdf-hash` failed |

Errors go to stderr as `ERROR[NAME]: message`; `--json` prints a machine-readable summary to stdout.

## Test-split safeguards

- `--split test` needs an existing `--protocol` file, `--unseal-test` and a non-empty `--purpose`; otherwise exit 5
  before any model or feature is touched.
- Those flags are rejected for `--split val`, so test access cannot be implied by a validation run.
- Test runs do not benchmark (`--benchmark` with test is rejected); benchmark on validation.
- Before opening the seal the CLI runs every requested mode on one *validation* clip. A blocked configuration exits 3
  and writes **no** ledger entry. Otherwise Member 2's `open_test` logs the access once, with all configs.
