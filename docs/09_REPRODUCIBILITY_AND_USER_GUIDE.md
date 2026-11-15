# 9. Reproducibility and user guide

## Choose the appropriate reproduction level

A clean checkout supports source inspection, synthetic correctness tests, CLI help/config validation and reading the public aggregate results. Real validation inference additionally requires licensed cached features and the verified external model/checkpoint. Reproducing raw encoder extraction requires another, heavier environment and media access. The accepted held-out experiment is already complete; this guide deliberately provides validation commands without instructions to reopen its seal.

The cleanup branches share the foundation identified in [INTEGRATED_BASE.md](cleanup/INTEGRATED_BASE.md). Each branch is independently reviewable; the lead performs manual merges. The proposed final public tree combines Member 1's code/safety records, Member 2's final reporting, Member 3's current runtime, these chapters and hygiene removal. Older frozen source hashes refer to their original revisions, not every present-day runtime/reporting extension.

## Install and inspect without research assets

From the repository root, use Python 3.10 or newer as required by `pyproject.toml`. An isolated environment prevents your normal Python packages from becoming an accidental dependency:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python scripts/check_environment.py --help
python -m avfuse.member3_runtime.cli --help
python -m avfuse.member3_runtime.cli --smoke
```

The smoke command prints synthetic toy-model scores/timings and explicitly labels them diagnostic. They demonstrate wiring, not deepfake accuracy. The package dependencies include torch, NumPy, scikit-learn and jsonschema; pytest/ruff are development extras. The accepted CPU benchmark environment is recorded in Chapter 6; installing today's compatible versions is not an exact recreation of that historical environment.

Run the synthetic suite with:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -q -p no:cacheprovider
```

Historical tests with private package/candidate inputs skip when those inputs are absent after the Member 1 cleanup is included. The explicit `tests/member1/day6_integration_test.py` program requires real assets and is not the ordinary public pytest suite. A skipped real-input check does not count as newly reproduced real inference.

## Obtain the external DiMoDif dependency

The historical parent repository contains a gitlink at `third_party/dimodif` without a `.gitmodules` mapping. Do not assume recursive clone supplies the model. Use an explicit external checkout, preferably outside the project, and pass its path to the CLI. The inspected local upstream revision is `830fe1ba4a5bfb53b809331fdcbcce0a977cbd36`:

```bash
git clone https://github.com/mever-team/dimodif.git ../dimodif-external
git -C ../dimodif-external checkout 830fe1ba4a5bfb53b809331fdcbcce0a977cbd36
```

Follow the [upstream instructions](https://github.com/mever-team/dimodif) for required dependencies, licensing and checkpoint acquisition. The selected files under that external root are:

```text
src/models.py
ckpt/dfd/dfd_lavdf_reduceonplateau_64_4_1_5_True.json
ckpt/dfd/dfd_lavdf_reduceonplateau_64_4_1_5_True.pth
```

Verify identities before loading the checkpoint. This portable Python command checks bytes without executing the model:

```bash
python - <<'CHECK'
from pathlib import Path
import hashlib
p = Path('../dimodif-external/ckpt/dfd/dfd_lavdf_reduceonplateau_64_4_1_5_True.pth')
expected = 'e1e59006d1bb6ad36a3859af5486a6649ed41bba23258b7c41e24f873be7948e'
with p.open('rb') as f:
    digest = hashlib.file_digest(f, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else None
if digest is None:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    digest = h.hexdigest()
assert digest == expected, f'checkpoint mismatch: {digest}'
print(digest)
CHECK
```

The saved model-config SHA-256 is `bea128dc7dad6b72bc0487dcc45d03ae3322e48fd597de1a5cd83ad036f30651`; upstream `src/models.py` is `e74b5e3b5fcee24353f9940dddd48d9ac1982ced725664db2518f63a6d853778`. A repository revision does not by itself identify a separately acquired checkpoint. The adapter strictly loads the state dictionary, freezes parameters and disables the MHA fastpath to preserve hooks used by the upstream feature pyramid.

## Supply a validation manifest and feature cache

For accepted-study validation use the supplied private `manifest_whole_val.jsonl`, hash `22d1d9e02403cfdbb05f2116da16bc4015f540547a4ac00bf6b7127c119e359c`, and its 1,200 cached examples. Do not regenerate its IDs or substitute the old windowed manifest. For a separate development demo, a supported manifest must still satisfy the shared schema/group checks and be clearly identified as a different cohort.

The expected NPZ layout is:

```text
private_inputs/
  manifest_whole_val.jsonl
  dimodif_emb/
    val/<clip_id>/features.npz
    val/<clip_id>/meta.json
```

Each NPZ contains float arrays `video_features (T_v,768)` and `audio_features (T_a,768)`. The explicit `--feature-format dimodif_npz` selects this mapping; the manifest's old PT `feature_path` is not opened. Do not rename array keys, align/pad them a second time, or substitute arbitrary embeddings. Metadata and original cache identities should accompany supplied features.

For independent extraction, inspect the supported interface first:

```bash
python scripts/m2_whole_video_manifest.py --help
python scripts/m2_dimodif_features.py --help
```

The latter requires the external auto-AVSR project, visual/audio models, MediaPipe and compatible torch/vision/audio dependencies. [The historical Member 2 runbook](member2/RUNBOOK.md) records the Python 3.10 extraction setup. Installation of the lightweight research package alone does not install that full raw-media pipeline. Supply media under its access terms and preserve extraction provenance; never add the resulting media or cache to Git.

## Run a validation-only demonstration

Set the three paths to your supplied assets. These shell variables are task-specific; quotes protect spaces in paths.

```bash
AVFUSE_UPSTREAM='../dimodif-external'
AVFUSE_MANIFEST='private_inputs/manifest_whole_val.jsonl'
AVFUSE_FEATURES='private_inputs/dimodif_emb'
python -m avfuse.member3_runtime.cli validate-config \
  --upstream-root "$AVFUSE_UPSTREAM" --checkpoint-name lavdf \
  --manifest "$AVFUSE_MANIFEST" --feature-root "$AVFUSE_FEATURES" \
  --feature-format dimodif_npz --device cpu --split val --require-lavdf-hash
python -m avfuse.member3_runtime.cli predict \
  --upstream-root "$AVFUSE_UPSTREAM" --checkpoint-name lavdf \
  --manifest "$AVFUSE_MANIFEST" --feature-root "$AVFUSE_FEATURES" \
  --feature-format dimodif_npz --device cpu --split val --require-lavdf-hash \
  --modes reference low_precision restore_value_output restore_query \
  --max-clips 1 --run-id validation-demo --output artifacts/validation-demo.jsonl
```

A one-clip subset is a **partial development run**, so the CLI's incomplete exit code 4 can be expected even when it writes valid subset predictions. It is not a complete experiment. Omit `--max-clips 1` only when intentionally running the full permitted validation cohort. The cleanup verified argument/help and synthetic behavior; it did not rerun this real-asset inference.

For a new validation benchmark on a supplied sample, `benchmark --help` lists the supported warmup/repeat options. That command measures resident cached-model inference only and creates new development measurements, which must not replace accepted original results. The `run` command combines prediction and benchmarking with manifests/provenance; `inspect <run_dir>` checks an existing run's integrity and completion status. Partial runs are not finalized as complete research runs.

## Inspect predictions and read results

Prediction rows must have exactly the agreed fields: run ID, clip ID, split, config, score and checkpoint hash. The scores are probabilities in [0,1], without embedded labels. Check schema/coverage against the permitted validation manifest with:

```bash
python scripts/m2_check_predictions.py \
  --manifest "$AVFUSE_MANIFEST" --predictions artifacts/validation-demo.jsonl \
  --out artifacts/validation-demo-check.json
```

For validation, this checker also reports orientation and label-based readiness diagnostics; it is not a schema-only operation. A subset fails the full-coverage contract check; use a full-cohort prediction file for a successful full-coverage check. This is intentional, rather than permission to delete unpredicted manifest rows or claim completeness for a demonstration. `m2_evaluate.py --help` and its `calibrate`/`evaluate` subcommand help describe the label-based evaluator. Reproducing the accepted validation comparison requires its already frozen protocol, not a newly selected threshold. Held-out unseal flags are not part of this guide.

Read [Chapter 6](06_EXPERIMENTS_AND_RESULTS.md), the [saved-result CSV index](results/README.md) and original [Member 2 final report](member2/results/TEST_RESULTS.md) for accepted results. CSVs are presentation extracts of existing aggregates, with source hashes; they contain no clip-level labels or predictions. [Chapter 10](10_EVIDENCE_AND_REFERENCES.md) links the accepted benchmark sources and distinguishes original artifacts from display tables. To inspect a finalized current-runtime run use `python -m avfuse.member3_runtime.cli inspect path/to/run --json`; this validates the saved manifest, not the truth of the original data collection.

## Common errors and clean-checkout limits

| Symptom | Explanation and response |
| --- | --- |
| Missing checkpoint/config/models.py | external DiMoDif is not populated; supply the explicit external root and required files |
| Checkpoint mismatch, exit 7 | the bytes differ from the expected model; obtain the correct asset rather than disabling the identity check |
| NPZ keys/rank/channels invalid | use the auto-AVSR cache format; generic PT tensors or raw frames are not the accepted input |
| Missing features despite a manifest path | use explicit split/clip NPZ mapping; the historical PT placeholder does not identify the real cache |
| Partial/incomplete, exit 4 | the requested subset or supported modes are not a complete full-cohort run |
| Sealed-test denied, exit 5 | the operation requests test access; use validation for demonstrations; this guide grants no new test authorization |
| Integrity failure | compare saved hashes and original artifacts; do not rewrite an inventory merely to silence a discrepancy |
| Accelerator unavailable | select a device actually available; the runtime refuses silent fallback |
| No CPU peak-memory value | the accepted method does not measure per-call CPU tensor peak memory; null is not zero |

A clean public tree does not contain licensed raw videos, the complete feature cache, canonical private test labels, the detector weight file, private handoff/recovery archives or every original candidate-source fixture. Synthetic tests and public summaries remain useful without them. Those boundaries are part of reproducibility: supplying the correct external assets and original identities is different from claiming every experiment can run from an unaided clone.
