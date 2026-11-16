# DeepTrace — Precision Experimentation Workbench

DeepTrace is the local Streamlit interface for the existing AVFuse-CausalQ research pipeline. Its three sections are **Experiment**, **Results** and **About**. It supports reading accepted historical evidence and explicitly authorized **validation-only** cached-feature inference. Reading saved results never starts a model or evaluator.

## Launch

From the repository root, using Python 3.10+:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pip install -r frontend/requirements.txt
# Optional: directories explicitly trusted by the server operator.
export AVFUSE_ASSET_ROOTS='/absolute/DiMoDif:/absolute/validation-assets'
cd frontend
PYTHONPATH=../src:. python -m streamlit run app.py --server.address 127.0.0.1
```

Launch from `frontend/` to load its light theme, loopback binding and 32 MiB upload cap. Set `AVFUSE_REPO_ROOT` before launch if the intended checkout differs. External roots authorize read access and trusted upstream Python code; configure only directories you trust. The browser cannot change the repository or authorized roots. For restricted macOS filesystem-watcher environments add `--server.fileWatcherType none`.

## Workflow

1. **Open Experiment.** The default source is **Explore accepted research results**. Choose the four existing precision modes and open the saved study. Results identify historical source/split/cohort, original metrics and recorded CPU workloads. Missing clip-level scores remain unavailable.
2. **Choose Run new validation experiment.** Open Execution settings. Select an actual discovered, schema-valid LAV-DF validation manifest when available, or enter an operator-authorized path. Configure external DiMoDif checkout, feature root/format, explicit device, unique run ID and optional experiment ID. All four modes default selected.
3. **Check Readiness.** The frontend checks every required cache path, expected/found/missing counts, manifest/schema/groups, model source/configuration, checkpoint SHA-256, device availability and one bounded real-loader cache. Failures name missing assets and do not launch the experiment CLI or consume a run ID. Success also requires the existing Member 3 `validate-config`; this is a software/assets check, not model inference.
4. **Optionally test one real validation sample.** Select the forward-check option and authorize the trusted assets. Readiness calls the existing `predict --max-clips 1` CLI with temporary output. It discards scores and reports sample compatibility only. This does not guarantee the full cohort will succeed.
5. **Run Experiment.** Explicit authorization and current readiness enable execution. Asset/configuration changes invalidate readiness; material input changes reset authorization. Draft settings and authorization survive navigation. Execution rechecks assets, then calls Member 3 `run --split val --require-lavdf-hash` with explicit modes/device. No fallback, raw media route, overwrite or test access is offered. Existing run directories remain protected.
6. **Open generated results.** Predictions and benchmarks are rediscovered and isolated by the selected run/split/checkpoint. Failed, incomplete and malformed runs remain inspectable. Inference completion leaves **label-grounded evaluation not performed** unless an associated evaluator report actually exists.
7. **Optionally evaluate validation labels.** Supply an existing frozen protocol matching the exact unchanged validation-only manifest/checkpoint and original validation-fixed threshold. The frontend invokes Member 2's evaluator with 1000 source-cluster bootstrap replicates and seed 0, without calibration or test flags. A protocol bound to the historical canonical manifest will be refused for a different validation-only subset. The app never rewrites that protocol to make it fit. If no compatible protocol/labels exist, performance metrics stay unavailable.
8. **Inspect evidence.** Results show available performance, manipulation recall, one-clip score differences and measured computational cost. Technical details expose provenance, failure records, original diagnostics, dated explicit integrity checks and downloads of actual files. An integrity receipt checks file consistency, not authorship or scientific validity.

## Required assets and four configurations

New inference requires the trusted external DiMoDif `src/models.py`, supported LAV-DF `.pth` and adjacent `.json` configuration, a labeled schema-valid validation-only manifest and its compatible features. The expected frozen weight SHA-256 is enforced, never editable in the UI. See the [backend reproduction guide](../docs/09_REPRODUCIBILITY_AND_USER_GUIDE.md) for asset identities and source revision.

- `reference`: FP32 reference using the original model forward.
- `low_precision`: simulated INT8 at Q/K/V and attention output.
- `restore_value_output`: preserve V and attention output at FP32.
- `restore_query`: preserve Q at FP32 as the control.

`dimodif_npz` uses `<feature_root>/val/<clip_id>/features.npz`, with finite floating rank-two `video_features` and `audio_features`, width 768. The manifest's historical planned PT path is ignored in this format. `shared_pt` resolves manifest `feature_path` within the selected root and delegates to the existing safe weights-only loader for `video_tokens`/`audio_tokens`. Escaping cache symlinks are rejected. Readiness loads at most one cache, capped at 32 MiB compressed/expanded; larger representative caches are refused with an inspection-limit explanation. Full tensor compatibility across every clip is established only during actual execution.

## Evidence and scientific limits

Saved CSVs under `docs/results/` retain original artifact `source_sha256` fields. They are aggregate extracts, not new evaluation. Imported evaluation and benchmark artifacts are presented independently and are never automatically attached to a run. New evaluation reports live outside finalized inference bundles, under `artifacts/member3/evaluations/<run>/<unique-id>/`; receipts bind their predictions, manifest, checkpoint, protocol and report hashes. Reports without a matching receipt are not shown as new-run evaluation.

The model uses audited joint-sequence AV self-attention. Simulated INT8 executes floating-point fake quantization, not integer kernels. Saved CPU timing is resident `cached_feature_model` timing, excluding raw-video preprocessing; operator and preprocessing measurements stay separate. CPU peak memory is unmeasured, not zero; theoretical dense-attention storage is not a hardware measurement. Missing external timing samples prevent benchmark verification.

The accepted study does not establish significant overall precision degradation or V/output restoration superiority. Nonsignificance does not prove equality. Source grouping does not establish speaker independence; validation checkpoint-selection bias and historical test-summary exposure remain disclosed. The interface offers no universal detector or deployment claim and never unseals the held-out test set.

This app is intended for a trusted local machine. Uploaded artifacts reach the Streamlit server; they are session inputs, not automatically committed or published. Raw media and arbitrary checkpoint uploads are not supported. Download controls export actual available evidence only.

## Verification

```bash
# From repository root, in the installed environment
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:frontend python -m pytest -q tests frontend/tests
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:frontend python -m ruff check --select F frontend
PYTHONPATH=src python -m avfuse.member3_runtime.cli --help
PYTHONPATH=src python -m avfuse.member3_runtime.cli --smoke
```

The last command uses synthetic diagnostic tokens; its output is never detector evidence. Tests use temporary synthetic fixtures and mocked subprocesses, plus one actual Member 2 fixture evaluation. They do not require licensed assets. Detailed phase history, exact local commands, test counts, browser checks and remaining limitations are recorded in [INTEGRATION.md](INTEGRATION.md).
