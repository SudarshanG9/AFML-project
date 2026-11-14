# AVFuse-CausalQ local research dashboard

This is a local Streamlit frontend designed against the checked-in backend at:

- Repository: `SudarshanG9/AFML-project`
- Inspected GitHub `main` commit: `902cd7b1096fdb2b745fb593c7230508f0805ff3`

The dashboard is **artifact-driven**. It reads the project's actual manifest, prediction, evaluation, benchmark, environment and run-provenance files. It does not invent demo scores or create a second model implementation.

## Install and launch

From the root of your `AFML-project` checkout:

```bash
# Install the project/backend dependencies if not already installed
python -m pip install -e ".[dev]"

# Install the dashboard-only dependencies
python -m pip install -r frontend/requirements.txt

# Start the local dashboard
streamlit run frontend/app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`).

If the repository is stored elsewhere:

```bash
# Linux/macOS
AVFUSE_REPO_ROOT=/path/to/AFML-project streamlit run frontend/app.py

# Windows PowerShell
$env:AVFUSE_REPO_ROOT="C:\\path\\to\\AFML-project"
streamlit run frontend/app.py
```

## Pages

1. **Overview** — locally discovered artifact counts and backend readiness.
2. **Dataset & features** — manifest schema and split-disjointness checks, group/category summaries, local or in-memory uploaded `.pt` / `.npz` feature metadata inspection, shortcut audits, feature-matched shortcut controls and checkpoint/source overlap reports.
3. **Fusion diagnostics** — modality ablation, Q/K/V boundary audit, theoretical attention-memory estimates and padding/mask audit outputs from Member 1's diagnostic scripts.
4. **Demo & predictions** — import multiple prediction JSONL files, validate score records, inspect score distributions and compare one clip across the four configurations; optional synthetic infrastructure smoke test.
5. **Evaluation** — the actual `metrics.json` shape produced by `scripts/m2_evaluate.py`, including overall/per-type cluster-bootstrap intervals and paired contrasts; backend-generated ROC/PR and recall plots; documented validation summary fallback clearly labelled as documentation-only.
6. **Benchmarks** — proposed schema-v0.2 benchmark JSONL validation and comparisons by scope/configuration/device.
7. **Runs & system** — environment reports, resolved configs, provenance, stage profiles, run manifests and the backend's artifact-integrity verifier.
8. **Artifact import** — supported files, local paths and privacy/input rules.

## What it intentionally does not do

The checked-in backend is not yet a fully wired real inference application. At the inspected repository revision:

- `src/avfuse/member3_runtime/cli.py` only accepts `--smoke`; the synthetic diagnostic uses random tensors and a random toy head.
- `src/avfuse/member1_fusion/dimodif_adapter.py` implements reference inference, but the three non-reference modes raise `NotImplementedError`.
- The abstract `ModelAdapter` does not provide a model factory or a complete four-mode CLI/API.
- The project has benchmark/measurement helpers and artifact-management functions, but no complete real-model benchmark orchestrator or committed benchmark JSONL output set.
- Raw-media upload-to-prediction is not exposed by the backend. Data preparation is Member 2's offline pipeline; the shared token cache expects `audio_tokens` and `video_tokens`.

For those reasons the **Run all four real configurations** control is deliberately disabled. The dashboard does not accept raw video/audio, arbitrary checkpoints, or prediction rows containing labels. It will display actual predictions and measured benchmarks once the backend produces them.

## File discovery and optional import

Default local paths:

```text
data/manifests/*.jsonl
artifacts/predictions/*.jsonl
artifacts/benchmarks/*.jsonl
artifacts/reports/**/metrics.json
artifacts/runs/**/environment_report.json
artifacts/runs/run-*/*
artifacts/day2/ablation_report.json
artifacts/day2/attention_audit.json
artifacts/dayA/attention_memory.json
artifacts/dayA/padding_mask_audit.json
artifacts/reports/**/shortcut_control.json
artifacts/reports/**/train_overlap.json
artifacts/reports/**/check_*.json
```

Manifest, prediction, metrics, benchmark, diagnostic and data-quality pages allow importing the corresponding JSON/JSONL file from the browser. The prediction and benchmark pages accept multiple files because the backend may emit one file per configuration/scope. `.pt` and `.npz` feature uploads are inspected for metadata only. Imported files are parsed in the current Streamlit session; they are not copied into the repository. The app does not access GitHub at runtime.

## Measurement and evaluation semantics preserved in the UI

- `operator` = isolated `fusion_attention()` call on pre-created Q/K/V tensors.
- `cached_feature_model` = cached-feature fusion + head; it excludes raw-media decode and frozen-encoder extraction.
- `preprocessing_once` = raw-media preprocessing as a separate one-off scope.
- `peak_memory_bytes = null` means not measured; it is never rendered as zero.
- Modality-ablation outputs use their own `fake_score`/`zero_audio`/`zero_video` structure; they are displayed as diagnostics, not reinterpreted as prediction-contract rows.
- `dense_oracle` is a correctness oracle, not a performance substitute for SDPA.
- `score_matrix_bytes` is a theoretical dense-score estimate, not the memory allocated by fused SDPA.
- Simulated INT8 keeps floating-point tensors and is not real integer-kernel execution.
- Evaluation metrics are loaded from Member 2's report and are not recomputed in the browser. Prediction rows do not carry ground-truth labels.
- The test split warning makes the test-access count visible; never tune a configuration from the held-out test report.

See [`BACKEND_REVIEW.md`](BACKEND_REVIEW.md) for the backend source-to-page mapping, verified artifact shapes and explicit gaps that informed the UI.

## Known contract status

`contracts/schemas/benchmark.schema.json` is described by `docs/CONTRACTS.md` as a proposed v0.2 contract pending review by Members 1 and 2. The dashboard validates against the checked-in schema but labels it as proposed, not finally approved.

## Troubleshooting

- `ModuleNotFoundError: avfuse`: install from repository root with `python -m pip install -e ".[dev]"`.
- Dashboard says repository not found: set the repository root to the checkout folder containing `pyproject.toml`, `src/`, and `contracts/`.
- No manifests/predictions/benchmarks are discovered: the corresponding local data/artifact folders may be absent because they are ignored by Git. Select/import a file or run the relevant Member 2/3 workflow locally.
