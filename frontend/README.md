# AVFuse-CausalQ local research dashboard

This eight-page Streamlit application adapts to the consolidated backend. It displays the completed accepted study and supports explicitly confirmed **validation-only** cached-feature development runs. It does not reproduce the study on launch or imply final release approval.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pip install -r frontend/requirements.txt
# Optional: server-operator-authorized external directories, separated by the OS path separator.
export AVFUSE_ASSET_ROOTS='/absolute/DiMoDif:/absolute/validation-assets'
cd frontend
PYTHONPATH=../src:. python -m streamlit run app.py --server.address 127.0.0.1
```

Launch from `frontend/` to apply its Streamlit configuration (loopback and 32 MiB uploads). The server operator chooses the repository through `AVFUSE_REPO_ROOT`; the browser cannot change it. The application is for a trusted local machine, not a public inference API. If exposed remotely, uploads are processed by the server, not exclusively on the visitor's device. External roots authorize read access and trusted upstream model code; configure only directories you trust.

## Pages

| Page | Purpose |
|---|---|
| Overview | Source/interface checks, repository status, artifact availability and scientific status |
| Dataset & features | Manifest schema, source/identity overlap, distributions and bounded validation-cache metadata inspection |
| Fusion diagnostics | Existing ablation, five-layer attention, mask and theoretical dense-memory reports; explicit missing states |
| Demo & predictions | Confirmed validation CLI execution; predictions isolated by run/split/checkpoint |
| Evaluation | Source-bound committed tables or a separately selected structured metrics JSON |
| Benchmarks | Saved CPU evidence or imported records checked by the backend; measurement identities kept separate |
| Runs & system | Discovered, incomplete and finalized bundles; explicit integrity verification and environment checks |
| Artifact import | Accepted formats, provenance boundaries and upload limitations |

## Execution and assets

`ui.py` handles presentation, `backend_status.py` adapts Member 1 interfaces and invokes Member 3 using subprocess argument lists, and `data_access.py` parses and validates artifacts using Member 2 schemas and Member 3 loaders/verifiers. No second detector exists.

Supply a trusted external DiMoDif checkout with the supported LAV-DF checkpoint and adjacent model configuration, a schema-valid **validation-only** manifest, and a feature root. The wrapper always requests `--split val`, `--require-lavdf-hash`, explicit device/modes and JSON output. It runs `validate-config` first. The backend retains final checkpoint, device and execution authorization checks. No automatic CPU fallback is performed. The form requires confirmation and a unique run ID; resubmitting an ID in the same session is blocked, and existing run directories are refused.

- `dimodif_npz`: `<feature_root>/val/<clip_id>/features.npz`, with floating finite rank-two `video_features` and `audio_features`, width 768.
- `shared_pt`: manifest `feature_path` resolved within the feature root; safely loaded `video_tokens` and `audio_tokens`, supported rank two/three, width 768 and matching batch dimensions.

Metadata inspection is capped at 32 MiB compressed and expanded; valid larger caches can still be inputs to authorized backend inference. Uploaded caches are temporarily inspected with the real loaders and are not inference authorization. No arbitrary checkpoint or raw-video upload is accepted. Test cache inspection is disabled. Missing assets remain unavailable; synthetic smoke is separately labeled software diagnostics, never detector evidence.

## Research and artifacts

Saved tables under [`docs/results`](../docs/results/) preserve original `source_sha256` values. They are aggregate presentations, not a new evaluation. Imported `metrics.json` is selected separately and never overwritten by those tables. Prediction rows do not identify a dataset: establish dataset identity from run provenance before scientific comparison. A missing mode remains missing.

Saved CPU timings are `cached_feature_model`, not raw-video end-to-end timing. Operator and preprocessing scopes remain distinct; stage medians are not summed. CPU peak memory is unmeasured, not zero. Fake INT8 is floating-point simulation, not native integer execution. Overall precision damage and V/output superiority were not statistically established; intervals crossing zero are not equivalence. See the [research index](../docs/README.md) for authoritative limitations and evidence.

Discovered run directories are not automatically valid. Inspect execution/failure summaries and trigger the backend integrity verifier explicitly. Hash consistency does not authenticate authorship. Imported benchmark records with missing external timing samples remain incompletely verified.

```bash
# From repository root
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:frontend python -m pytest -q tests frontend/tests
python -m avfuse.member3_runtime.cli --help
python -m avfuse.member3_runtime.cli --smoke
```

See [INTEGRATION.md](INTEGRATION.md) and the [Phase 4 report](../docs/integration/PHASE_4_FRONTEND_REPORT.md) for verification evidence, limitations and deferred release checks.
