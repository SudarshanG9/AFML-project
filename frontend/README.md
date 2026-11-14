# AVFuse-CausalQ local research dashboard

A local Streamlit dashboard for the consolidated SudarshanG9/AFML-project repository. It uses the real Member 1/2/3 Python packages and local experiment artifacts; it does not create a second detector or fabricate scores.

## Install and launch

From the repository root, run:
    python -m pip install -e ".[dev]"
    python -m pip install -r frontend/requirements.txt
    streamlit run frontend/app.py

Open the local URL printed by Streamlit (normally http://localhost:8501). Set AVFUSE_REPO_ROOT to the checkout path if the app cannot locate the repository automatically.

## Pages

1. Overview — branch/commit, local artifact counts and source-level readiness.
2. Dataset & features — manifest checks, split summaries, feature metadata, shortcut and overlap audits.
3. Fusion diagnostics — ablation, Q/K/V attention boundary, theoretical memory and padding/mask reports.
4. Demo & predictions — import prediction JSONL or run validation-only cached-feature inference through the real Member 3 CLI after supplying local checkpoint, manifest and feature paths.
5. Evaluation — structured metrics and paired confidence intervals from Member 2; Markdown summaries are labelled as documentation.
6. Benchmarks — records by scope, configuration and device.
7. Runs & system — environment, provenance, stage profile and artifact integrity.
8. Artifact import — supported inputs and local import guidance.

## Real inference boundary

The real CLI consumes cached encoder features and an external DiMoDif checkout/checkpoint. Supported formats:
- dimodif_npz: <feature-root>/<split>/<clip_id>/features.npz with video_features and audio_features.
- shared_pt: the manifest feature_path points to a .pt dictionary with video_tokens and audio_tokens.

The Demo page is restricted to the validation split. Do not use it to open the held-out test split. Test execution remains protected by Member 2's protocol/unseal/purpose controls and C9/C10 owner-approval gates.

The dashboard does not upload raw video/audio, arbitrary model checkpoints or labels. It does not run preprocessing or face extraction. The synthetic --smoke control is a software wiring check; its outputs are not deepfake predictions.

## Artifact discovery

The app searches nested data/ and artifacts/ folders for manifests, predictions, benchmark records, Member 3 run folders and diagnostic/evaluation reports. Large licensed media, cached feature payloads and checkpoints may remain local/ignored and are not required to be committed to Git.

See BACKEND_REVIEW.md for the source-to-dashboard mapping, branch audit and remaining research gates.
