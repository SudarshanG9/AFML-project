from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from .data_access import (
    CONFIGS, documented_validation_table, documented_test_results, documented_benchmark_summary,
    human_bytes, load_json, load_jsonl, manifest_summary, parse_jsonl_text,
    prediction_summary, shortcut_audit_summary, feature_file_summary_bytes, documented_shortcut_table,
    validate_benchmarks, validate_manifest, validate_metrics_report, validate_predictions,
)
from .backend_status import (git_info, inspect_backend, run_cached_feature_experiment, run_diagnostic_smoke, run_environment_check)


CFG_LABELS = {
    "reference": "Reference · FP32",
    "low_precision": "Low precision · simulated INT8",
    "restore_value_output": "Value/output restored",
    "restore_query": "Query restored · control",
}


def inject_styles() -> None:
    st.markdown("""
    <style>
    :root { --avf-bg: #0b1220; --avf-panel: #111c2e; --avf-line: rgba(148,163,184,.16); }
    .stApp { background: #0b1220; }
    [data-testid="stSidebar"] { background: #0e1828; border-right: 1px solid rgba(148,163,184,.14); }
    [data-testid="stHeader"] { background: rgba(11,18,32,.88); }
    .block-container { max-width: 1440px; padding-top: 1.4rem; padding-bottom: 3rem; }
    .avf-brand { display:flex; align-items:center; gap:.65rem; font-size:1.05rem; font-weight:750; letter-spacing:-.02em; }
    .avf-mark { width:34px; height:34px; border-radius:11px; display:grid; place-items:center; background:linear-gradient(135deg,#2c80ff,#31b8a6); color:white; font-weight:800; }
    .avf-eyebrow { color:#7dd3fc; font-size:.72rem; letter-spacing:.13em; text-transform:uppercase; font-weight:700; }
    .avf-subtitle { color:#9badc4; margin-top:-.35rem; }
    .avf-card { background:linear-gradient(180deg,rgba(20,34,55,.96),rgba(16,28,46,.96)); border:1px solid rgba(148,163,184,.16); border-radius:16px; padding:16px 18px; height:100%; }
    .avf-card-label { color:#91a4bd; font-size:.76rem; text-transform:uppercase; letter-spacing:.07em; }
    .avf-card-value { color:#eff6ff; font-size:1.55rem; font-weight:750; margin-top:.28rem; letter-spacing:-.03em; }
    .avf-card-foot { color:#91a4bd; font-size:.76rem; margin-top:.18rem; }
    .avf-note { border:1px solid rgba(56,189,248,.23); background:rgba(14,116,144,.10); padding:12px 14px; border-radius:12px; color:#c4eafa; }
    .avf-warn { border:1px solid rgba(245,158,11,.25); background:rgba(180,83,9,.10); padding:12px 14px; border-radius:12px; color:#f9dfb4; }
    .avf-danger { border:1px solid rgba(248,113,113,.25); background:rgba(153,27,27,.10); padding:12px 14px; border-radius:12px; color:#fecaca; }
    .avf-small { color:#91a4bd; font-size:.82rem; }
    div[data-testid="stMetric"] { background:rgba(17,28,46,.75); border:1px solid rgba(148,163,184,.14); padding:13px 14px; border-radius:14px; }
    div[data-testid="stMetricLabel"] p { color:#9badc4; }
    div[data-testid="stMetricValue"] { color:#eff6ff; }
    div[data-testid="stDataFrame"] { border:1px solid rgba(148,163,184,.14); border-radius:12px; overflow:hidden; }
    .stButton>button { border-radius:10px; min-height:2.6rem; }
    hr { border-color:rgba(148,163,184,.14); }
    </style>
    """, unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(f'<div class="avf-eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="avf-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def card(label: str, value: str, foot: str = "") -> None:
    st.markdown(
        f'<div class="avf-card"><div class="avf-card-label">{label}</div>'
        f'<div class="avf-card-value">{value}</div><div class="avf-card-foot">{foot}</div></div>',
        unsafe_allow_html=True,
    )


def read_uploaded(upload: Any) -> str:
    return upload.getvalue().decode("utf-8-sig")


def load_jsonl_with_error(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        return load_jsonl(path)
    except Exception as exc:
        st.error(f"Could not read {label} `{path.name}`: {type(exc).__name__}: {exc}")
        return []


def parse_uploaded_jsonl_with_error(upload: Any, label: str) -> list[dict[str, Any]]:
    try:
        return parse_jsonl_text(read_uploaded(upload))
    except Exception as exc:
        st.error(f"Could not parse uploaded {label} `{upload.name}`: {type(exc).__name__}: {exc}")
        return []


def choose_many_local_or_upload(root: Path, label: str, paths: list[Path], key: str,
                                 file_types: list[str]) -> tuple[list[Path], list[Any] | None]:
    """Select multiple local artifacts or upload multiple files for this session only."""
    col1, col2 = st.columns([1.25, 1])
    with col1:
        options = [str(p.relative_to(root)) if p.is_relative_to(root) else str(p) for p in paths]
        chosen = st.multiselect(f"Local {label}", options, key=f"{key}_local_many")
    with col2:
        uploaded = st.file_uploader(
            f"Or import {label}", type=file_types, accept_multiple_files=True,
            key=f"{key}_upload_many",
        )
    selected_paths = [
        p for p in paths
        if (str(p.relative_to(root)) if p.is_relative_to(root) else str(p)) in chosen
    ]
    return selected_paths, (uploaded or None)


def choose_local_or_upload(root: Path, label: str, paths: list[Path], key: str,
                            file_types: list[str]) -> tuple[Path | None, Any | None]:
    col1, col2 = st.columns([1.25, 1])
    with col1:
        options = [str(p.relative_to(root)) if p.is_relative_to(root) else str(p) for p in paths]
        chosen = st.selectbox(f"Local {label}", ["— choose a file —", *options], key=f"{key}_local")
    with col2:
        uploaded = st.file_uploader(f"Or import {label}", type=file_types, key=f"{key}_upload")
    selected_path = None
    if chosen != "— choose a file —":
        selected_path = next((p for p in paths if (str(p.relative_to(root)) if p.is_relative_to(root) else str(p)) == chosen), None)
    return selected_path, uploaded


def panel_title(title: str, caption: str | None = None) -> None:
    st.subheader(title)
    if caption:
        st.caption(caption)


def render_overview(root: Path, files: dict[str, list[Path]]) -> None:
    page_header("AVFUSE-CAUSALQ / RESEARCH WORKSPACE", "Experiment overview",
                "A local, artifact-driven dashboard for precision-controlled audio-visual fusion experiments.")
    status = inspect_backend(root)
    git = git_info(root)
    st.markdown("<div class='avf-warn'><b>Research integrity:</b> this dashboard never invents metrics. Empty states mean the corresponding artifact is not present locally. Simulated INT8 is floating-point fake quantization, not an INT8 kernel.</div>", unsafe_allow_html=True)
    st.write("")
    manifest_count = len(files["manifests"])
    prediction_count = len(files["predictions"])
    benchmark_count = len(files["benchmarks"])
    metrics_count = len(files["metrics"])
    run_count = len(files["run_dirs"])
    cols = st.columns(5)
    metrics = [
        ("Manifests", str(manifest_count), "local JSONL files"),
        ("Prediction files", str(prediction_count), "local JSONL files"),
        ("Benchmark files", str(benchmark_count), "local JSONL files"),
        ("Evaluation reports", str(metrics_count), "structured metrics.json"),
        ("Run folders", str(run_count), "with run provenance"),
    ]
    for col, (label, value, foot) in zip(cols, metrics):
        with col:
            card(label, value, foot)
    st.write("")
    left, right = st.columns([1.15, 1])
    with left:
        panel_title("Backend readiness", "Detected from the checked-in source tree, not from assumptions in the proposal.")
        rows = [
            ("Python package / shared contracts", (root / "pyproject.toml").is_file() and (root / "contracts/schemas").is_dir(), "Package and JSON schemas found"),
            ("Four-mode model inference", status["four_mode_model_ready"], status["adapter_mode_status"]),
            ("Four-mode cached-feature CLI", status["real_inference_ready"], status["cli_status"]),
            ("Local prediction outputs", prediction_count > 0, f"{prediction_count} files discovered"),
            ("Structured benchmark outputs", benchmark_count > 0, f"{benchmark_count} files discovered"),
        ]
        for label, ok, detail in rows:
            icon = "✅" if ok else "⏳"
            st.markdown(f"**{icon} {label}**  \n<span class='avf-small'>{detail}</span>", unsafe_allow_html=True)
            st.write("")
    with right:
        panel_title("Repository snapshot")
        st.markdown(f"<div class='avf-card'><div class='avf-card-label'>Repository</div><div class='avf-card-value' style='font-size:1.12rem'>AFML-project</div><div class='avf-card-foot'>SudarshanG9/AFML-project</div><hr><div class='avf-card-label'>Branch / commit</div><div style='color:#e2e8f0;font-weight:650;margin-top:5px'>{git.get('branch') or 'not available'} · {git.get('commit') or 'not available'}</div><div class='avf-card-foot'>Working tree dirty: {git.get('dirty') or 'unknown'}</div></div>", unsafe_allow_html=True)
        st.write("")
        st.info("The application reads local files under data/ and artifacts/. It does not transmit dataset clips or experiment files to a server.")
    st.divider()
    panel_title("What the checked-in backend actually supports")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Available building blocks**")
        st.markdown("- Member 1 pretrained DiMoDif adapter with four precision-mode attention interventions\n- Member 2 manifest, prediction validation, evaluation, paired analysis and sealed-test controls\n- Member 3 cached-feature CLI, explicit NPZ/PT loaders, run orchestration, benchmarks and provenance\n- Shared schemas, diagnostics, tests and CI workflow")
    with c2:
        st.markdown("**Still requires real local inputs / research authorization**")
        st.markdown("- External DiMoDif source checkout, matching checkpoint and configuration\n- Complete local validation manifest and feature cache in an explicit loader format\n- Raw-media upload-to-inference is not provided; preprocessing remains separate\n- C9/C10 freeze and held-out test authorization remain independent research gates")
    for note in status.get("notes", []):
        st.caption(note)
    st.caption("Readiness is checked from the current local checkout. It does not verify external weights, reproduce metrics or authorize test access.")


def render_dataset(root: Path, files: dict[str, list[Path]]) -> None:
    page_header("DATA / INPUT CONTRACT", "Dataset & cached features",
                "Inspect the manifest and the feature-cache boundary without uploading or decoding raw media.")
    st.markdown("<div class='avf-note'><b>Input rule:</b> the experiment operates on post-encoder audio/video features. Raw AV decoding and feature extraction belong to Member 2's offline pipeline and are not repeated during precision sweeps.</div>", unsafe_allow_html=True)
    path, upload = choose_local_or_upload(root, "manifest.jsonl", files["manifests"], "manifest", ["jsonl", "txt"])
    rows: list[dict[str, Any]] = []
    source = None
    if upload is not None:
        rows = parse_uploaded_jsonl_with_error(upload, "manifest JSONL")
        source = f"uploaded: {upload.name} (session only)"
    elif path:
        rows = load_jsonl_with_error(path, "manifest JSONL")
        source = str(path)
    if not rows:
        st.info("No manifest selected. Choose a local `data/manifests/*.jsonl` file or import a manifest JSONL file to inspect it.")
        st.markdown("**Expected manifest fields**: `clip_id`, `dataset`, `source_id`, `identity_id`, `split`, `label`, `manipulation_type`, `feature_path`.")
        st.caption("LAV-DF's documented identity field is set equal to source_id because speaker IDs are not present; the dataset notes this as a limitation.")
    else:
        schema = root / "contracts" / "schemas" / "manifest.schema.json"
        errors = validate_manifest(rows, schema)
        summary = manifest_summary(rows)
        st.caption(f"Source: {source} · {len(rows):,} rows")
        if errors:
            st.error(f"Manifest checks found {len(errors)} issue(s). Do not use this manifest for a run until resolved.")
            with st.expander("Validation details"):
                st.code("\n".join(errors[:50]))
        else:
            st.success("Manifest passes the checked-in JSON schema and source/identity split-disjoint checks.")
        c = st.columns(4)
        for col, (label, value) in zip(c, [("Clips", summary["clips"]), ("Source groups", summary["source_groups"]),
                                          ("Identity groups", summary["identity_groups"]), ("Feature paths", summary["feature_paths_present"]) ]):
            with col:
                st.metric(label, f"{value:,}")
        split_df = pd.DataFrame([
            {"Split": split, **{typ: summary["types"].get(split, {}).get(typ, 0) for typ in MANIFEST_DISPLAY_TYPES},
             "Total": count}
            for split, count in sorted(summary["splits"].items())
        ])
        panel_title("Clip distribution", "Categories: real, audio fake, video fake and both fake.")
        if not split_df.empty:
            st.dataframe(split_df, use_container_width=True, hide_index=True)
            long = pd.DataFrame([{"Split": split, "Manipulation type": typ.replace("_", " "), "Clips": n}
                                 for split, types in summary["types"].items() for typ, n in types.items()])
            if not long.empty:
                fig = px.bar(long, x="Split", y="Clips", color="Manipulation type", barmode="group", template="plotly_dark")
                fig.update_layout(height=340, margin=dict(l=10,r=10,t=25,b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
        panel_title("Inspect one cached feature file", "This reads tensor/array metadata only; it does not call the detector.")
        clip_ids = [str(r.get("clip_id")) for r in rows if r.get("clip_id")]
        if clip_ids:
            selected_clip = st.selectbox("Clip ID", clip_ids, key="feature_clip")
            selected_row = next((r for r in rows if r.get("clip_id") == selected_clip), {})
        else:
            selected_row = {}
            st.warning("No usable clip_id values are available for feature inspection.")
        feature_path = Path(str(selected_row.get("feature_path", "")))
        if not feature_path.is_absolute():
            feature_path = root / feature_path
        try:
            feature_path = feature_path.resolve()
            feature_inside_repo = feature_path.is_relative_to(root.resolve())
        except OSError:
            feature_inside_repo = False
        uploaded_feature = st.file_uploader("Optionally import one feature cache for metadata inspection", type=["npz", "pt"], key="feature_metadata_upload")
        if uploaded_feature is not None:
            if st.button("Inspect uploaded feature metadata", key="inspect_uploaded_feature"):
                try:
                    info = feature_file_summary_bytes(uploaded_feature.getvalue(), uploaded_feature.name)
                    st.json(info)
                except Exception as exc:
                    st.error(f"Feature inspection failed: {type(exc).__name__}: {exc}")
        elif feature_inside_repo and feature_path.is_file():
            st.caption(f"Feature path: {feature_path.relative_to(root.resolve())}")
            if st.button("Inspect cached feature shapes", key="inspect_feature"):
                try:
                    from .data_access import feature_file_summary
                    info = feature_file_summary(feature_path)
                    st.json(info)
                except Exception as exc:
                    st.error(f"Feature inspection failed: {type(exc).__name__}: {exc}")
        else:
            if not feature_inside_repo:
                st.warning("The manifest feature_path resolves outside the selected repository; the dashboard will not read paths outside the checkout.")
            else:
                st.warning(f"Feature file is not present at the manifest path: {feature_path}")
        st.caption("The repo's DiMoDif handoff reports NPZ arrays named `audio_features` / `video_features`, while the shared cache contract and adapter expect `audio_tokens` / `video_tokens`. The dashboard flags this mismatch; it does not silently rename or convert feature tensors.")
        st.caption("Raw video/audio upload is intentionally not offered: no checked-in raw-media-to-prediction UI/API exists, and licensed media is meant to stay in the local Member 2 pipeline.")

    st.divider()
    panel_title("Data-quality and leakage reports", "These are separate QA artifacts; a clean split audit does not imply that every nuisance feature is clean.")
    audit_path, audit_upload = choose_local_or_upload(root, "shortcut_audit.json", files.get("shortcut_audits", []), "shortcut_audit", ["json"])
    audit = None
    if audit_upload is not None:
        try: audit = json.loads(read_uploaded(audit_upload))
        except Exception as exc: st.error(f"Could not parse shortcut audit: {exc}")
    elif audit_path:
        try: audit = load_json(audit_path)
        except Exception as exc: st.error(f"Could not read shortcut audit: {exc}")
    if audit:
        summary = shortcut_audit_summary(audit)
        c1, c2, c3 = st.columns(3)
        c1.metric("Audit verdict", "Passed" if summary["passed"] is True else ("Flags found" if summary["passed"] is False else "Unspecified"))
        c2.metric("Flagged feature/group rows", summary["flag_count"])
        c3.metric("Feature measurements", summary["feature_measurements"])
        st.caption(f"Rule: {summary['rule']}")
        flags = summary["flags"]
        if flags:
            st.warning(f"{len(flags)} potential shortcut flag(s) were recorded. A flag identifies a feature that separates classes; it does not prove the detector uses that feature.")
            st.dataframe(pd.DataFrame({"Flag": flags}), use_container_width=True, hide_index=True)
        else:
            st.success("No shortcut flags in this audit artifact.")
        with st.expander("Inspect full shortcut audit JSON"):
            st.json(audit)
    else:
        st.info("No `shortcut_audit.json` selected or discovered. The backend writes this under the reports directory when `scripts/m2_audit.py` or its whole-video counterpart is run.")
        documented_audit = root / "docs/member2/results/AUDIT_WHOLE_RESULTS.md"
        if documented_audit.is_file():
            doc_text = documented_audit.read_text(encoding="utf-8", errors="replace")
            doc_rows = documented_shortcut_table(doc_text)
            if doc_rows:
                st.warning("Showing the repository's committed whole-video shortcut-audit summary, not a local machine-readable `shortcut_audit.json`. The recorded values are historical experiment evidence and should be read with the caveats in the source document.")
                st.dataframe(pd.DataFrame(doc_rows), use_container_width=True, hide_index=True)
                with st.expander("Read committed audit summary"):
                    st.markdown(doc_text)
    overlap_path, overlap_upload = choose_local_or_upload(root, "train-overlap report", files.get("train_overlap_reports", []), "train_overlap", ["json"])
    overlap = None
    if overlap_upload is not None:
        try: overlap = json.loads(read_uploaded(overlap_upload))
        except Exception as exc: st.error(f"Could not parse train-overlap report: {exc}")
    elif overlap_path:
        try: overlap = load_json(overlap_path)
        except Exception as exc: st.error(f"Could not read train-overlap report: {exc}")
    if overlap:
        st.metric("Checkpoint training-source overlap", "Clean" if overlap.get("clean") is True else ("Overlap detected" if overlap.get("clean") is False else "Not stated"))
        st.json(overlap)

    control_path, control_upload = choose_local_or_upload(root, "shortcut_control.json", files.get("shortcut_controls", []), "shortcut_control", ["json"])
    control = None
    if control_upload is not None:
        try: control = json.loads(read_uploaded(control_upload))
        except Exception as exc: st.error(f"Could not parse shortcut control: {exc}")
    elif control_path:
        try: control = load_json(control_path)
        except Exception as exc: st.error(f"Could not read shortcut control: {exc}")
    if control:
        st.write("**Feature-matched shortcut control**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Matched videos", f"{control.get('n_matched', '—')} / {control.get('n_all', '—')}")
        c2.metric("Feature AUROC · all", f"{control.get('feature_shortcut_auroc_all', float('nan')):.3f}" if isinstance(control.get("feature_shortcut_auroc_all"), (int, float)) else "—")
        c3.metric("Feature AUROC · matched", f"{control.get('feature_shortcut_auroc_matched', float('nan')):.3f}" if isinstance(control.get("feature_shortcut_auroc_matched"), (int, float)) else "—")
        detector_all = control.get("detector_auroc_all", {})
        detector_matched = control.get("detector_auroc_matched", {})
        st.dataframe(pd.DataFrame([
            {"Subset": "All videos", "Detector AUROC": detector_all.get("value"), "95% CI": f"[{detector_all.get('ci_low', float('nan')):.4f}, {detector_all.get('ci_high', float('nan')):.4f}]" if detector_all else "—"},
            {"Subset": "Feature-matched", "Detector AUROC": detector_matched.get("value"), "95% CI": f"[{detector_matched.get('ci_low', float('nan')):.4f}, {detector_matched.get('ci_high', float('nan')):.4f}]" if detector_matched else "—"},
        ]), use_container_width=True, hide_index=True)
        st.metric("AUROC drop after matching", f"{control.get('drop'):+.4f}" if isinstance(control.get('drop'), (int, float)) else "—")
        st.caption(f"Feature: {control.get('feature', 'not recorded')} · Configuration: {control.get('config', 'not recorded')} · Split: {control.get('split', 'not recorded')} · A small matched-subset drop is consistent with less dependence on that nuisance feature; interpret alongside confidence intervals.")


MANIFEST_DISPLAY_TYPES = ("real", "audio_fake", "video_fake", "both_fake")



def _load_optional_json_artifact(root: Path, label: str, paths: list[Path], key: str) -> tuple[dict[str, Any] | None, str | None]:
    path, upload = choose_local_or_upload(root, label, paths, key, ["json"])
    if upload is not None:
        try:
            obj = json.loads(read_uploaded(upload))
            if not isinstance(obj, dict):
                raise ValueError("Expected a JSON object")
            return obj, f"uploaded: {upload.name} (session only)"
        except Exception as exc:
            st.error(f"Could not parse {label}: {type(exc).__name__}: {exc}")
            return None, None
    if path:
        try:
            return load_json(path), str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        except Exception as exc:
            st.error(f"Could not read {label}: {type(exc).__name__}: {exc}")
    return None, None


def render_diagnostics(root: Path, files: dict[str, list[Path]]) -> None:
    page_header("FUSION / MEMBER 1 DIAGNOSTICS", "Fusion diagnostics",
                "Inspect the backend's modality-ablation, attention-boundary, dense-memory estimate and padding/mask reports. These are diagnostics, not benchmark accuracy results.")
    st.markdown("<div class='avf-note'><b>Interpretation guard:</b> the Day 2 attention audit inspects a joint-sequence self-attention block in the DiMoDif detector. The score-matrix size is a theoretical dense estimate, not the memory allocated by fused SDPA. No diagnostic artifact by itself establishes end-to-end inference performance.</div>", unsafe_allow_html=True)

    panel_title("1. Modality ablation", "Produced by `scripts/day2_inference.py` as `artifacts/day2/ablation_report.json`.")
    ablation, ablation_source = _load_optional_json_artifact(root, "ablation_report.json", files.get("ablation_reports", []), "ablation_report")
    if ablation:
        st.caption(f"Source: {ablation_source}")
        summary = ablation.get("summary", {})
        a, b, c = st.columns(3)
        a.metric("Clips inspected", summary.get("total_clips", "—"))
        b.metric("Mean audio ablation logit change", f"{summary['mean_audio_logit_change']:.5g}" if isinstance(summary.get("mean_audio_logit_change"), (int, float)) else "—")
        c.metric("Mean video ablation logit change", f"{summary['mean_video_logit_change']:.5g}" if isinstance(summary.get("mean_video_logit_change"), (int, float)) else "—")
        results = ablation.get("results", [])
        if isinstance(results, list) and results:
            table = []
            for item in results:
                ref = item.get("reference", {})
                zero_audio = item.get("zero_audio", {})
                zero_video = item.get("zero_video", {})
                table.append({"Clip": item.get("clip_id", item.get("file", "—")),
                              "Reference fake score": ref.get("fake_score"),
                              "Audio removed": zero_audio.get("fake_score"),
                              "Video removed": zero_video.get("fake_score"),
                              "Audio logit Δ": item.get("delta_audio_logit"),
                              "Video logit Δ": item.get("delta_video_logit"),
                              "Valid tokens": item.get("valid_length")})
            st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)
        st.caption(f"Checkpoint SHA-256: `{ablation.get('checkpoint_sha256', 'not recorded')}` · Device: `{ablation.get('device', 'not recorded')}`. This ablation output uses `fake_score` and is distinct from the shared prediction JSONL contract.")
    else:
        st.info("No modality-ablation report selected or discovered.")

    st.divider()
    panel_title("2. Attention-boundary audit", "Produced by `scripts/day2_attention_audit.py` as `artifacts/day2/attention_audit.json`.")
    attention, attention_source = _load_optional_json_artifact(root, "attention_audit.json", files.get("attention_audits", []), "attention_audit")
    if attention:
        cfg = attention.get("config", {})
        cols = st.columns(4)
        cols[0].metric("Observed layers", len(attention.get("layers", [])))
        cols[1].metric("Valid source tokens", attention.get("valid_length", "—"))
        cols[2].metric("Model heads", cfg.get("nhead", "—"))
        cols[3].metric("Model max length", cfg.get("max_length", "—"))
        layer_rows = []
        for item in attention.get("layers", []):
            layer_rows.append({"Layer": item.get("layer"), "Self-attention": item.get("self_attention"),
                               "Q shape": str(item.get("Q_shape")), "K shape": str(item.get("K_shape")),
                               "V shape": str(item.get("V_shape")), "Nq": item.get("N_q"), "Nk": item.get("N_k"),
                               "Head dim": item.get("head_dim"), "Mask matches": item.get("mask_matches_original"),
                               "dtype": item.get("dtype")})
        if layer_rows:
            st.dataframe(pd.DataFrame(layer_rows), use_container_width=True, hide_index=True)
        mask = attention.get("attention_mask", {})
        st.write("**Attention mask inspection**")
        st.json(mask)
        st.caption(f"Source: {attention_source} · Checkpoint SHA-256: `{attention.get('checkpoint_sha256', 'not recorded')}` · Feature file: `{attention.get('feature_file', 'not recorded')}`")
    else:
        st.info("No attention-boundary audit selected or discovered.")

    st.divider()
    panel_title("3. Theoretical attention-memory analysis", "Produced by `scripts/dayA_memory_analysis.py` as `artifacts/dayA/attention_memory.json`.")
    memory, memory_source = _load_optional_json_artifact(root, "attention_memory.json", files.get("attention_memory_reports", []), "attention_memory")
    if memory:
        st.caption(memory.get("interpretation", "The backend's meaning note is not present."))
        st.code(memory.get("formula", "B * H * Nq * Nk * bytes_per_element"), language="text")
        records = memory.get("layers", [])
        if isinstance(records, list) and records:
            mem_df = pd.DataFrame([{"Layer": r.get("layer"), "Q shape": str(r.get("q_shape")),
                                    "K shape": str(r.get("k_shape")), "Score matrix (MiB)": r.get("score_matrix_mib"),
                                    "QKV (MiB)": r.get("qkv_mib"), "dtype": r.get("dtype")} for r in records])
            st.dataframe(mem_df, use_container_width=True, hide_index=True)
            if "Score matrix (MiB)" in mem_df and "QKV (MiB)" in mem_df:
                long = mem_df.melt(id_vars=["Layer"], value_vars=["Score matrix (MiB)", "QKV (MiB)"],
                                   var_name="Estimate", value_name="MiB")
                fig = px.bar(long, x="Layer", y="MiB", color="Estimate", barmode="group", template="plotly_dark")
                fig.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10,r=10,t=25,b=10))
                st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Source: {memory_source} · Checkpoint SHA-256: `{memory.get('checkpoint_sha256', 'not recorded')}`. Dense score size is not a measurement of SDPA's actual allocation.")
    else:
        st.info("No attention-memory report selected or discovered.")

    st.divider()
    panel_title("4. Padding and attention-mask audit", "Produced by `scripts/dayA_padding_audit.py` as `artifacts/dayA/padding_mask_audit.json`.")
    padding, padding_source = _load_optional_json_artifact(root, "padding_mask_audit.json", files.get("padding_audits", []), "padding_audit")
    if padding:
        cols = st.columns(4)
        cols[0].metric("Valid length", padding.get("valid_length", "—"))
        cols[1].metric("Maximum length", padding.get("max_length", "—"))
        cols[2].metric("Total attention tokens", padding.get("attention_tokens", "—"))
        cols[3].metric("Padding / modality", padding.get("padding_per_modality", "—"))
        st.write({"Video padded values are zero": padding.get("video_input_padding_zero"),
                  "Audio padded values are zero": padding.get("audio_input_padding_zero"),
                  "Mask semantics": padding.get("mask_semantics"),
                  "Separate key-padding mask": padding.get("separate_key_padding_mask")})
        st.write("**Permitted attention pairs by query group and key group**")
        permissions = padding.get("permitted_attention_pairs", {})
        if permissions:
            st.dataframe(pd.DataFrame(permissions).fillna(0).astype(int), use_container_width=True)
        st.caption(f"Source: {padding_source} · Checkpoint SHA-256: `{padding.get('checkpoint_sha256', 'not recorded')}` · Feature file: `{padding.get('feature_file', 'not recorded')}`")
    else:
        st.info("No padding-mask audit selected or discovered.")


def render_demo(root: Path, files: dict[str, list[Path]]) -> None:
    page_header("INFERENCE / CLIP INSPECTION", "Demo & predictions",
                "Run validation-only cached-feature inference through the real Member 3 CLI, then inspect its prediction artifacts.")
    status = inspect_backend(root)
    st.markdown("<div class='avf-warn'><b>Input boundary:</b> this runs the pretrained detector on cached encoder features, not raw video/audio. The intervention modes use floating-point fake quantization, not an INT8 integer kernel. The UI never opens the held-out test split.</div>", unsafe_allow_html=True)
    st.subheader("Run real cached-feature validation")
    if not status["real_inference_ready"]:
        st.error("The required four-mode adapter or Member 3 CLI components were not detected in this checkout.")
    else:
        st.info("This form runs the checked-in CLI on the validation split only. It validates paths/configuration first, then writes a run folder with predictions and a cached-feature benchmark.")
        manifest_options = files.get("manifests", [])
        default_manifest = str(manifest_options[0].relative_to(root)) if manifest_options and manifest_options[0].is_relative_to(root) else (str(manifest_options[0]) if manifest_options else "data/pilot/test_day2/manifest_rows.jsonl")
        with st.form("real_cached_feature_run"):
            upstream_root = st.text_input("External DiMoDif checkout (contains src/models.py and ckpt/)", placeholder="Path to your local DiMoDif repository")
            checkpoint_name = st.selectbox("Checkpoint", ["lavdf", "avdeepfake1m"], index=0)
            manifest_path = st.text_input("Whole-video manifest JSONL", value=default_manifest)
            feature_root = st.text_input("Feature root", value=".", help="For shared_pt this resolves manifest feature_path entries; for dimodif_npz it is the parent of <split>/<clip_id>/features.npz.")
            feature_format = st.selectbox("Feature format", ["shared_pt", "dimodif_npz"], index=0)
            device = st.selectbox("Execution device (no automatic fallback)", ["cpu", "cuda", "cuda:0", "mps"], index=0)
            modes = st.multiselect("Precision configurations", list(CONFIGS), default=list(CONFIGS), format_func=lambda value: CFG_LABELS.get(value, value))
            require_hash = st.checkbox("Require the agreed LAV-DF checkpoint SHA-256", value=True, disabled=checkpoint_name != "lavdf")
            run_id = st.text_input("Optional run ID", value="")
            experiment_id = st.text_input("Optional experiment ID", value="")
            confirm_inputs = st.checkbox("I have verified that this machine has the external checkpoint and local feature files; run validation only.", value=False)
            submitted = st.form_submit_button("Validate inputs and run validation experiment", disabled=not status["real_inference_ready"])
        if submitted:
            from pathlib import Path as LocalPath
            upstream = LocalPath(upstream_root).expanduser()
            manifest = LocalPath(manifest_path).expanduser()
            features = LocalPath(feature_root).expanduser()
            upstream = upstream if upstream.is_absolute() else root / upstream
            manifest = manifest if manifest.is_absolute() else root / manifest
            features = features if features.is_absolute() else root / features
            if not upstream_root.strip() or not upstream.is_dir():
                st.error("The external DiMoDif checkout directory was not found.")
            elif not manifest_path.strip() or not manifest.is_file():
                st.error("The manifest JSONL file was not found.")
            elif not feature_root.strip() or not features.is_dir():
                st.error("The feature root directory was not found.")
            elif not modes:
                st.error("Select at least one precision configuration.")
            elif not confirm_inputs:
                st.warning("Confirm the local checkpoint and feature inputs before starting the run.")
            else:
                with st.spinner("Validating configuration and running the selected validation modes..."):
                    code, output = run_cached_feature_experiment(
                        root, upstream_root=str(upstream.resolve()), checkpoint_name=checkpoint_name,
                        manifest=str(manifest.resolve()), feature_root=str(features.resolve()),
                        feature_format=feature_format, device=device, modes=modes,
                        run_id=run_id, experiment_id=experiment_id, require_lavdf_hash=require_hash)
                if code == 0:
                    st.success("Backend CLI exited successfully. Evaluate predictions with Member 2 before making research claims.")
                else:
                    st.error(f"Backend CLI exited with code {code}. Review the diagnostic output; the run may be blocked or incomplete.")
                st.code(output or "No output", language="text")
                st.caption("A successful command is not research release approval.")
    st.divider()
    st.subheader("Prediction artifacts")
    paths = files["predictions"]
    selected_paths, uploads = choose_many_local_or_upload(root, "prediction JSONL files", paths, "predictions", ["jsonl", "txt"])
    rows: list[dict[str, Any]] = []
    source = None
    if uploads:
        rows = [row for upload_file in uploads for row in parse_uploaded_jsonl_with_error(upload_file, "prediction JSONL")]
        source = ", ".join(f"{u.name} (session only)" for u in uploads)
    elif selected_paths:
        rows = [row for selected_path in selected_paths for row in load_jsonl_with_error(selected_path, "prediction JSONL")]
        source = ", ".join(str(p.relative_to(root)) if p.is_relative_to(root) else str(p) for p in selected_paths)
    if rows:
        errors = validate_predictions(rows, root / "contracts/schemas/prediction.schema.json")
        st.caption(f"Source: {source} · {len(rows):,} prediction rows")
        if errors:
            st.error(f"Prediction contract check failed with {len(errors)} issue(s).")
            with st.expander("Validation details"):
                st.code("\n".join(errors[:50]))
        else:
            st.success("Prediction rows match the current schema; labels remain outside the prediction records.")
        df = pd.DataFrame(prediction_summary(rows))
        st.dataframe(df, use_container_width=True, hide_index=True)
        numeric = [r for r in rows if isinstance(r.get("score_fake"), (int, float))]
        if numeric:
            plot_df = pd.DataFrame(numeric)
            if "config" in plot_df and "score_fake" in plot_df:
                fig = px.box(plot_df, x="config", y="score_fake", color="config", points=False,
                             category_orders={"config": list(CONFIGS)}, template="plotly_dark")
                fig.update_layout(height=340, showlegend=False, xaxis_title="Precision configuration",
                                  yaxis_title="score_fake", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
        clip_ids = sorted({str(r.get("clip_id")) for r in rows if r.get("clip_id")})
        if clip_ids:
            panel_title("Compare one clip across configurations", "A missing row stays missing; the dashboard does not infer a score for configurations that were not run.")
            chosen_clip = st.selectbox("Clip ID", clip_ids, key="prediction_clip_compare")
            clip_rows = [r for r in rows if str(r.get("clip_id")) == chosen_clip]
            score_by_config = {str(r.get("config")): r for r in clip_rows}
            clip_table = []
            for config_name in CONFIGS:
                record = score_by_config.get(config_name)
                clip_table.append({"Configuration": CFG_LABELS.get(config_name, config_name),
                                   "score_fake": record.get("score_fake") if record else None,
                                   "Split": record.get("split") if record else "not present",
                                   "Run ID": record.get("run_id") if record else "—"})
            st.dataframe(pd.DataFrame(clip_table), use_container_width=True, hide_index=True)
            st.caption("`score_fake` is the backend's fake-class score in [0, 1]. It is not re-calibrated by the frontend.")
    else:
        st.info("No prediction JSONL file selected or discovered. The backend contract is one row per clip × configuration, with `score_fake` and no ground-truth label.")
    st.divider()
    panel_title("Prediction readiness check", "Member 2's readiness command checks contract validity, checkpoint hash, complete manifest join and score direction without unsealing test labels.")
    check_path, check_upload = choose_local_or_upload(root, "prediction check report (check.json)", files.get("prediction_checks", []), "prediction_check", ["json"])
    check_report = None
    if check_upload is not None:
        try: check_report = json.loads(read_uploaded(check_upload))
        except Exception as exc: st.error(f"Could not parse prediction readiness report: {exc}")
    elif check_path:
        try: check_report = load_json(check_path)
        except Exception as exc: st.error(f"Could not read prediction check report: {exc}")
    if check_report:
        st.json(check_report)
        st.caption("A check report is an input-quality and score-orientation check; it is not an accuracy report.")
    else:
        st.caption("No prediction check JSON discovered. The relevant backend script is `scripts/m2_check_predictions.py`.")

    st.divider()
    st.subheader("Model execution readiness")
    st.write(f"**Adapter status:** {status['adapter_mode_status']}")
    st.write(f"**CLI status:** {status['cli_status']}")
    st.caption("The form above invokes the real cached-feature CLI on validation only. The test split remains governed by Member 2’s protocol/unseal/purpose checks and C9/C10 owner approvals. No raw AV clip upload is provided.")
    with st.expander("Run synthetic infrastructure smoke test", expanded=False):
        st.warning("This uses random synthetic tokens and a randomly initialized diagnostic head. Its scores are not deepfake predictions and must never enter evaluation reports.")
        if st.button("Run diagnostic smoke test", key="run_smoke"):
            with st.spinner("Running the backend's --smoke diagnostic..."):
                code, output = run_diagnostic_smoke(root)
            if code == 0:
                st.success("Smoke command exited successfully.")
            else:
                st.error(f"Smoke command exited with code {code}.")
            st.code(output or "No output", language="text")


def _metric_value(metric: Any) -> tuple[float | None, str]:
    if not isinstance(metric, dict):
        return None, ""
    value = metric.get("value")
    lo, hi = metric.get("ci_low"), metric.get("ci_high")
    if isinstance(value, (int, float)):
        ci = f" [{lo:.4f}, {hi:.4f}]" if isinstance(lo, (int,float)) and isinstance(hi, (int,float)) else ""
        return float(value), ci
    return None, ""


def render_evaluation(root: Path, files: dict[str, list[Path]]) -> None:
    page_header("FORENSIC EVALUATION / MEMBER 2", "Evaluation results",
                "Overall and manipulation-stratified metrics, with validation-frozen thresholds and source-cluster confidence intervals.")
    path, upload = choose_local_or_upload(root, "metrics.json", files["metrics"], "metrics", ["json"])
    report = None
    source = None
    if upload:
        try:
            report = json.loads(read_uploaded(upload))
            source = f"uploaded: {upload.name} (session only)"
        except Exception as exc:
            st.error(f"Could not parse metrics JSON: {exc}")
    elif path:
        try:
            report = load_json(path)
            source = str(path)
        except Exception as exc:
            st.error(f"Could not read metrics report: {type(exc).__name__}: {exc}")
    # A committed held-out summary is documentation, not a freshly loaded metrics.json artifact.
    test_doc = root / "docs/member2/results/TEST_RESULTS.md"
    if test_doc.is_file():
        text = test_doc.read_text(encoding="utf-8", errors="replace")
        documented = documented_test_results(text)
        if documented["rows"]:
            meta = documented["metadata"]
            st.info("Showing Member 2's committed held-out test summary. These are documented results, not a newly computed evaluation or a verified live prediction import.")
            st.caption(f"Documentation source: {test_doc.relative_to(root)} · Reported prediction run: {meta.get('run_id', 'not stated')}")
            if meta.get("total_test_videos"):
                st.write(f"**Reported test cohort:** {meta['total_test_videos']:,} videos · **Source groups:** {meta.get('source_groups', 'not stated'):,}")
            if meta.get("threshold") is not None:
                st.write(f"**Validation-frozen threshold:** {meta['threshold']:.6f} · **Bootstrap:** {meta.get('bootstrap_resamples', 'not stated')} resamples, seed {meta.get('bootstrap_seed', 'not stated')}")
            table = []
            for row in documented["rows"]:
                table.append({
                    "Configuration": CFG_LABELS.get(row["config"], row["config"]),
                    "AUROC": f"{row['auroc']:.4f} [{row['auroc_ci_low']:.4f}, {row['auroc_ci_high']:.4f}]",
                    "AUPRC": f"{row['auprc']:.4f}", "BCE": f"{row['bce']:.4f}",
                    "Recall": f"{row['recall']:.3f}", "Real FPR": f"{row['real_fpr']:.3f}",
                })
            panel_title("Published held-out results", "Transcribed from Member 2's checked-in report; AUROC confidence intervals are shown as published.")
            st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)
            per_type = documented.get("per_type", [])
            if per_type:
                panel_title("Reference performance by manipulation type")
                st.dataframe(pd.DataFrame([{
                    "Manipulation": row["manipulation_type"].replace("_", " "),
                    "AUROC": row["auroc"], "Recall": row["recall"],
                } for row in per_type]), use_container_width=True, hide_index=True)
            paired = documented.get("paired_bce", [])
            if paired:
                panel_title("Paired BCE contrasts", "Intervals crossing zero are inconclusive; they do not establish equivalence.")
                st.dataframe(pd.DataFrame([{
                    "Contrast": row["contrast"], "Delta BCE": row["delta_bce"],
                    "95% CI": f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]",
                    "CI excludes zero": row["ci_low"] > 0 or row["ci_high"] < 0,
                } for row in paired]), use_container_width=True, hide_index=True)
            st.warning("Report limitations: simulated INT8 uses floating-point arithmetic; no hardware speedup is established. Source groups do not prove speaker independence, duration was not controlled, and test-label-derived summaries were inspected before freeze. The report records three inference attempts, including a failed pre-scoring attempt and a stopped attempt.")
            with st.expander("Read the complete Member 2 test report"):
                st.markdown(text)
            st.caption("For machine-readable paired metrics, import the corresponding structured metrics JSON when available. This documentation view does not synthesize missing confidence intervals or prediction files.")
    if report:
        errors = validate_metrics_report(report)
        split = report.get("split")
        st.caption(f"Source: {source} · split: {split}")
        if errors:
            st.error("Metrics report structure needs review: " + "; ".join(errors[:8]))
        configs = report.get("configs", {})
        if not isinstance(configs, dict):
            st.error("Cannot display this report because `configs` is not an object.")
            st.json(report)
            return
        threshold = report.get("threshold")
        st.write("")
        available_configs = [cfg for cfg in CONFIGS if cfg in configs]
        selected_config = st.selectbox("Inspect configuration", available_configs, format_func=lambda x: CFG_LABELS.get(x, x), key="evaluation_config") if available_configs else None
        selected_block = configs.get(selected_config, {}) if selected_config else {}
        c = st.columns(4)
        for col, metric_name, label in zip(c, ("auroc", "auprc", "recall", "bce"), ("AUROC", "AUPRC", "Recall", "BCE loss")):
            value, ci = _metric_value(selected_block.get(metric_name))
            with col:
                col.metric(label, f"{value:.4f}" if value is not None else "Not available", help=f"{ci.strip()} · {CFG_LABELS.get(selected_config, selected_config)}" if value is not None else "Not present in this report")
        if isinstance(threshold, (int, float)):
            st.info(f"Frozen operating threshold: **{threshold:.6f}** · calibration is expected to come from validation (reference) at the project's target real false-positive rate.")
        if split == "test":
            accesses = report.get("test_accesses_so_far", "not recorded")
            st.warning(f"Held-out test report. Test access count recorded by this report: {accesses}. Do not use test results to tune the model, threshold or precision policy.")
        metric_rows = []
        for cfg, block in configs.items():
            row = {"Configuration": CFG_LABELS.get(cfg, cfg)}
            for m in ("auroc", "auprc", "recall", "real_fpr", "bce"):
                value, ci = _metric_value(block.get(m))
                row[m.upper() if m != "real_fpr" else "Real FPR"] = f"{value:.4f}{ci}" if value is not None else "—"
            row["Clips"] = block.get("n")
            row["Source clusters"] = block.get("n_clusters")
            metric_rows.append(row)
        panel_title("Overall results", "Each CI is a source/identity-cluster bootstrap interval, not an independent-clip interval.")
        st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)
        if path is not None:
            report_dir = path.parent
            figure_cols = st.columns(2)
            for col, figure_name, caption in zip(figure_cols, ("curves.png", "recall_by_type.png"),
                                                 ("Backend-generated ROC and precision-recall curves", "Backend-generated recall by manipulation type with cluster-bootstrap intervals")):
                figure_path = report_dir / figure_name
                if figure_path.is_file():
                    with col:
                        st.image(str(figure_path), caption=caption, use_column_width=True)
            summary_path = report_dir / "summary.md"
            if summary_path.is_file():
                with st.expander("Read backend-generated summary.md"):
                    st.markdown(summary_path.read_text(encoding="utf-8", errors="replace"))
        chart_rows = []
        for cfg, block in configs.items():
            for metric in ("auroc", "auprc", "recall"):
                value, _ = _metric_value(block.get(metric))
                if value is not None:
                    chart_rows.append({"Configuration": CFG_LABELS.get(cfg,cfg), "Metric": metric.upper(), "Value": value})
        if chart_rows:
            fig = px.bar(pd.DataFrame(chart_rows), x="Metric", y="Value", color="Configuration", barmode="group", range_y=[0,1], template="plotly_dark")
            fig.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10,r=10,t=25,b=10))
            st.plotly_chart(fig, use_container_width=True)
        panel_title("Manipulation-stratified metrics", "The evaluator compares each fake type against all real clips in the same split.")
        strata = []
        for cfg, block in configs.items():
            for typ, stats in block.get("per_type", {}).items():
                row = {"Configuration": cfg, "Attack type": typ.replace("_", " "), "Fake clips": stats.get("n_fake"), "Real clips": stats.get("n_real")}
                for m in ("auroc", "auprc", "recall"):
                    v, ci = _metric_value(stats.get(m))
                    row[m.upper()] = f"{v:.4f}{ci}" if v is not None else "—"
                strata.append(row)
        if strata:
            st.dataframe(pd.DataFrame(strata), use_container_width=True, hide_index=True)
        paired = report.get("paired", [])
        if isinstance(paired, list) and paired:
            panel_title("Paired configuration comparisons", "`delta` is b − a; a result is marked statistically distinguishable only if the paired 95% CI excludes zero.")
            p_rows = []
            for comparison in paired:
                for metric, block in comparison.get("metrics", {}).items():
                    lo, hi = block.get("ci_low"), block.get("ci_high")
                    p_rows.append({"Comparison": f"{comparison.get('b')} − {comparison.get('a')}", "Metric": metric,
                                   "Delta": block.get("delta"), "95% CI": f"[{lo:.4f}, {hi:.4f}]" if isinstance(lo,(int,float)) and isinstance(hi,(int,float)) else "—",
                                   "CI excludes 0": bool(block.get("excludes_zero"))})
            st.dataframe(pd.DataFrame(p_rows), use_container_width=True, hide_index=True)
        st.caption(f"Bootstrap resamples: {report.get('n_boot', 'not recorded')} · seed: {report.get('seed', 'not recorded')} · manifest SHA-256: {report.get('manifest_sha256', 'not recorded')}")
        return
    # Structured metrics may not have been committed. Show the repo's documented validation snapshot as documentation, not as a fresh artifact.
    doc = root / "docs/member2/results/VAL_4MODE_RESULTS.md"
    if doc.is_file():
        text = doc.read_text(encoding="utf-8", errors="replace")
        parsed = documented_validation_table(text)
        if parsed:
            st.info("No structured `metrics.json` was found. The table below is parsed from the repository's documented validation summary; it is not a newly loaded machine-readable report and it is not the held-out test result.")
            st.caption(f"Documentation source: {doc.relative_to(root)}")
            st.dataframe(pd.DataFrame(parsed), use_container_width=True, hide_index=True)
            st.markdown("**Interpretation recorded in the source document:** all four validation configurations have the same rounded AUROC/AUPRC/recall values; paired BCE confidence intervals cross zero. This supports a validation-set robustness observation, not a claim that targeted restoration rescues degradation. Test-set results are still needed for the final claim.")
            with st.expander("Read validation documentation"):
                st.markdown(text)
            return
    st.info("No structured metrics.json found. Import one or generate it using Member 2's `scripts/m2_evaluate.py evaluate` workflow after valid predictions and a frozen protocol exist.")
    st.caption("Metrics JSON contract: split, threshold, test_accesses_so_far, n_boot, seed, manifest_sha256, per-config overall/per-type blocks, and paired comparisons.")


def render_benchmarks(root: Path, files: dict[str, list[Path]]) -> None:
    page_header("RUNTIME / MEMBER 3", "Benchmark explorer",
                "Read the Member 3 benchmark records without conflating operator, cached-feature model, or preprocessing costs.")
    st.markdown("<div class='avf-warn'><b>Precision disclosure:</b> all non-reference configurations are simulated fake quantization in floating-point arithmetic. The runtime records do not measure a real INT8 kernel or prove INT8 hardware speedup.</div>", unsafe_allow_html=True)

    with st.expander("Documented benchmark snapshot (optional report upload)"):
        report = st.file_uploader(
            "Upload a Markdown benchmark report",
            type=["md", "markdown", "txt"],
            key="documented_benchmark_report",
        )
        if report is not None:
            try:
                snapshot = documented_benchmark_summary(report.getvalue().decode("utf-8"))
            except UnicodeDecodeError:
                snapshot = None
                st.error("The report must be UTF-8 text.")
            if snapshot is None:
                st.info("No recognized typical-workload p50 summary was found. Upload the report containing the 'Verified typical p50 ms' summary.")
            else:
                st.caption("Documented snapshot only · not a live benchmark run or a validated benchmark-record artifact.")
                snapshot_df = pd.DataFrame(
                    [{"Configuration": config, "p50 (ms)": latency}
                     for config, latency in snapshot["p50_ms"].items()]
                )
                st.dataframe(snapshot_df, use_container_width=True, hide_index=True)
                st.json({"scope": snapshot["scope"], "device": snapshot["device"],
                         "backend": snapshot["backend"], "extra_cost_ms": snapshot["extra_cost_ms"]})
    selected_paths, uploads = choose_many_local_or_upload(root, "benchmark JSONL files", files["benchmarks"], "benchmarks", ["jsonl", "txt"])
    rows: list[dict[str, Any]] = []
    source = None
    if uploads:
        rows = [row for upload_file in uploads for row in parse_uploaded_jsonl_with_error(upload_file, "benchmark JSONL")]
        source = ", ".join(f"{u.name} (session only)" for u in uploads)
    elif selected_paths:
        rows = [row for selected_path in selected_paths for row in load_jsonl_with_error(selected_path, "benchmark JSONL")]
        source = ", ".join(str(p.relative_to(root)) if p.is_relative_to(root) else str(p) for p in selected_paths)
    if not rows:
        st.info("No benchmark JSONL selected or discovered. The integrated Member 3 CLI can benchmark cached-feature model execution and writes benchmark_records.jsonl inside its run folder; no local run output has been found yet.")
        st.markdown("**Expected scopes**")
        st.markdown("- `operator`: isolated fusion attention only\n- `cached_feature_model`: cached-feature fusion + head\n- `preprocessing_once`: raw-media preprocessing, reported separately")
        st.caption("CPU and MPS peak memory are represented as null / not measured by the backend; do not plot them as zero.")
        return
    schema = root / "contracts/schemas/benchmark.schema.json"
    errors = validate_benchmarks(rows, schema)
    st.caption(f"Source: {source} · {len(rows)} benchmark record(s). Local discovery includes both `artifacts/benchmarks/*.jsonl` and finalized run bundles' `benchmark_records.jsonl`.")
    st.warning("The checked-in benchmark schema is documented as proposed v0.2 and pending cross-member review. Validation below means schema-conforming to the current file, not that the contract has been finally approved.")
    if errors:
        st.error(f"{len(errors)} record validation issue(s).")
        with st.expander("Validation details"):
            st.code("\n".join(errors[:50]))
        st.info("The benchmark explorer stops here for nonconforming rows so it does not silently chart malformed fields. Correct the JSONL against the checked-in schema and re-import it.")
        return
    else:
        st.success("All records validate against the checked-in benchmark schema v0.2 proposal.")
    df = pd.DataFrame(rows)
    if df.empty:
        return
    selectors = st.columns(3)
    with selectors[0]:
        scopes = sorted(df["scope"].dropna().astype(str).unique()) if "scope" in df else []
        scope = st.selectbox("Measurement scope", ["All", *scopes])
    with selectors[1]:
        configs = sorted(df["config"].dropna().astype(str).unique()) if "config" in df else []
        config = st.selectbox("Configuration", ["All", *configs])
    with selectors[2]:
        devices = sorted(df["device"].dropna().astype(str).unique()) if "device" in df else []
        device = st.selectbox("Device", ["All", *devices])
    view = df.copy()
    if scope != "All": view = view[view["scope"] == scope]
    if config != "All": view = view[view["config"] == config]
    if device != "All": view = view[view["device"] == device]
    for col in ("p50_ms", "p95_ms", "mean_ms", "peak_memory_bytes", "conversion_overhead_ms"):
        if col in view:
            view[col] = pd.to_numeric(view[col], errors="coerce")
    p50_values = view.get("p50_ms", pd.Series(dtype=float)).dropna()
    p95_values = view.get("p95_ms", pd.Series(dtype=float)).dropna()
    mem_values = view.get("peak_memory_bytes", pd.Series(dtype=float)).dropna()
    c = st.columns(3)
    c[0].metric("Series", f"{len(view):,}")
    c[1].metric("Median p50", f"{p50_values.median():.3f} ms" if len(p50_values) else "Not available")
    c[2].metric("Peak memory", human_bytes(mem_values.max()) if len(mem_values) else "Not measured")
    if len(view) and {"config", "p50_ms", "p95_ms"}.issubset(view.columns):
        chart_df = view.melt(id_vars=[c for c in ("config","scope","device") if c in view], value_vars=[c for c in ("p50_ms","p95_ms") if c in view], var_name="Latency statistic", value_name="Latency (ms)")
        fig = px.bar(chart_df, x="config", y="Latency (ms)", color="Latency statistic", barmode="group", facet_col="scope" if "scope" in view and view["scope"].nunique()>1 else None, template="plotly_dark")
        fig.update_layout(height=390, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10,r=10,t=35,b=10))
        st.plotly_chart(fig, use_container_width=True)
    display_cols = [c for c in ("run_id","config","scope","device","backend","batch_size","warmup","repeats","p50_ms","p95_ms","peak_memory_bytes","memory_method","conversion_overhead_ms","conversion_overhead_included_in_latency") if c in view]
    st.dataframe(view[display_cols], use_container_width=True, hide_index=True)
    if "workload" in view:
        with st.expander("Input shapes and precision disclosure"):
            for idx, record in view.iterrows():
                st.markdown(f"**{record.get('config')} · {record.get('scope')} · {record.get('device')}**")
                st.json({"workload": record.get("workload"), "precision": record.get("precision"), "measurement": record.get("measurement")})
    st.caption("`peak_memory_bytes = null` means unavailable / not measured. The naive dense score-matrix estimate is theoretical and is not the memory allocation of fused SDPA.")


def render_runs(root: Path, files: dict[str, list[Path]]) -> None:
    page_header("REPRODUCIBILITY / MEMBER 3", "Runs & system",
                "Inspect environment diagnostics, saved configuration snapshots, hashes and artifact-integrity checks.")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Environment diagnostics")
        reports = files["environment_reports"]
        selected = st.selectbox("Environment report", ["— none —", *[str(p.relative_to(root)) for p in reports]], key="env_select")
        report_path = next((p for p in reports if str(p.relative_to(root)) == selected), None)
        if report_path:
            try: st.json(load_json(report_path))
            except Exception as exc: st.error(f"Could not read environment report: {exc}")
        else:
            st.caption("No environment_report.json found in artifacts/runs.")
        if st.button("Run local environment check", use_container_width=True):
            with st.spinner("Checking Python, OS, CPU/RAM, PyTorch, available devices and tiny dtype/SDPA operations..."):
                code, output, out_path = run_environment_check(root)
            if code == 0:
                st.success(f"Environment report written: {out_path.relative_to(root)}")
                if out_path.exists():
                    try: st.json(load_json(out_path))
                    except Exception as exc: st.error(f"Environment report was written but could not be parsed: {exc}")
            else:
                st.error(output or f"Environment check exited with code {code}.")
            if output:
                with st.expander("Command output"):
                    st.code(output)
    with col2:
        st.subheader("Run integrity")
        run_dirs = files["run_dirs"]
        chosen = st.selectbox("Run folder", ["— none —", *[str(p.relative_to(root)) for p in run_dirs]], key="run_select")
        run_dir = next((p for p in run_dirs if str(p.relative_to(root)) == chosen), None)
        if run_dir:
            manifest_path = run_dir / "run_manifest.json"
            if manifest_path.is_file():
                try:
                    run_manifest = load_json(manifest_path)
                except Exception as exc:
                    st.error(f"Could not parse run manifest: {exc}")
                    run_manifest = {}
                st.write(f"**Run ID:** {run_manifest.get('run_id', run_dir.name)}")
                st.write(f"**Artifacts listed:** {len(run_manifest.get('artifacts', []))}")
                try:
                    import sys
                    src = str(root / "src")
                    if src not in sys.path: sys.path.insert(0, src)
                    from avfuse.member3_runtime.artifacts import verify_run_integrity
                    verdict = verify_run_integrity(run_dir)
                    if verdict.get("ok"):
                        st.success("Run artifact integrity check passed.")
                    else:
                        st.error("Run artifact integrity check found issues.")
                    st.json(verdict)
                except Exception as exc:
                    st.warning(f"Could not execute the backend integrity verifier: {type(exc).__name__}: {exc}")
                with st.expander("Run manifest"):
                    st.json(run_manifest)
                for filename, title in (("resolved_config.json", "Resolved experiment configuration"),
                                        ("provenance.json", "Run provenance and hashes")):
                    artifact_path = run_dir / filename
                    if artifact_path.is_file():
                        with st.expander(title):
                            st.json(load_json(artifact_path))
            else:
                st.warning("Selected folder has no run_manifest.json; it is not a finalized Member 3 run bundle.")
        else:
            st.caption("No finalized run folders found under artifacts/runs/run-*.")
    st.divider()
    panel_title("Run bundle contents", "Expected files created by Member 3's RunContext when a run is finalized.")
    st.code("""artifacts/runs/<run_id>/
├── resolved_config.json
├── environment_report.json
├── provenance.json
├── benchmark_records.jsonl
├── stage_profile.json             # optional
├── run_manifest.json
└── integrity/
    └── artifact_hashes.json""", language="text")
    st.caption("The run identifier is unique, not a content hash. Integrity hashes detect accidental changes; they are not tamper-proof against someone rewriting the manifest and its anchor.")
    st.divider()
    panel_title("Pipeline stage profile", "Stage profiles preserve separate one-time preprocessing and cached-model timings; missing stages are not treated as zero and stage medians are never summed into end-to-end latency.")
    stage_profiles = files.get("stage_profiles", [])
    selected_stage = st.selectbox("Stage profile artifact", ["— none —", *[str(p.relative_to(root)) for p in stage_profiles]], key="stage_profile_select")
    stage_path = next((p for p in stage_profiles if str(p.relative_to(root)) == selected_stage), None)
    if stage_path:
        try:
            profile = load_json(stage_path)
        except Exception as exc:
            st.error(f"Could not parse stage profile: {exc}")
            profile = {}
        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("Profile evidence", str(profile.get("evidence_status", "not stated")))
        pc2.metric("Complete required stages", "Yes" if profile.get("complete") is True else "No")
        pc3.metric("End-to-end measured", "Yes" if profile.get("end_to_end_measured") is True else "No")
        if profile.get("end_to_end_measured") is False:
            st.info("The backend explicitly states that independently measured stage timings are not an end-to-end latency and must not be summed.")
        stages = profile.get("stages", [])
        if isinstance(stages, list) and stages:
            stage_rows = []
            for stage in stages:
                row = {k: stage.get(k) for k in ("stage", "scope_group", "stage_kind", "status", "input_kind", "p50_ms", "p95_ms", "measurement_origin") if k in stage}
                if row: stage_rows.append(row)
            if stage_rows:
                st.dataframe(pd.DataFrame(stage_rows), use_container_width=True, hide_index=True)
        st.json(profile)
    else:
        st.caption("No finalized run stage_profile.json was discovered. The profiler's in-memory output does not automatically become a report unless a caller writes it through RunContext.")


def render_import_help(root: Path) -> None:
    page_header("LOCAL FILE WORKFLOW", "Artifact import & file handling",
                "The dashboard reads research artifacts from the local repository or from files you explicitly import into the browser session.")
    st.markdown("<div class='avf-note'><b>Privacy by design:</b> imported JSON/JSONL files are parsed in the current Streamlit session. The dashboard does not upload them to a remote service or copy them into Git. Move the canonical files into the repository's documented data/ or artifacts/ folders if you want automatic discovery on the next launch.</div>", unsafe_allow_html=True)
    st.subheader("Supported file types")
    st.dataframe(pd.DataFrame([
        {"File": "manifest*.jsonl", "Purpose": "Clip IDs, dataset, source/identity groups, split, label, manipulation type and local feature path"},
        {"File": "predictions*.jsonl", "Purpose": "One score_fake row per clip × precision configuration; labels remain outside this file"},
        {"File": "metrics.json", "Purpose": "Member 2 overall/per-type metrics, cluster-bootstrap CIs and paired comparisons"},
        {"File": "benchmark*.jsonl", "Purpose": "Member 3 schema-v0.2 benchmark records (proposed contract)"},
        {"File": "environment_report.json", "Purpose": "Python/OS/CPU/RAM/PyTorch/device capability diagnostics; not performance results"},
        {"File": "shortcut_audit.json", "Purpose": "Member 2 nuisance-feature shortcut AUROC flags by split and attack type"},
        {"File": "shortcut_control.json", "Purpose": "Detector AUROC on all videos versus a feature-matched subset for a flagged shortcut"},
        {"File": "train_overlap.json / check_*.json", "Purpose": "Checkpoint/source overlap and prediction readiness checks"},
        {"File": "ablation_report.json", "Purpose": "Member 1 reference / zero-audio / zero-video modality-ablation outputs"},
        {"File": "attention_audit.json", "Purpose": "Observed Q/K/V shapes, sequence lengths, masks and per-layer attention boundary"},
        {"File": "attention_memory.json / padding_mask_audit.json", "Purpose": "Theoretical dense-score/QKV sizes and permitted attention pairs around padding"},
        {"File": "stage_profile.json", "Purpose": "Independently measured preprocessing/model stage timings and explicit missing-stage statuses"},
        {"File": "run_manifest.json + provenance.json", "Purpose": "Run contents, hashes, environment and integrity metadata"},
    ]), use_container_width=True, hide_index=True)
    st.caption("You can select multiple prediction and benchmark JSONL files together because the backend may emit one file per configuration or run scope. Single-report JSON artifacts are imported one at a time.")
    st.subheader("Where the backend expects files")
    st.code("""data/manifests/*.jsonl
artifacts/predictions/*.jsonl
artifacts/benchmarks/*.jsonl
artifacts/reports/<split>/metrics.json
artifacts/runs/<run_id>/*""", language="text")
    st.subheader("Files deliberately not accepted as experiment inputs")
    st.markdown("- Raw video/audio files: no upload-to-inference route exists in the checked-in backend.\n- Arbitrary PyTorch checkpoints: the dashboard does not load or execute user-supplied model weights.\n- Test labels or labels embedded in prediction rows: prediction contract explicitly excludes ground truth.\n- Synthetic smoke scores: never treated as accuracy, AUROC, or research evidence.\n- Uploaded `.pt` / `.npz` feature files: metadata inspection only; no inference, conversion or persistent copy is performed.")
    st.info(f"Repository root detected: `{root}`. Set `AVFUSE_REPO_ROOT` before launch or edit the root in the sidebar if the code is stored elsewhere.")
