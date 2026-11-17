from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from .data_access import (
    CONFIGS, documented_benchmark_summary, dimodif_npz_inspection_path,
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
    .stApp { background:#fff; color:#20202a; }
    [data-testid="stSidebar"] { background:#f5f2fc; border-right:1px solid #e8e2f2; }
    [data-testid="stHeader"] { background:rgba(255,255,255,.95); }
    .block-container { max-width:1450px; padding-top:2.6rem; padding-bottom:3rem; }
    h1 { font-size:1.9rem !important; letter-spacing:-.035em; }
    h2 { font-size:1.35rem !important; }
    h3 { font-size:1.1rem !important; }
    .topic { color:#6250a5; background:#f3effb; border-radius:20px; padding:6px 12px; font-size:.8rem; }
    .journey { border-bottom:1px solid #e8e5ed; padding:12px 0 22px; margin-bottom:24px; color:#77717f; }
    .journey b { color:#6554c0; }
    [data-testid="stSidebar"] [role="radiogroup"] label { border-radius:8px; padding:10px; }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) { background:#e7dff8; color:#5643a0; }
    [data-testid="stVerticalBlockBorderWrapper"] > div { border-color:#e4e1e9; border-radius:12px; }
    .avf-subtitle,.avf-small { color:#686370; }
    .avf-note,.avf-warn,.avf-danger { border:1px solid #e4e1e9; padding:12px; border-radius:10px; }
    .stButton>button { border-radius:8px; min-height:2.6rem; }
    hr { border-color:#e8e5ed; }
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
    if len(upload.getvalue()) > 32 * 1024 * 1024:
        raise ValueError("Upload exceeds 32 MiB limit")
    return upload.getvalue().decode("utf-8-sig")


def load_jsonl_with_error(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        return load_jsonl(path)
    except Exception as exc:
        st.session_state["artifact_import_failed"] = True
        st.error(f"Could not read {label} `{path.name}`: {type(exc).__name__}: {exc}")
        return []


def parse_uploaded_jsonl_with_error(upload: Any, label: str) -> list[dict[str, Any]]:
    try:
        return parse_jsonl_text(read_uploaded(upload))
    except Exception as exc:
        st.session_state["artifact_import_failed"] = True
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
    with st.expander("Separate readiness states"):
        st.json({key: status[key] for key in ("source_available", "interfaces_imported", "software_checked", "assets_available", "execution_authorized", "run_completed", "artifact_integrity_verified")})
    st.info("Accepted study completed: overall precision damage and V/output superiority were not statistically established. This dashboard does not automatically reproduce the study.")
    last = st.session_state.get("last_run_result")
    if last:
        st.json(last)
    verified = st.session_state.get("verified_runs", {})
    st.caption(f"Past integrity checks in this session: {len(verified)}. These are dated observations, not continuous verification; inspect a run to check for subsequent file changes.")
    git = git_info(root)
    st.markdown("<div class='avf-warn'><b>Research integrity:</b> this dashboard never invents metrics. Empty states mean the corresponding artifact is not present locally. Simulated INT8 is floating-point fake quantization, not an INT8 kernel.</div>", unsafe_allow_html=True)
    st.write("")
    manifest_count = len(files["manifests"])
    prediction_count = len(files["predictions"])
    benchmark_count = len(files["benchmarks"])
    metrics_count = len(files["metrics"])
    run_count = sum((path / "run_manifest.json").is_file() for path in files["run_dirs"])
    cols = st.columns(5)
    metrics = [
        ("Manifests", str(manifest_count), "local JSONL files"),
        ("Prediction files", str(prediction_count), "local JSONL files"),
        ("Benchmark files", str(benchmark_count), "local JSONL files"),
        ("Evaluation reports", str(metrics_count), "structured metrics.json"),
        ("Run manifests", str(run_count), "presence only; integrity unverified"),
    ]
    for col, (label, value, foot) in zip(cols, metrics):
        with col:
            card(label, value, foot)
    st.caption(
        f"Reproducibility inventory: {len(files.get('environment_reports', []))} environment report(s) · "
        f"{len(files.get('stage_profiles', []))} stage profile(s). Run integrity is checked on the Runs & system page."
    )
    st.write("")
    left, right = st.columns([1.15, 1])
    with left:
        panel_title("Backend readiness", "Detected from the checked-in source tree, not from assumptions in the proposal.")
        rows = [
            ("Python package / shared contracts", (root / "pyproject.toml").is_file() and (root / "contracts/schemas").is_dir(), "Package and JSON schemas found"),
            ("Four-mode adapter interface", status["adapter_interface_available"], status["adapter_mode_status"]),
            ("Cached-feature CLI interface", status["execution_interface_available"], status["cli_status"]),
            ("Local prediction outputs", prediction_count > 0, f"{prediction_count} files discovered"),
            ("Structured benchmark outputs", benchmark_count > 0, f"{benchmark_count} files discovered"),
        ]
        for label, ok, detail in rows:
            icon = "✅" if ok else "⏳"
            st.markdown(f"**{icon} {label}**  \n<span class='avf-small'>{detail}</span>", unsafe_allow_html=True)
            st.write("")
    with right:
        panel_title("Repository snapshot")
        dirty = git.get("dirty")
        dirty_display = "yes" if dirty is True else ("no" if dirty is False else "unknown")
        st.markdown(f"<div class='avf-card'><div class='avf-card-label'>Repository</div><div class='avf-card-value' style='font-size:1.12rem'>AFML-project</div><div class='avf-card-foot'>SudarshanG9/AFML-project</div><hr><div class='avf-card-label'>Branch / commit</div><div style='color:#e2e8f0;font-weight:650;margin-top:5px'>{git.get('branch') or 'not available'} · {git.get('commit') or 'not available'}</div><div class='avf-card-foot'>Working tree dirty: {dirty_display}</div></div>", unsafe_allow_html=True)
        st.write("")
        st.info("The application reads local files under data/ and artifacts/. Uploaded files are received by the Streamlit server; the app does not automatically publish them to GitHub or third-party services.")
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
    st.session_state["artifact_import_failed"] = False
    st.info("Accepted whole-video study: validation 1,200; test 2,000. These are recorded cohort sizes, not a claim of local asset availability. Source disjointness does not establish speaker disjointness.")
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
                fig = px.bar(long, x="Split", y="Clips", color="Manipulation type", barmode="group", template="plotly_white")
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
        default_npz_root = "."
        if path is not None and path.is_relative_to(root):
            default_npz_root = str(path.parent.relative_to(root))
        feature_layout = st.selectbox(
            "Feature layout for inspection",
            ["shared_pt", "dimodif_npz"],
            format_func=lambda value: "Shared token cache (.pt)" if value == "shared_pt" else "DiMoDif NPZ (<split>/<clip_id>/features.npz)",
            key="feature_inspection_layout",
            help="Choose the same explicit format used by the Member 3 loader; the layout is never guessed from the filename.",
        )
        feature_path: Path | None = None
        feature_path_error: str | None = None
        if feature_layout == "dimodif_npz":
            npz_feature_root = st.text_input(
                "DiMoDif feature root", value=default_npz_root, key="npz_feature_inspection_root",
                help="The loader looks under <root>/<split>/<clip_id>/features.npz. External roots must be authorized by the server operator through AVFUSE_ASSET_ROOTS.",
            )
            try:
                feature_path = dimodif_npz_inspection_path(
                    root, npz_feature_root, selected_row.get("split"), selected_row.get("clip_id")
                )
            except (OSError, ValueError) as exc:
                feature_path_error = str(exc)
        else:
            raw_feature_path = selected_row.get("feature_path")
            if raw_feature_path:
                feature_path = Path(str(raw_feature_path)).expanduser()
                if not feature_path.is_absolute():
                    feature_path = root / feature_path
                try:
                    feature_path = feature_path.resolve()
                except OSError as exc:
                    feature_path_error = str(exc)
            else:
                feature_path_error = "The selected manifest row does not provide a feature_path."
        feature_inside_repo = False
        if feature_path is not None:
            try:
                from .data_access import authorized_path
                authorized_path(root, feature_path)
                feature_inside_repo = True
            except (OSError, ValueError):
                feature_path_error = "The feature path could not be resolved safely."
        uploaded_feature = st.file_uploader("Optionally import one feature cache for metadata inspection", type=["npz", "pt"], key="feature_metadata_upload")
        if uploaded_feature is not None:
            if st.button("Inspect uploaded feature metadata", key="inspect_uploaded_feature"):
                try:
                    info = feature_file_summary_bytes(uploaded_feature.getvalue(), uploaded_feature.name)
                    st.json(info)
                except Exception as exc:
                    st.error(f"Feature inspection failed: {type(exc).__name__}: {exc}")
        elif selected_row.get("split") == "test":
            st.warning("Test cache inspection is disabled; only saved aggregate test results are displayed.")
        elif feature_path is not None and feature_inside_repo and feature_path.is_file():
            st.caption(f"Feature path: {feature_path}")
            if st.button("Inspect cached feature shapes", key="inspect_feature"):
                try:
                    from .data_access import feature_file_summary
                    info = feature_file_summary(feature_path)
                    st.json(info)
                except Exception as exc:
                    st.error(f"Feature inspection failed: {type(exc).__name__}: {exc}")
        elif feature_path is not None and not feature_inside_repo:
            st.warning("The feature path is outside the server-authorized roots; inspection is blocked.")
        elif feature_path_error:
            st.warning(feature_path_error)
        else:
            st.warning(f"Feature file is not present at the selected layout path: {feature_path}")
        st.caption("For dimodif_npz, the explicit loader reads audio_features and video_features and exposes them to the adapter as audio_tokens and video_tokens (FP32). The shared .pt format expects audio_tokens and video_tokens directly. The NPZ loader uses the selected feature root and does not use the manifest's planned .pt feature_path.")
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
    st.caption("Five joint self-attention layers. Dense score-matrix size is theoretical, not measured peak memory. Modality ablations diagnose input dependence; they do not isolate causal cross-modal dependence.")
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
            st.caption("Q, K and V shapes use [batch, heads, sequence length, head dimension]. For example, [1, 4, 1202, 16] means one sequence, four heads, 1,202 positions and 16 values per head.")
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
                fig = px.bar(long, x="Layer", y="MiB", color="Estimate", barmode="group", template="plotly_white")
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
    st.session_state["artifact_import_failed"] = False
    page_header("INFERENCE / CLIP INSPECTION", "Demo & predictions",
                "Run validation-only cached-feature inference through the real Member 3 CLI, then inspect its prediction artifacts.")
    status = inspect_backend(root)
    st.markdown("<div class='avf-warn'><b>Input boundary:</b> this runs the pretrained detector on cached encoder features, not raw video/audio. The intervention modes use floating-point fake quantization, not an INT8 integer kernel. The UI never opens the held-out test split.</div>", unsafe_allow_html=True)
    st.subheader("Run real cached-feature validation")
    st.caption("Server operator must authorize external roots with AVFUSE_ASSET_ROOTS before launch.")
    if not status["execution_interface_available"]:
        st.error("The required four-mode adapter or Member 3 CLI components were not detected in this checkout.")
    else:
        st.info("This form runs the checked-in CLI on the validation split only. It validates paths/configuration first, then writes a run folder with predictions and a cached-feature benchmark.")
        default_manifest = ""
        with st.form("real_cached_feature_run"):
            upstream_root = st.text_input("External DiMoDif checkout (contains src/models.py and ckpt/)", placeholder="Path to your local DiMoDif repository")
            checkpoint_name = st.selectbox("Checkpoint", ["lavdf"], index=0)
            manifest_path = st.text_input("Whole-video manifest JSONL", value=default_manifest)
            feature_root = st.text_input("Feature root", value=".", help="For shared_pt this resolves manifest feature_path entries; for dimodif_npz it is the parent of <split>/<clip_id>/features.npz.")
            feature_format = st.selectbox("Feature format", ["shared_pt", "dimodif_npz"], index=0)
            device = st.selectbox("Execution device (no automatic fallback)", ["cpu", "cuda", "cuda:0", "mps"], index=0)
            modes = st.multiselect("Precision configurations", list(CONFIGS), default=list(CONFIGS), format_func=lambda value: CFG_LABELS.get(value, value))
            require_hash = st.checkbox("Require the agreed LAV-DF checkpoint SHA-256", value=True, disabled=True)
            run_id = st.text_input("Unique run ID (required)", value="")
            experiment_id = st.text_input("Optional experiment ID", value="")
            confirm_inputs = st.checkbox("I have verified that this machine has the external checkpoint and local feature files; run validation only.", value=False)
            submitted = st.form_submit_button("Validate inputs and run validation experiment", disabled=not status["execution_interface_available"])
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
            elif not run_id or run_id in st.session_state.get("submitted_runs", set()):
                st.error("Enter a new unique run ID; previously submitted IDs cannot execute twice in this session.")
            elif not modes:
                st.error("Select at least one precision configuration.")
            elif not confirm_inputs:
                st.warning("Confirm the local checkpoint and feature inputs before starting the run.")
            else:
                with st.spinner("Validating configuration and running the selected validation modes..."):
                    outcome = run_cached_feature_experiment(
                        root, upstream_root=str(upstream.resolve()), checkpoint_name=checkpoint_name,
                        manifest=str(manifest.resolve()), feature_root=str(features.resolve()),
                        feature_format=feature_format, device=device, modes=modes,
                        run_id=run_id, experiment_id=experiment_id, require_lavdf_hash=require_hash)
                if outcome.payload.get("phase") == "execution" and outcome.payload.get("cli_launched"):
                    st.session_state.setdefault("submitted_runs", set()).add(run_id)
                st.session_state["last_run_result"] = {"exit_code": outcome.code, **outcome.payload}
                code, output = outcome
                if outcome.payload:
                    st.json(outcome.payload)
                from avfuse.member3_runtime.cli import EXIT_NAMES
                st.write(f"Backend state: {EXIT_NAMES.get(code, 'TIMEOUT' if code == 124 else 'UNKNOWN')}")
                files.update(__import__("avfuse_ui.data_access", fromlist=["discover_files"]).discover_files(root))
                if code == 0:
                    st.success("Backend CLI exited successfully. Evaluate predictions with Member 2 before making research claims.")
                else:
                    if outcome.payload.get("phase") == "preflight":
                        st.error("Readiness check failed. No experiment CLI was launched.")
                    else:
                        st.error(f"Backend {outcome.payload.get('phase', 'command')} failed with code {code}.")
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
    if st.session_state.get("artifact_import_failed"):
        st.warning("Import incomplete: at least one selected file failed; comparisons are disabled.")
        return
    if rows:
        errors = validate_predictions(rows, root / "contracts/schemas/prediction.schema.json")
        st.caption(f"Source: {source} · {len(rows):,} prediction rows")
        if errors:
            st.error(f"Prediction contract check failed with {len(errors)} issue(s).")
            with st.expander("Validation details"):
                st.code("\n".join(errors[:50]))
        else:
            st.success("Prediction rows match the current schema; labels remain outside the prediction records.")
        if errors:
            return
        identities = sorted({(str(r.get("run_id")), str(r.get("split")), str(r.get("checkpoint_sha256"))) for r in rows})
        identity = st.selectbox("Prediction identity (run, split, checkpoint)", identities)
        st.caption("Dataset identity must be confirmed from the run provenance; prediction rows alone do not establish it.")
        rows = [r for r in rows if (str(r.get("run_id")), str(r.get("split")), str(r.get("checkpoint_sha256"))) == identity]
        df = pd.DataFrame(prediction_summary(rows))
        st.dataframe(df, use_container_width=True, hide_index=True)
        numeric = [r for r in rows if isinstance(r.get("score_fake"), (int, float))]
        if numeric:
            plot_df = pd.DataFrame(numeric)
            if "config" in plot_df and "score_fake" in plot_df:
                fig = px.box(plot_df, x="config", y="score_fake", color="config", points=False,
                             category_orders={"config": list(CONFIGS)}, template="plotly_white")
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
    st.warning("Synthetic smoke is infrastructure-only: it uses random tokens and a randomly initialized diagnostic head. Its scores are not deepfake predictions, accuracy metrics or benchmark evidence.")
    with st.expander("Run synthetic infrastructure smoke test — diagnostic only", expanded=False):
        st.caption("This command checks software wiring only. Never copy its scores into prediction or evaluation artifacts.")
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


def render_evaluation(root: Path, files: dict[str, list[Path]], *, imported_only=False) -> None:
    page_header("FORENSIC EVALUATION / MEMBER 2", "Evaluation results",
                "Overall and manipulation-stratified metrics, with validation-frozen thresholds and source-cluster confidence intervals.")
    display_mode = "Imported evaluation artifact" if imported_only else st.radio("Evaluation source", ["Saved research results", "Imported evaluation artifact"])
    if display_mode == "Saved research results":
        from .data_access import saved_result_table
        st.info("Saved research results: committed source-bound aggregate presentations; no evaluation is executed.")
        st.warning("Overall precision damage and V/output superiority were not statistically established. Confidence intervals crossing zero do not prove equivalence.")
        st.caption("Frozen threshold 0.6627789735794068; source-cluster bootstrap. Fake INT8 uses floating-point arithmetic. Source groups are not verified speaker identities; checkpoint-selection and historical test-summary exposure limit interpretation.")
        for filename in ("test_overall_metrics.csv", "test_manipulation_metrics.csv", "test_paired_comparisons.csv", "validation_overall_metrics.csv"):
            with st.expander(filename, expanded=filename == "test_overall_metrics.csv"):
                try:
                    records = saved_result_table(root, filename)
                    st.caption(f"{len(records)} saved records; source_sha256 identifies the original evidence. Paired delta = mode_b minus mode_a.")
                    st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)
                except (OSError, ValueError) as exc:
                    st.warning(f"Saved table unavailable: {exc}")
        figures = list((root / "docs/member2/results/test_figures").glob("*.png"))
        for figure in figures:
            st.image(str(figure), caption=f"Original saved figure: {figure.name}")
        return
    path, upload = choose_local_or_upload(root, "metrics.json", files["metrics"], "metrics", ["json"])
    report = None
    source = None
    if upload:
        try:
            report = json.loads(read_uploaded(upload))
            if not isinstance(report, dict):
                raise ValueError("Expected a metrics JSON object")
            source = f"uploaded: {upload.name} (session only)"
        except Exception as exc:
            st.error(f"Could not parse metrics JSON: {exc}")
    elif path:
        try:
            report = load_json(path)
            source = str(path)
        except Exception as exc:
            st.error(f"Could not read metrics report: {type(exc).__name__}: {exc}")
    if report:
        errors = validate_metrics_report(report)
        if errors:
            st.error("Metrics report structure needs review: " + "; ".join(errors[:8]))
            return
        split = report.get("split")
        st.caption(f"Source: {source} · split: {split}")
        if errors:
            st.error("Metrics report structure needs review: " + "; ".join(errors[:8]))
            return
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
        panel_title("Overall results", "Each CI is a source-cluster bootstrap interval, not an independent-clip interval.")
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
            fig = px.bar(pd.DataFrame(chart_rows), x="Metric", y="Value", color="Configuration", barmode="group", range_y=[0,1], template="plotly_white")
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
            with st.expander("Manipulation-stratified metrics"):
                st.dataframe(pd.DataFrame(strata), use_container_width=True, hide_index=True)
        paired = report.get("paired", [])
        if isinstance(paired, list) and paired:
            with st.expander("Paired configuration comparisons — supporting analysis"):
                st.caption("`delta` is b − a; a result is marked statistically distinguishable only if the paired 95% CI excludes zero.")
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
    st.info("No evaluation available: select a structured metrics.json or choose Saved research results.")


def render_benchmarks(root: Path, files: dict[str, list[Path]], *, imported_only=False) -> None:
    st.session_state["artifact_import_failed"] = False
    page_header("RUNTIME / MEMBER 3", "Benchmark explorer",
                "Read the Member 3 benchmark records without conflating operator, cached-feature model, or preprocessing costs.")
    st.markdown("<div class='avf-warn'><b>Precision disclosure:</b> all non-reference configurations are simulated fake quantization in floating-point arithmetic. The runtime records do not measure a real INT8 kernel or prove INT8 hardware speedup.</div>", unsafe_allow_html=True)

    display_mode = "Imported benchmark records" if imported_only else st.radio("Benchmark source", ["Saved CPU research results", "Imported benchmark records"])
    if display_mode == "Saved CPU research results":
        from .data_access import saved_result_table
        st.info("Saved CPU evidence: cached_feature_model scope; three workloads × four modes. Not raw-video end-to-end latency or native INT8 acceleration. Peak CPU memory was not measured.")
        for filename in ("cpu_model_runtime.csv", "cpu_model_timing_samples.csv"):
            try:
                records = saved_result_table(root, filename)
                st.caption(f"{filename}: {len(records)} accepted records; original source hashes retained.")
                st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)
            except (OSError, ValueError) as exc:
                st.warning(f"Saved evidence unavailable: {exc}")
        return
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
    if st.session_state.get("artifact_import_failed"):
        st.warning("Import incomplete: at least one selected file failed; comparisons are disabled.")
        return
    schema = root / "contracts/schemas/benchmark.schema.json"
    errors = []
    if uploads:
        errors = validate_benchmarks(rows, schema)
    else:
        for selected_path in selected_paths:
            errors.extend(validate_benchmarks(load_jsonl_with_error(selected_path, "benchmark JSONL"), schema, run_dir=selected_path.parent))
    st.caption(f"Source: {source} · {len(rows)} benchmark record(s). Local discovery includes both `artifacts/benchmarks/*.jsonl` and finalized run bundles' `benchmark_records.jsonl`.")
    st.warning("The checked-in benchmark schema is documented as proposed v0.2 and pending cross-member review. Validation below means schema-conforming to the current file, not that the contract has been finally approved.")
    if errors:
        st.error(f"{len(errors)} record validation issue(s).")
        with st.expander("Validation details"):
            st.code("\n".join(errors[:50]))
        st.info("The benchmark explorer stops here for nonconforming rows so it does not silently chart malformed fields. Correct the JSONL against the checked-in schema and re-import it.")
        return
    else:
        st.success("Records passed backend schema and cross-field checks, including available timing samples.")
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
    mem_values = view.get("peak_memory_bytes", pd.Series(dtype=float)).dropna()
    c = st.columns(3)
    c[0].metric("Series", f"{len(view):,}")
    c[1].metric("Latency", "See identity groups")
    c[2].metric("Peak memory", human_bytes(mem_values.max()) if len(mem_values) else "Not measured")
    group_columns = [column for column in ("run_id", "scope", "device", "backend", "config", "workload") if column in view.columns]
    if "workload" in view:
        view["workload"] = view["workload"].map(lambda value: json.dumps(value, sort_keys=True))
    if group_columns:
        grouped = view.groupby(group_columns, dropna=False)
        comparison = grouped.size().rename("Records").to_frame()
        for latency_name in ("p50_ms", "p95_ms"):
            if latency_name in view.columns:
                comparison[f"Median recorded {latency_name}"] = grouped[latency_name].median()
        with st.expander("Grouped benchmark comparison", expanded=True):
            st.dataframe(comparison.reset_index(), use_container_width=True, hide_index=True)
            st.caption("Group values are medians of the per-record p50/p95 measurements shown in the JSONL. They are not pooled per-call percentiles; compare only records with matching scope, device and workload.")
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
    run_dirs = files.get("run_dirs", [])
    discovered_runs = run_dirs
    st.caption("Run counts are discovery counts. Validated runs: not counted until explicit integrity checks; all discovered runs remain unverified in this view.")
    provenance_count = sum((path / "provenance.json").is_file() for path in discovered_runs)
    hash_manifest_count = sum((path / "integrity" / "artifact_hashes.json").is_file() for path in discovered_runs)
    status_cols = st.columns(5)
    for col, label, value in zip(
        status_cols,
        ("Environment reports", "Discovered run directories", "Provenance files", "Artifact hash manifests", "Stage profiles"),
        (len(files.get("environment_reports", [])), len(discovered_runs), provenance_count,
         hash_manifest_count, len(files.get("stage_profiles", []))),
    ):
        col.metric(label, value)
    st.caption("These counts show artifact availability only. Integrity is checked only by the verification button below; a present hash manifest is not itself a passing verification.")
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
            st.caption("No environment report selected; choose a discovered report above.")
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
            from .data_access import discover_files
            files.update(discover_files(root))
            if output:
                with st.expander("Command output"):
                    st.code(output)
    with col2:
        st.subheader("Run integrity")
        chosen = st.selectbox("Run folder", ["— none —", *[str(p.relative_to(root)) for p in discovered_runs]], key="run_select")
        run_dir = next((p for p in discovered_runs if str(p.relative_to(root)) == chosen), None)
        if run_dir:
            from .run_inspection import (optional_metadata, manifest_structure_valid,
                                         verify_with_receipt, receipt_is_current)
            metadata = {}
            for filename in ("run_manifest.json", "execution_summary.json", "failure_summary.json",
                             "resolved_config.json", "provenance.json"):
                value, error = optional_metadata(run_dir / filename)
                metadata[filename] = value
                if error:
                    st.error(error)
                if value is not None:
                    with st.expander(filename, expanded=filename.endswith("summary.json")):
                        st.json(value)
            manifest = metadata["run_manifest.json"]
            valid_structure = manifest_structure_valid(manifest)
            if manifest is None:
                st.warning("No readable run manifest: discovered run is incomplete or invalid.")
            elif not valid_structure:
                st.error("Invalid run_manifest.json structure; finalization is not established.")
            else:
                st.write(f"**Run ID:** {manifest['run_id']}")
                st.write(f"**Artifacts listed:** {len(manifest['artifacts'])}")
                st.caption("Finalization metadata present; integrity is not established by its presence.")
            summary = metadata["execution_summary.json"]
            failure = metadata["failure_summary.json"]
            if failure is not None:
                st.warning("Failure summary present. This run is not established as a completed experiment.")
            elif summary is not None:
                st.write(f"Recorded execution status: {summary.get('status', 'not stated')}")
            else:
                st.caption("Execution completion not established: no readable execution summary.")
            st.caption("Hash consistency does not authenticate authorship or establish scientific evaluation success.")
            if st.button("Verify selected run integrity", key="verify_run"):
                try:
                    receipt = verify_with_receipt(run_dir)
                    st.session_state.setdefault("verified_runs", {})[str(run_dir)] = receipt
                except Exception as exc:
                    st.session_state.setdefault("verified_runs", {}).pop(str(run_dir), None)
                    st.error(f"Integrity verification failed for {run_dir.name}: {type(exc).__name__}: {exc}")
            receipt = st.session_state.get("verified_runs", {}).get(str(run_dir))
            if receipt is None:
                st.info("Not verified: use the explicit integrity button when needed.")
            else:
                st.caption(f"Past check: {receipt['checked_at_utc']}. Not a continuous guarantee.")
                if not receipt_is_current(run_dir, receipt):
                    st.warning("Not verified for current files: metadata changed since or during the past check. Verify again.")
                elif receipt["verdict"].get("ok"):
                    st.success("Artifact integrity passed at the recorded check time; scientific validity is separate.")
                else:
                    st.error("Artifact integrity failed at the recorded check time.")
                st.json(receipt["verdict"])
        else:
            st.caption("No run selected. Select a discovered run to inspect metadata without automatically hashing it.")
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
                if not isinstance(stage, dict):
                    st.error("Invalid stage_profile.json entry: expected an object.")
                    continue
                row = {k: stage.get(k) for k in ("stage", "scope_group", "stage_kind", "status", "input_kind", "p50_ms", "p95_ms", "measurement_origin") if k in stage}
                if row: stage_rows.append(row)
            if stage_rows:
                st.dataframe(pd.DataFrame(stage_rows), use_container_width=True, hide_index=True)
        st.json(profile)
    else:
        st.caption("No stage profile selected. Choose a discovered artifact above; missing stages are not measurements.")


def render_import_help(root: Path) -> None:
    page_header("LOCAL FILE WORKFLOW", "Artifact import & file handling",
                "The dashboard reads research artifacts from the local repository or from files uploaded to the Streamlit server. Remote uploads do not remain exclusively on your device.")
    st.markdown("<div class='avf-note'><b>Local file handling:</b> imported JSON/JSONL files are parsed by the Streamlit process for the current session and are not copied into Git or persisted by this dashboard. Run Streamlit locally or use only a trusted host: if this app is served remotely, the selected files are received by that host. Move canonical files into the repository's documented data/ or artifacts/ folders for automatic discovery on the next launch.</div>", unsafe_allow_html=True)
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
    st.info(f"Repository root detected: `{root}`. Set `AVFUSE_REPO_ROOT` before launch if the code is stored elsewhere; the sidebar does not edit server paths.")


# DeepTrace primary workflow. Existing artifact explorers remain available in details.
MODE_HELP = {
    "reference": "FP32 reference preserves the original model forward.",
    "low_precision": "Simulated INT8 rounds Q, K, V and attention output in floating point.",
    "restore_value_output": "Value/output restoration preserves V and attention output at FP32.",
    "restore_query": "Query restoration control preserves Q at FP32.",
}


def _use_registered_manifest():
    value = st.session_state["registered_manifest"]
    if value != "Enter a path":
        st.session_state["draft_manifest"] = value
        st.session_state.setdefault("experiment_draft", {})["manifest"] = value


def _open_generated_results():
    last = st.session_state.get("last_run_result", {})
    if last.get("run_dir"):
        st.session_state["selected_run"] = last["run_dir"]
        st.session_state["result_source"] = "New validation run"
    _open_results()


def _open_saved_results():
    st.session_state["result_source"] = "Saved evidence"
    _open_results()


def _open_results():
    st.session_state["workbench_page"] = "Results"


def _draft_widget(kind, label, name, default, **kwargs):
    # Durable non-widget state survives Streamlit's cleanup of hidden page widgets.
    draft = st.session_state.setdefault("experiment_draft", {})
    key = "draft_" + name
    if key not in st.session_state:
        st.session_state[key] = draft.get(name, default)
    value = getattr(st, kind)(label, key=key, **kwargs)
    draft[name] = value
    return value


def _journey(stage=1):
    steps = ["1. Configure Experiment", "2. Run Experiment", "3. View Results"]
    st.markdown('<div class="journey">' + ' &nbsp; → &nbsp; '.join(
        f"<b>{s}</b>" if i == stage else s for i, s in enumerate(steps, 1)) + '</div>', unsafe_allow_html=True)


def render_experiment(root: Path, files: dict[str, list[Path]]) -> None:
    journey = st.empty()
    left, right = st.columns([1.55, 1], gap="large")
    with left, st.container(border=True):
        st.subheader("Experiment setup")
        source = _draft_widget("radio", "Experiment source", "source", "Explore accepted research results",
            options=["Explore accepted research results", "Run new validation experiment"])
        saved = source == "Explore accepted research results"
        st.caption("Saved evidence · historical study; no inference is executed." if saved else
                   "New validation run · authorized local cached features required.")
        st.write("**Precision configurations**")
        modes = _draft_widget("multiselect", "Configurations", "modes", list(CONFIGS), options=list(CONFIGS),
                             format_func=lambda x: CFG_LABELS[x])
        for mode in modes:
            st.caption(MODE_HELP[mode])
        if saved:
            st.info("Explore the accepted LAV-DF study, its source-bound metrics and recorded CPU costs.")
            st.button("View accepted results", type="primary", on_click=_open_saved_results)
        else:
            st.write("**Dataset: LAV-DF · validation only**")
            st.caption("Assets required. No raw media upload or held-out test execution.")
            with st.expander("Execution settings"):
                from .workbench_service import available_validation_manifests
                manifests = available_validation_manifests(root, files.get("manifests", []))
                if manifests:
                    st.selectbox("Available validation manifests", ["Enter a path", *[str(p) for p in manifests]],
                        format_func=lambda x: str(Path(x).relative_to(root)) if x != "Enter a path" else x,
                        key="registered_manifest", on_change=_use_registered_manifest)
                else:
                    st.caption("No schema-valid LAV-DF validation manifests discovered. Supply an authorized manifest path below.")
                _draft_widget("text_input", "External DiMoDif checkout", "upstream_root", "")
                _draft_widget("text_input", "Validation manifest", "manifest", "")
                _draft_widget("text_input", "Feature root", "feature_root", "")
                _draft_widget("selectbox", "Feature format", "feature_format", "dimodif_npz", options=["dimodif_npz", "shared_pt"])
                _draft_widget("selectbox", "Device", "device", "cpu", options=["cpu", "cuda", "cuda:0", "mps"])
                _draft_widget("text_input", "Run ID", "run_id", "")
                _draft_widget("text_input", "Experiment ID (optional)", "experiment_id", "")
                st.caption("The frozen LAV-DF checkpoint SHA-256 is required and enforced by the backend.")
            from .workbench_service import validate_experiment_readiness, execute_validation_experiment, readiness_is_current
            draft = st.session_state["experiment_draft"]
            request = {k: draft.get(k, "") for k in ("upstream_root", "manifest", "feature_root", "feature_format", "device", "run_id", "experiment_id")}
            request.update(checkpoint_name="lavdf", require_lavdf_hash=True, modes=modes)
            if st.session_state.get("authorization_request") != request:
                st.session_state["authorization_request"] = request
                st.session_state.pop("readiness", None)
                st.session_state["draft_authorized"] = False
                draft["authorized"] = False
            prior_readiness = st.session_state.get("readiness")
            if prior_readiness and prior_readiness.code == 0 and not readiness_is_current(prior_readiness):
                st.session_state["draft_authorized"] = False
                draft["authorized"] = False
            authorized = _draft_widget("checkbox", "I authorize validation inference with these trusted local assets.", "authorized", False)
            smoke = _draft_widget("checkbox", "Test one real validation sample during readiness", "smoke", False,
                help="Loads the licensed model through the existing CLI. Temporary scores are discarded; this is not a completed experiment.")
            if st.button("Check Readiness", disabled=bool(smoke and not authorized)):
                with st.spinner("Checking assets and software configuration…"):
                    st.session_state["readiness"] = validate_experiment_readiness(root, smoke=smoke, **request)
            readiness = st.session_state.get("readiness")
            ready = bool(readiness and readiness.code == 0 and readiness_is_current(readiness))
            if readiness:
                if readiness.code:
                    st.error(readiness.diagnostics)
                elif not ready:
                    st.warning("Assets changed since readiness. Check again before execution.")
                else:
                    st.success(readiness.diagnostics)
                with st.expander("Asset readiness details"):
                    st.json({k: readiness.payload.get(k) for k in ("phase", "status", "cli_launched", "asset_counts", "missing_assets", "sample_inspected", "smoke_passed")})
            else:
                st.caption("Not configured · Check Readiness identifies missing assets before any experiment starts.")
            if st.button("Run Experiment", disabled=not (ready and authorized), type="primary"):
                with st.spinner("Executing the validation cohort and cached-feature benchmark…"):
                    outcome = execute_validation_experiment(root, **request)
                st.session_state["last_run_result"] = {"exit_code": outcome.code, **outcome.payload}
                st.session_state["last_execution_request"] = request
                st.session_state["run_diagnostics"] = outcome.diagnostics
                st.session_state.pop("readiness", None)
                if outcome.payload.get("run_dir"):
                    st.session_state["selected_run"] = outcome.payload["run_dir"]
                    st.session_state["result_source"] = "New validation run"
                files.update(__import__("avfuse_ui.data_access", fromlist=["discover_files"]).discover_files(root))
                st.rerun()
            last = st.session_state.get("last_run_result")
            if last:
                st.caption(f"Most recent submission · run {last.get('run_id', 'not started')}")
                if last.get("status") == "complete" and last.get("exit_code") == 0:
                    st.success("Inference completed. Label-grounded evaluation has not been performed.")
                elif last.get("phase") == "preflight":
                    st.error("Readiness check failed. No experiment CLI was launched.")
                else:
                    st.warning(f"Execution status: {last.get('status', 'unknown')} · exit {last.get('exit_code')}. Inspect the available run artifacts.")
                with st.expander("Execution details"):
                    st.json(last)
                    st.code(st.session_state.get("run_diagnostics", ""), language="text")
                st.button("Open generated results", on_click=_open_generated_results,
                          disabled=not bool(last.get("run_dir") and Path(last["run_dir"]).is_dir()))
    with right, st.container(border=True):
        st.subheader("Results preview")
        st.write("One performance table, one manipulation chart and one cost comparison.")
        st.caption("All source-bound, with optional technical details and explicit missing-data states.")
        st.write("**Accepted findings**")
        st.write("Overall degradation from simulated low precision and a V/output restoration advantage were not statistically established.")
        st.caption("Confidence intervals crossing zero do not establish equivalence.")
        st.button("Investigate results →", on_click=_open_results)

    stage = 1
    if not saved:
        readiness = st.session_state.get("readiness")
        if readiness and readiness.code == 0 and readiness_is_current(readiness):
            stage = 2
        last = st.session_state.get("last_run_result", {})
        if (last.get("exit_code") == 0 and last.get("status") == "complete"
                and st.session_state.get("last_execution_request") == request):
            stage = 3
    with journey:
        _journey(stage)


def render_results(root: Path, files: dict[str, list[Path]]) -> None:
    from .workbench_results import render
    render(root, files)


def render_about(root: Path, files: dict[str, list[Path]]) -> None:
    st.subheader("About DeepTrace")
    st.write("Audio and video can each carry manipulation cues. DeepTrace investigates how numerical precision in fusion attention affects a pretrained detector's decisions, and whether selectively preserving precision recovers performance.")
    st.write("Cached audio/video features → pretrained DiMoDif → attention precision intervention → fake scores → separate evaluation and benchmarking.")
    st.caption("The audited model uses joint-sequence audio-visual self-attention across five layers.")
    for mode in CONFIGS:
        st.write(f"**{CFG_LABELS[mode]}** — {MODE_HELP[mode]}")
    st.write("The engineering contribution includes model instrumentation, dense/SDPA correctness verification, controlled precision interventions, source-aware evaluation, hardware measurement and reproducible artifacts.")
    st.warning("Cached-feature input only; no arbitrary raw video upload. Simulated INT8 is floating-point fake quantization, not integer-kernel deployment. New inference requires external licensed assets and is validation-only. Overall precision damage and V/output superiority were not statistically established; these negative/inconclusive results do not prove equality or universal robustness.")
    st.caption("Source groups do not establish speaker independence. Checkpoint-selection bias and historical test-summary exposure limit interpretation. No universal deepfake-detection or deployment claim follows.")
    st.markdown("[Methodology and research documentation](https://github.com/SudarshanG9/AFML-project/tree/main/docs)")
    with st.expander("Artifact formats and local privacy"):
        render_import_help(root)
    with st.expander("Original attention and masking diagnostics"):
        render_diagnostics(root, files)
