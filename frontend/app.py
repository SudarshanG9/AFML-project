from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

from avfuse_ui.backend_status import detect_repo_root
from avfuse_ui.data_access import discover_files
from avfuse_ui.ui import (
    inject_styles, render_benchmarks, render_dataset, render_demo,
    render_diagnostics, render_evaluation, render_import_help, render_overview, render_runs,
)

st.set_page_config(
    page_title="AVFuse-CausalQ | Research Dashboard",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_styles()

start = Path(os.environ.get("AVFUSE_REPO_ROOT", Path(__file__).resolve().parents[1]))
default_root = detect_repo_root(start)

with st.sidebar:
    st.markdown('<div class="avf-brand"><div class="avf-mark">AV</div><div>AVFuse-CausalQ<br><span style="color:#91a4bd;font-size:.72rem;font-weight:500">Research dashboard</span></div></div>', unsafe_allow_html=True)
    st.write("")
    root_input = st.text_input("Repository root", value=str(default_root), help="Folder containing pyproject.toml, src/, contracts/ and artifacts/.")
    repo_root = Path(root_input).expanduser().resolve()
    valid_root = (repo_root / "pyproject.toml").is_file() and (repo_root / "src/avfuse").is_dir()
    if not valid_root:
        st.error("This folder does not look like the AVFuse-CausalQ repository.")
    st.divider()
    page = st.radio("WORKSPACE", ["Overview", "Dataset & features", "Fusion diagnostics", "Demo & predictions", "Evaluation", "Benchmarks", "Runs & system", "Artifact import"], label_visibility="visible")
    st.divider()
    st.caption("Local-only · Artifact-driven · No fabricated metrics")

if not valid_root:
    st.title("Repository not found")
    st.write("Point the Repository root field to your local `AFML-project` checkout.")
    st.stop()

# Make the project package importable for backend checks and the existing integrity verifier.
src = str(repo_root / "src")
if src not in sys.path:
    sys.path.insert(0, src)

files = discover_files(repo_root)

if page == "Overview":
    render_overview(repo_root, files)
elif page == "Dataset & features":
    render_dataset(repo_root, files)
elif page == "Fusion diagnostics":
    render_diagnostics(repo_root, files)
elif page == "Demo & predictions":
    render_demo(repo_root, files)
elif page == "Evaluation":
    render_evaluation(repo_root, files)
elif page == "Benchmarks":
    render_benchmarks(repo_root, files)
elif page == "Runs & system":
    render_runs(repo_root, files)
else:
    render_import_help(repo_root)

with st.sidebar:
    st.divider()
    st.caption(f"Discovered locally: {len(files['manifests'])} manifest · {len(files['predictions'])} prediction file(s) · {len(files['benchmarks'])} benchmark file(s) · {len(files['run_dirs'])} finalized run folder(s)")
