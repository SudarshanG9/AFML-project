from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

from avfuse_ui.backend_status import detect_repo_root
from avfuse_ui.data_access import discover_files
from avfuse_ui.ui import inject_styles, render_experiment, render_results, render_about

st.set_page_config(page_title="DeepTrace | Precision Experimentation Workbench",
                   page_icon="🧪", layout="wide", initial_sidebar_state="expanded")
inject_styles()
root = detect_repo_root(Path(os.environ.get("AVFUSE_REPO_ROOT", Path(__file__).resolve().parents[1]))).resolve()
if not (root / "pyproject.toml").is_file() or not (root / "src/avfuse").is_dir():
    st.error("Repository not found. Set AVFUSE_REPO_ROOT before starting Streamlit.")
    st.stop()
if str(root / "src") not in sys.path:
    sys.path.insert(0, str(root / "src"))

# Preserve selected result source/run when their widgets are hidden by navigation.
for key in ("result_source", "selected_run"):
    if key in st.session_state:
        st.session_state[key] = st.session_state[key]

with st.sidebar:
    st.title("DeepTrace")
    st.caption("Precision Experimentation Workbench")
    st.divider()
    page = st.radio("Workspace", ["Experiment", "Results", "About"], key="workbench_page",
                    label_visibility="collapsed")

st.title("Precision Experimentation Workbench")
st.write("Run controlled experiments to investigate how fusion precision affects audio-visual deepfake detection.")
st.markdown('<span class="topic">Audio-visual forensics · Controlled precision</span>', unsafe_allow_html=True)
st.write("")
files = discover_files(root)
if page == "Experiment":
    render_experiment(root, files)
elif page == "Results":
    render_results(root, files)
else:
    render_about(root, files)
