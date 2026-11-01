"""Audit the runtime environment and write a JSON report.

Usage (Windows PowerShell / Command Prompt / Linux / macOS):

    python scripts/check_environment.py
    python scripts/check_environment.py --output artifacts/runs/my_environment.json

The default report path is ``artifacts/runs/environment_report.json`` inside the
repository (independent of the current working directory). All logic lives in
``avfuse.member3_runtime.environment``; this file is only a thin launcher.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    from avfuse.member3_runtime.environment import DEFAULT_OUTPUT, main
except ImportError:  # package not pip-installed: fall back to the in-repo source tree
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from avfuse.member3_runtime.environment import DEFAULT_OUTPUT, main

if __name__ == "__main__":
    sys.exit(main(default_output=REPO_ROOT / DEFAULT_OUTPUT, repo_root=REPO_ROOT))
