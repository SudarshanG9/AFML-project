from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def detect_repo_root(start: Path) -> Path:
    start = start.resolve()
    candidates = [start, *start.parents]
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "avfuse").is_dir():
            return candidate
    return start.parent if start.name == "frontend" else start


def git_info(root: Path) -> dict[str, str | None]:
    def run(args: list[str]) -> str | None:
        try:
            proc = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True,
                                  timeout=5, check=False)
            return proc.stdout.strip() if proc.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None
    return {"branch": run(["branch", "--show-current"]),
            "commit": run(["rev-parse", "--short", "HEAD"]),
            "dirty": "yes" if run(["status", "--porcelain"]) else "no"}


def inspect_backend(root: Path) -> dict[str, Any]:
    readme = root / "README.md"
    adapter = root / "src" / "avfuse" / "member1_fusion" / "dimodif_adapter.py"
    cli = root / "src" / "avfuse" / "member3_runtime" / "cli.py"
    model_adapter = root / "src" / "avfuse" / "member1_fusion" / "model_adapter.py"
    adapter_text = adapter.read_text(encoding="utf-8", errors="replace") if adapter.exists() else ""
    cli_text = cli.read_text(encoding="utf-8", errors="replace") if cli.exists() else ""
    model_text = model_adapter.read_text(encoding="utf-8", errors="replace") if model_adapter.exists() else ""
    reference_only = "mode != \"reference\"" in adapter_text and "NotImplementedError" in adapter_text
    smoke_only = "--smoke" in cli_text and "Real model not yet integrated" in cli_text
    # The base ModelAdapter is intentionally abstract even when a concrete model
    # adapter is complete; readiness must be derived from the concrete adapter/CLI.
    concrete_adapter = "class DiMoDifAdapter" in adapter_text and "def predict_from_cached_features" in adapter_text
    return {
        "repository_readme": readme.exists(),
        "model_adapter_file": adapter.exists(),
        "adapter_mode_status": "reference-only in checked-in DiMoDifAdapter" if reference_only else
                               ("adapter source found; inspect manually" if adapter.exists() else "missing"),
        "four_mode_model_ready": bool(concrete_adapter and not reference_only),
        "cli_status": "synthetic smoke only" if smoke_only else ("CLI found" if cli.exists() else "missing"),
        "real_inference_ready": bool(concrete_adapter and not reference_only and not smoke_only),
        "notes": [
            "The checked-in CLI accepts --smoke and explicitly rejects a real-model run.",
            "The checked-in DiMoDifAdapter returns predictions only for reference mode; other modes raise NotImplementedError.",
            "The ModelAdapter base class is abstract. A complete four-mode model runner is not exposed by the current CLI.",
        ] if smoke_only or reference_only else [],
    }


def run_diagnostic_smoke(root: Path) -> tuple[int, str]:
    env = os.environ.copy()
    src = str(root / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    command = [sys.executable, "-m", "avfuse.member3_runtime.cli", "--smoke"]
    try:
        proc = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True,
                           timeout=120, check=False)
        return proc.returncode, (proc.stdout + ("\n" + proc.stderr if proc.stderr else "")).strip()
    except subprocess.TimeoutExpired:
        return 124, "Diagnostic smoke test exceeded 120 seconds and was stopped."
    except OSError as exc:
        return 1, f"Could not start diagnostic smoke test: {exc}"


def run_environment_check(root: Path) -> tuple[int, str, Path]:
    out = root / "artifacts" / "runs" / "environment_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    src = str(root / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    script = root / "scripts" / "check_environment.py"
    if not script.is_file():
        return 1, "scripts/check_environment.py was not found.", out
    try:
        proc = subprocess.run([sys.executable, str(script), "--output", str(out)], cwd=root,
                              env=env, capture_output=True, text=True, timeout=120, check=False)
        return proc.returncode, (proc.stdout + ("\n" + proc.stderr if proc.stderr else "")).strip(), out
    except subprocess.TimeoutExpired:
        return 124, "Environment check exceeded 120 seconds and was stopped.", out
    except OSError as exc:
        return 1, f"Could not start environment check: {exc}", out
