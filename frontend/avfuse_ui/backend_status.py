from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

REQUIRED_RUNTIME_FILES = (
    "__main__.py", "experiment.py", "feature_loader.py",
    "model_benchmark.py", "predictions.py", "runner.py",
)
REQUIRED_COMMANDS = ("validate-config", "run", "predict", "benchmark", "inspect")


def detect_repo_root(start: Path) -> Path:
    start = start.resolve()
    for candidate in (start, *start.parents):
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
    """Source inspection only; does not verify a checkpoint or run inference."""
    member1 = root / "src" / "avfuse" / "member1_fusion"
    member3 = root / "src" / "avfuse" / "member3_runtime"
    adapter = member1 / "dimodif_adapter.py"
    intervention = member1 / "dimodif_intervention.py"
    cli = member3 / "cli.py"
    adapter_text = adapter.read_text(encoding="utf-8", errors="replace") if adapter.is_file() else ""
    intervention_text = intervention.read_text(encoding="utf-8", errors="replace") if intervention.is_file() else ""
    cli_text = cli.read_text(encoding="utf-8", errors="replace") if cli.is_file() else ""
    adapter_has_intervention = (
        "def forward_with_mode" in adapter_text
        and "intervene_dimodif_attention" in adapter_text
        and "self.forward_with_mode" in adapter_text
        and intervention.is_file()
        and "fusion_attention(" in intervention_text
    )
    reference_only = (
        'if mode != "reference"' in adapter_text and "NotImplementedError" in adapter_text
    ) or ("raise NotImplementedError" in adapter_text and not adapter_has_intervention)
    four_mode_model_ready = adapter_has_intervention and not reference_only
    command_support = all(f'"{name}"' in cli_text for name in REQUIRED_COMMANDS)
    runtime_files = all((member3 / name).is_file() for name in REQUIRED_RUNTIME_FILES)
    real_cli_ready = command_support and runtime_files
    real_inference_ready = four_mode_model_ready and real_cli_ready
    if real_cli_ready:
        cli_status = "real cached-feature CLI (validation / prediction / benchmark / inspect)"
    elif "--smoke" in cli_text:
        cli_status = "synthetic smoke only"
    else:
        cli_status = "CLI or required runtime components missing"
    if four_mode_model_ready:
        adapter_status = "four-mode DiMoDif intervention path present; checkpoint required at runtime"
    elif reference_only:
        adapter_status = "reference-only in checked-in DiMoDifAdapter"
    elif adapter.is_file():
        adapter_status = "adapter source found; four-mode path not verified"
    else:
        adapter_status = "missing"
    notes = []
    if real_inference_ready:
        notes.extend([
            "The checked-in CLI supports real cached-feature execution; no raw-media inference is implied.",
            "The DiMoDif adapter dispatches all four precision configurations through the attention-intervention path.",
            "A real run still requires the external DiMoDif checkout, matching checkpoint/config, manifest and cached feature files.",
            "Source readiness does not reproduce metrics or authorize held-out test access.",
        ])
    else:
        if not real_cli_ready:
            notes.append("The complete Member 3 cached-feature CLI/runtime file set was not detected.")
        if not four_mode_model_ready:
            notes.append("A complete four-mode DiMoDif intervention dispatch was not detected.")
    return {
        "repository_readme": (root / "README.md").is_file(),
        "model_adapter_file": adapter.is_file(),
        "intervention_file": intervention.is_file(),
        "adapter_mode_status": adapter_status,
        "four_mode_model_ready": four_mode_model_ready,
        "cli_status": cli_status,
        "real_cli_ready": real_cli_ready,
        "real_inference_ready": real_inference_ready,
        "notes": notes,
    }


def _command_env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    src = str(root / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def run_diagnostic_smoke(root: Path) -> tuple[int, str]:
    command = [sys.executable, "-m", "avfuse.member3_runtime.cli", "--smoke"]
    try:
        proc = subprocess.run(command, cwd=root, env=_command_env(root), capture_output=True,
                              text=True, timeout=120, check=False)
        return proc.returncode, (proc.stdout + ("\n" + proc.stderr if proc.stderr else "")).strip()
    except subprocess.TimeoutExpired:
        return 124, "Diagnostic smoke test exceeded 120 seconds and was stopped."
    except OSError as exc:
        return 1, f"Could not start diagnostic smoke test: {exc}"


def run_environment_check(root: Path) -> tuple[int, str, Path]:
    out = root / "artifacts" / "runs" / "environment_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    script = root / "scripts" / "check_environment.py"
    if not script.is_file():
        return 1, "scripts/check_environment.py was not found.", out
    try:
        proc = subprocess.run([sys.executable, str(script), "--output", str(out)], cwd=root,
                              env=_command_env(root), capture_output=True, text=True,
                              timeout=120, check=False)
        return proc.returncode, (proc.stdout + ("\n" + proc.stderr if proc.stderr else "")).strip(), out
    except subprocess.TimeoutExpired:
        return 124, "Environment check exceeded 120 seconds and was stopped.", out
    except OSError as exc:
        return 1, f"Could not start environment check: {exc}", out


def run_cached_feature_experiment(
    root: Path, *, upstream_root: str, checkpoint_name: str, manifest: str,
    feature_root: str, feature_format: str, device: str, modes: Sequence[str],
    run_id: str = "", experiment_id: str = "", require_lavdf_hash: bool = True,
    timeout_seconds: int = 86400,
) -> tuple[int, str]:
    """Validate then run real cached-feature inference on validation only."""
    if not modes:
        return 2, "Select at least one precision mode."
    if checkpoint_name not in {"lavdf", "avdeepfake1m"}:
        return 2, "Select a supported checkpoint name."
    if feature_format not in {"dimodif_npz", "shared_pt"}:
        return 2, "Select a supported feature format."
    common = [
        "--upstream-root", upstream_root, "--checkpoint-name", checkpoint_name,
        "--manifest", manifest, "--feature-root", feature_root,
        "--feature-format", feature_format, "--device", device,
        "--modes", *modes, "--split", "val",
    ]
    if checkpoint_name == "lavdf" and require_lavdf_hash:
        common.append("--require-lavdf-hash")
    env = _command_env(root)
    validate_cmd = [sys.executable, "-m", "avfuse.member3_runtime.cli", "validate-config", *common]
    try:
        checked = subprocess.run(validate_cmd, cwd=root, env=env, capture_output=True,
                                 text=True, timeout=300, check=False)
    except subprocess.TimeoutExpired:
        return 124, "Backend validation exceeded 300 seconds and was stopped."
    except OSError as exc:
        return 1, f"Could not start backend validation: {exc}"
    validate_output = (checked.stdout + ("\n" + checked.stderr if checked.stderr else "")).strip()
    if checked.returncode != 0:
        return checked.returncode, "VALIDATE-CONFIG FAILED\n" + validate_output
    run_options = ["--runs-root", "artifacts/member3/runs"]
    if run_id.strip():
        run_options.extend(["--run-id", run_id.strip()])
    if experiment_id.strip():
        run_options.extend(["--experiment-id", experiment_id.strip()])
    run_cmd = [sys.executable, "-m", "avfuse.member3_runtime.cli", "run", *common, *run_options]
    try:
        executed = subprocess.run(run_cmd, cwd=root, env=env, capture_output=True,
                                  text=True, timeout=timeout_seconds, check=False)
    except subprocess.TimeoutExpired:
        return 124, "Backend run exceeded its configured time limit. Check the run folder for failure_summary.json."
    except OSError as exc:
        return 1, f"Could not start backend run: {exc}"
    run_output = (executed.stdout + ("\n" + executed.stderr if executed.stderr else "")).strip()
    return executed.returncode, "VALIDATE-CONFIG PASSED\n" + validate_output + "\n\nRUN\n" + run_output
