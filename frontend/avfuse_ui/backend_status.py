from __future__ import annotations

import os
import json
import re
import uuid
from dataclasses import dataclass, field
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


def git_info(root: Path) -> dict[str, Any]:
    def run(args: list[str]) -> str | None:
        try:
            proc = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True,
                                  timeout=5, check=False)
            return proc.stdout.strip() if proc.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None
    dirty = run(["status", "--porcelain"])
    return {"branch": run(["branch", "--show-current"]),
            "commit": run(["rev-parse", "--short", "HEAD"]),
            "dirty": bool(dirty) if dirty is not None else None}


def inspect_backend(root: Path) -> dict[str, Any]:
    """Source/import/parser checks only; never construct a model or verify assets."""
    import importlib
    required = ("member1_fusion.dimodif_adapter", "member1_fusion.dimodif_intervention",
                "member2_data.cache", "member3_runtime.cli", "member3_runtime.artifacts",
                "member3_runtime.feature_loader", "shared.contracts")
    source_available = all((root / "src/avfuse" / (name.replace(".", "/") + ".py")).is_file() for name in required)
    imported = False
    detail = "Required source unavailable"
    if source_available:
        try:
            modules = [importlib.import_module("avfuse." + name) for name in required]
            if not all(Path(module.__file__).resolve().is_relative_to((root / "src").resolve()) for module in modules):
                raise ImportError("Installed package does not match selected repository")
            adapter, _, _, cli, _, _, _ = modules
            imported = callable(adapter.DiMoDifAdapter.forward_with_mode)
            parser = cli.build_parser()
            for name in REQUIRED_COMMANDS:
                parser.parse_args([name, *({"predict": ["--output", "unused.jsonl"], "inspect": ["unused"]}.get(name, []))])
            detail = "Actual adapter imports and CLI parser checked; no model loaded"
        except (ImportError, AttributeError, SystemExit) as exc:
            imported = False
            detail = f"Software interface check failed: {exc}"
    return {
        "source_available": source_available, "interfaces_imported": imported,
        "software_checked": "Interfaces imported; synthetic smoke requires explicit action" if imported else detail,
        "assets_available": "Not checked; select explicit local assets",
        "execution_authorized": "Only confirmed validation requests",
        "run_completed": "Not checked",
        "artifact_integrity_verified": "Not checked; explicit verification required",
        "repository_readme": (root / "README.md").is_file(),
        "model_adapter_file": source_available, "intervention_file": source_available,
        "adapter_mode_status": detail, "adapter_interface_available": imported,
        "cli_status": detail, "cli_interface_available": imported,
        "execution_interface_available": imported,
        "notes": ["Source and software checks do not verify external assets, complete a run, or establish research validity."]}


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


def _invoke_cached_feature_experiment(
    root: Path, *, upstream_root: str, checkpoint_name: str, manifest: str,
    feature_root: str, feature_format: str, device: str, modes: Sequence[str],
    run_id: str = "", experiment_id: str = "", require_lavdf_hash: bool = True,
    timeout_seconds: int = 86400,
) -> tuple[int, str]:
    """Validate then run real cached-feature inference on validation only."""
    from avfuse.shared.contracts import PRECISION_MODES
    from avfuse.member1_fusion.dimodif_adapter import CHECKPOINTS
    from .data_access import load_jsonl, validate_manifest, authorized_path
    if not modes or any(mode not in PRECISION_MODES for mode in modes):
        return 2, "Select at least one precision mode."
    if checkpoint_name not in {"lavdf", "avdeepfake1m"}:
        return 2, "Select a supported checkpoint name."
    if feature_format not in {"dimodif_npz", "shared_pt"}:
        return 2, "Select a supported feature format."
    if not re.fullmatch(r"cpu|cuda(?::[0-9]+)?|mps", device):
        return 2, "An explicit supported device is required."
    if checkpoint_name != "lavdf" or not require_lavdf_hash:
        return 7, "This study UI requires the frozen LAV-DF checkpoint hash."
    for value in (run_id, experiment_id):
        if value and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", value):
            return 2, "Invalid run or experiment identifier."
    run_id = run_id or "ui-" + uuid.uuid4().hex
    if (root / "artifacts/member3/runs" / run_id).exists():
        return 2, "Run ID already exists; duplicate execution refused."
    try:
        upstream = authorized_path(root, upstream_root)
        manifest_path = authorized_path(root, manifest)
        features = authorized_path(root, feature_root)
        checkpoint = upstream / "ckpt/dfd" / (CHECKPOINTS[checkpoint_name] + ".pth")
        if not features.is_dir() or not checkpoint.is_file() or not checkpoint.with_suffix(".json").is_file():
            return 6, "Missing feature root, checkpoint or model configuration."
        rows = load_jsonl(manifest_path)
        errors = validate_manifest(rows, root / "contracts/schemas/manifest.schema.json")
        if errors or any(row.get("split") != "val" for row in rows):
            return 2, "Use a schema-valid validation-only manifest: " + "; ".join(errors[:3])
        from avfuse.member3_runtime.feature_loader import dimodif_npz_path
        for row in rows:
            cache = (dimodif_npz_path(features, "val", row["clip_id"]) if feature_format == "dimodif_npz"
                     else features / row["feature_path"])
            cache = authorized_path(root, cache)
            if not cache.is_relative_to(features) or not cache.is_file():
                return 6, "Missing cache or cache escapes selected feature root."
        upstream_root, manifest, feature_root = str(upstream), str(manifest_path), str(features)
    except (OSError, ValueError) as exc:
        return 6, str(exc)
    common = [
        "--upstream-root", upstream_root, "--checkpoint-name", checkpoint_name,
        "--manifest", manifest, "--feature-root", feature_root,
        "--feature-format", feature_format, "--device", device,
        "--modes", *modes, "--split", "val",
    ]
    if checkpoint_name == "lavdf" and require_lavdf_hash:
        common.append("--require-lavdf-hash")
    env = _command_env(root)
    validate_cmd = [sys.executable, "-m", "avfuse.member3_runtime.cli", "validate-config", *common, "--json"]
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
    try:
        validation = json.loads(checked.stdout)
        if not isinstance(validation, dict) or validation.get("ok") is not True:
            raise ValueError("Missing successful validation status")
    except ValueError as exc:
        return 1, f"Malformed backend validation JSON: {exc}"
    run_options = ["--runs-root", "artifacts/member3/runs"]
    if run_id.strip():
        run_options.extend(["--run-id", run_id.strip()])
    if experiment_id.strip():
        run_options.extend(["--experiment-id", experiment_id.strip()])
    run_cmd = [sys.executable, "-m", "avfuse.member3_runtime.cli", "run", *common, *run_options, "--json"]
    try:
        executed = subprocess.run(run_cmd, cwd=root, env=env, capture_output=True,
                                  text=True, timeout=timeout_seconds, check=False)
    except subprocess.TimeoutExpired:
        return 124, "Backend run exceeded its configured time limit. Check the run folder for failure_summary.json."
    except OSError as exc:
        return 1, f"Could not start backend run: {exc}"
    run_output = (executed.stdout + ("\n" + executed.stderr if executed.stderr else "")).strip()
    from avfuse.member3_runtime.cli import EXIT_NAMES
    if executed.returncode == 0:
        try:
            result = json.loads(executed.stdout)
            if not isinstance(result, dict) or result.get("status") != "complete" or result.get("finalized") is not True:
                raise ValueError("Successful exit lacks completed and finalized status")
        except ValueError as exc:
            return 1, f"Malformed backend run JSON: {exc}"
    run_output = f"Status: {EXIT_NAMES.get(executed.returncode, 'UNKNOWN')}\nRun directory: {root / 'artifacts/member3/runs' / run_id}\n" + run_output
    return RunResult(executed.returncode, run_output,
                     {"validation": validation, "execution": result if executed.returncode == 0 else None,
                      "run_dir": str(root / "artifacts/member3/runs" / run_id)})


@dataclass
class RunResult:
    code: int
    diagnostics: str
    payload: dict[str, Any] = field(default_factory=dict)

    def __iter__(self):
        return iter((self.code, self.diagnostics))

    def __getitem__(self, index):
        return (self.code, self.diagnostics)[index]


def run_cached_feature_experiment(root: Path, **request: Any) -> RunResult:
    """Expose structured status separately from raw subprocess diagnostics."""
    result = _invoke_cached_feature_experiment(root, **request)
    return result if isinstance(result, RunResult) else RunResult(*result)
