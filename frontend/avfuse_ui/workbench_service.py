"""Frontend coordination only; scientific loaders, runners and evaluators stay authoritative."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import re

from .data_access import authorized_path, load_json, load_jsonl, validate_manifest, feature_file_summary
from avfuse.member3_runtime.provenance import sha256_file as checkpoint_digest


@dataclass
class AssetReadiness:
    code: int = 0
    phase: str = "preflight"
    status: str = "assets_checked"
    cli_launched: bool = False
    errors: list[str] = field(default_factory=list)
    missing_assets: list[str] = field(default_factory=list)
    asset_counts: dict[str, int] = field(default_factory=dict)
    request: dict[str, Any] = field(default_factory=dict)
    file_stamp: list[tuple] = field(default_factory=list)
    sample_inspected: bool = False

    @property
    def ready(self):
        return not self.errors and self.code == 0


def _device_error(device: str) -> str | None:
    import torch
    if not re.fullmatch(r"cpu|cuda(?::[0-9]+)?|mps", device):
        return "Choose an explicit supported device: cpu, cuda[:N] or mps."
    if device.startswith("cuda"):
        index = int(device.split(":")[1]) if ":" in device else 0
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            return f"Device {device} is unavailable; no automatic fallback."
    if device == "mps" and not torch.backends.mps.is_available():
        return "Device mps is unavailable; no automatic fallback."
    return None


def inspect_experiment_assets(root: Path, **request: Any) -> AssetReadiness:
    """Inspect every required path, but load at most one bounded cache, never a model."""
    from avfuse.shared.contracts import PRECISION_MODES
    from avfuse.member1_fusion.dimodif_adapter import CHECKPOINTS
    from avfuse.member2_data.readiness import DIMODIF_LAVDF_SHA256
    from avfuse.member3_runtime.feature_loader import dimodif_npz_path
    result = AssetReadiness(request=dict(request))

    def fail(message, code=2, missing=False):
        result.errors.append(message)
        result.code = result.code or code
        result.status = "not_ready"
        if missing:
            result.missing_assets.append(message)

    modes = request.get("modes", [])
    if not modes or any(m not in PRECISION_MODES for m in modes) or len(set(modes)) != len(modes):
        fail("Select one or more distinct canonical precision configurations.")
    if request.get("checkpoint_name") != "lavdf" or request.get("require_lavdf_hash", True) is not True:
        fail("The frozen LAV-DF checkpoint SHA-256 is required.", 7)
    fmt = request.get("feature_format")
    if fmt not in {"dimodif_npz", "shared_pt"}:
        fail("Unsupported feature format; choose dimodif_npz or shared_pt.")
    for name in ("run_id", "experiment_id"):
        value = request.get(name, "")
        if (name == "run_id" and not value) or (value and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", value)):
            fail(f"Enter a safe {name}: letters, digits, hyphens or underscores, maximum 96 characters.")
    run_id = request.get("run_id", "")
    runs_root = root / "artifacts/member3/runs"
    try:
        resolved_runs = authorized_path(root, runs_root)
        if not resolved_runs.is_relative_to(root.resolve()):
            fail("Run output must stay inside the selected repository.")
        if run_id and (runs_root / run_id).exists():
            fail("Run ID already exists; duplicate execution refused.")
    except (OSError, ValueError) as exc:
        fail(str(exc), 6)
    error = _device_error(request.get("device", ""))
    if error:
        fail(error)
    paths = {}
    for key, label, directory in (("upstream_root", "DiMoDif checkout", True),
                                  ("manifest", "Validation manifest", False),
                                  ("feature_root", "Feature root", True)):
        try:
            if not str(request.get(key, "")).strip():
                raise ValueError(f"{label}: not configured")
            path = authorized_path(root, request[key])
            if not (path.is_dir() if directory else path.is_file()):
                raise ValueError(f"{label} missing: {path}")
            paths[key] = path
            result.request[key] = str(path)
        except (OSError, ValueError) as exc:
            fail(str(exc), 6, True)
    watched = []
    upstream = paths.get("upstream_root")
    if upstream:
        checkpoint = upstream / "ckpt/dfd" / (CHECKPOINTS["lavdf"] + ".pth")
        for label, path in (("Checkpoint", checkpoint), ("Model configuration", checkpoint.with_suffix(".json")),
                            ("DiMoDif model source", upstream / "src/models.py")):
            try:
                path = authorized_path(root, path)
                if not path.is_relative_to(upstream) or not path.is_file():
                    raise ValueError(f"{label} missing or outside selected checkout: {path}")
                watched.append(path)
                if label == "Checkpoint" and checkpoint_digest(path) != DIMODIF_LAVDF_SHA256:
                    fail("Checkpoint SHA-256 mismatch; obtain the frozen LAV-DF weights.", 7)
                if label == "Model configuration":
                    cfg = load_json(path).get("config")
                    required = {"task", "max_length", "d_model", "nhead", "d_hid", "nlayers", "win_size", "feature_pyramid"}
                    if not isinstance(cfg, dict) or not required <= cfg.keys() or cfg.get("task") != "dfd":
                        fail("Model configuration lacks the required DiMoDif DFD architecture fields.")
            except (OSError, ValueError) as exc:
                fail(str(exc), 6, True)
    rows = []
    if "manifest" in paths:
        try:
            rows = load_jsonl(paths["manifest"])
            watched.append(paths["manifest"])
            for message in validate_manifest(rows, root / "contracts/schemas/manifest.schema.json"):
                fail(message)
            if any(row.get("split") != "val" for row in rows):
                fail("Use a validation-only manifest; held-out test execution is unavailable.")
            if any(str(row.get("dataset", "")).lower() != "lavdf" for row in rows):
                fail("This workbench supports the LAV-DF validation contract only.")
        except (OSError, ValueError) as exc:
            fail(f"Validation manifest: {exc}", 6)
    expected, found, missing = len(rows), 0, 0
    sample = None
    if rows and "feature_root" in paths and fmt in {"dimodif_npz", "shared_pt"} and not any(r.get("split") != "val" for r in rows):
        base = paths["feature_root"]
        for row in rows:
            try:
                if fmt == "dimodif_npz":
                    cache = dimodif_npz_path(base, "val", row["clip_id"])
                else:
                    if not row.get("feature_path"):
                        raise ValueError("manifest feature_path is missing")
                    cache = base / row["feature_path"]
                cache = authorized_path(root, cache)
                if not cache.is_relative_to(base):
                    raise ValueError("cache escapes the selected feature root")
                if not cache.is_file():
                    missing += 1
                    if len(result.missing_assets) < 20:
                        result.missing_assets.append(f"Missing cache for clip {row['clip_id']}: {cache}")
                    continue
                found += 1
                watched.append(cache)
                sample = sample or cache
            except (OSError, ValueError, KeyError) as exc:
                fail(f"Cache for clip {row.get('clip_id', '?')}: {exc}", 6)
        if missing:
            fail(f"Feature cache incomplete: expected {expected}, found {found}, missing {missing}.", 6)
        # Do not load all tensors: one sample through the real, bounded backend loader.
        if sample and not result.errors:
            try:
                expected_suffix = ".npz" if fmt == "dimodif_npz" else ".pt"
                if sample.suffix.lower() != expected_suffix:
                    raise ValueError(f"Selected {fmt} cache must use {expected_suffix}")
                feature_file_summary(sample)
                result.sample_inspected = True
            except Exception as exc:  # Safe loaders can raise pickle/archive/index errors for corrupt input.
                fail(f"Representative cache compatibility: {type(exc).__name__}: {exc}", 6)
    result.asset_counts = {"expected": expected, "found": found, "missing": missing}
    for path in watched:
        try:
            stat = path.stat()
            result.file_stamp.append((str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
        except OSError as exc:
            fail(f"Asset changed while checking: {path}: {exc}", 6)
    return result


def validate_experiment_readiness(root: Path, *, smoke: bool = False, **request):
    from .backend_status import validate_cached_feature_experiment, RunResult, _command_env
    result = validate_cached_feature_experiment(root, **request)
    if result.code or not smoke:
        return result
    import json
    import subprocess
    import sys
    import tempfile
    from .backend_status import _common_args
    # Use the existing predict CLI and runner. Output is temporary compatibility evidence.
    with tempfile.TemporaryDirectory(prefix="deeptrace-forward-") as directory:
        command = [sys.executable, "-m", "avfuse.member3_runtime.cli", "predict",
                   *_common_args(request), "--max-clips", "1", "--run-id", "readiness-sample",
                   "--output", str(Path(directory) / "sample.jsonl"), "--json"]
        try:
            proc = subprocess.run(command, cwd=root, env=_command_env(root), capture_output=True,
                                  text=True, timeout=300, check=False)
            payload = json.loads(proc.stdout)
            if (proc.returncode not in (0, 4) or not isinstance(payload, dict)
                    or payload.get("status") != "complete" or payload.get("published") is not True
                    or payload.get("n_clips") != 1 or payload.get("split") != "val"
                    or set(payload.get("completed_modes", [])) != set(request["modes"])):
                return RunResult(proc.returncode or 1, "Representative inference did not complete.\n" + proc.stderr,
                                 {"phase": "inference_smoke", "cli_launched": True, "status": "failed"})
        except subprocess.TimeoutExpired:
            return RunResult(124, "Representative inference exceeded 300 seconds.",
                             {"phase": "inference_smoke", "cli_launched": True, "status": "timeout"})
        except (OSError, ValueError) as exc:
            return RunResult(1, f"Representative inference failed: {exc}",
                             {"phase": "inference_smoke", "cli_launched": True, "status": "failed"})
    result.payload.update(smoke_passed=True, phase="inference_smoke")
    result.diagnostics += "\nOne real validation sample passed the selected modes; this does not guarantee full-cohort success."
    return result


def execute_validation_experiment(root: Path, **request):
    from .backend_status import run_cached_feature_experiment
    return run_cached_feature_experiment(root, **request)



def readiness_is_current(result) -> bool:
    if result.code or not result.payload.get("file_stamp"):
        return False
    try:
        for name, size, modified, changed in result.payload["file_stamp"]:
            stat = Path(name).stat()
            if (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != (size, modified, changed):
                return False
        return True
    except OSError:
        return False
