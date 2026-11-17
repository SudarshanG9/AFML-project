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


def load_experiment_results(root: Path, run_directory: str | Path) -> dict[str, Any]:
    """Load a single run with explicit identity guards, never aggregate arbitrary files."""
    from .data_access import validate_predictions, validate_benchmarks, validate_metrics_report
    from .run_inspection import optional_metadata, manifest_structure_valid
    run = authorized_path(root, run_directory)
    if not run.is_relative_to((root / 'artifacts').resolve()) or not run.is_dir():
        raise ValueError('Choose a discovered run directory under repository artifacts.')
    out: dict[str, Any] = dict(run_dir=str(run), errors=[], predictions=[], benchmarks=[], metrics=None,
                              available_artifacts=[], completion_recorded=False)
    for name in ('execution_summary.json', 'failure_summary.json', 'run_manifest.json',
                 'resolved_config.json', 'provenance.json', 'environment_report.json', 'stage_profile.json'):
        path = authorized_path(root, run / name)
        if not path.is_relative_to(run):
            out['errors'].append(f'{name} escapes the run directory'); continue
        value, error = optional_metadata(path)
        out[name] = value
        if error: out['errors'].append(error)
        if path.is_file(): out['available_artifacts'].append(path)
    summary = out.get('execution_summary.json') or {}
    run_id = summary.get('run_id')
    sha = summary.get('checkpoint_sha256')
    out['run_id'] = run_id or run.name
    out['split'] = summary.get('split')
    out['requested_modes'] = summary.get('requested_modes', [])
    manifest = out.get('run_manifest.json')
    out['completion_recorded'] = bool(summary.get('status') == 'complete'
        and summary.get('dataset_complete') is True and out['split'] == 'val'
        and manifest_structure_valid(manifest) and manifest.get('run_id') == run_id
        and not out.get('failure_summary.json') and not (run / 'failure_summary.json').exists())
    for name, key in (('predictions.jsonl', 'predictions'), ('benchmark_records.jsonl', 'benchmarks')):
        path = authorized_path(root, run / name)
        if not path.is_relative_to(run):
            out['errors'].append(f'{name} escapes the run directory'); continue
        if not path.is_file(): continue
        out['available_artifacts'].append(path)
        try:
            rows = load_jsonl(path)
            errors = (validate_predictions(rows, root / 'contracts/schemas/prediction.schema.json')
                      if key == 'predictions' else validate_benchmarks(rows, root / 'contracts/schemas/benchmark.schema.json', run_dir=run))
            if not rows: errors.append('No records')
            if summary.get('data_origin') != 'real_cached_features':
                errors.append('Synthetic or unspecified origin: scores/timings are not detector evidence.')
            if not run_id or not sha or out['split'] != 'val':
                errors.append('Validation run identity is not established by execution_summary.json.')
            if any(row.get('run_id') != run_id for row in rows): errors.append('Records belong to another run ID.')
            if key == 'predictions':
                if any(row.get('split') != 'val' or row.get('checkpoint_sha256') != sha for row in rows):
                    errors.append('Prediction split/checkpoint differs from the selected run.')
                pred = summary.get('predictions') or {}
                if not isinstance(pred, dict):
                    pred = {}
                    errors.append('Invalid prediction summary structure.')
                if pred.get('n_rows') is not None and len(rows) != pred['n_rows']:
                    errors.append('Prediction row count differs from execution summary.')
                if summary.get('predictions_sha256') and checkpoint_digest(path) != summary['predictions_sha256']:
                    errors.append('Predictions differ from the execution summary hash.')
            else:
                if any(row.get('workload', {}).get('checkpoint_sha256') != sha for row in rows):
                    errors.append('Benchmark checkpoint differs from the selected run.')
            if errors:
                out['errors'].extend(f'{name}: {error}' for error in errors)
            else:
                out[key] = rows
        except (OSError, ValueError, RuntimeError) as exc:
            out['errors'].append(f'{name}: {exc}')
    out['available_modes'] = sorted({row['config'] for row in out['predictions']})
    out['missing_modes'] = [mode for mode in __import__('avfuse.shared.contracts', fromlist=['PRECISION_MODES']).PRECISION_MODES
                            if mode not in out['available_modes']]
    # Only reports with a frontend evaluation receipt bound to these exact predictions are associated.
    evaluation_root = root / 'artifacts/member3/evaluations' / run.name
    if evaluation_root.is_dir():
        for receipt_path in sorted(evaluation_root.glob('*/receipt.json'), reverse=True):
            try:
                receipt_path = authorized_path(root, receipt_path)
                receipt = load_json(receipt_path)
                pred_path = run / 'predictions.jsonl'
                metric_path = authorized_path(root, receipt_path.parent / 'metrics.json')
                if (receipt.get('run_id') != run_id or receipt.get('split') != 'val'
                        or receipt.get('predictions_sha256') != checkpoint_digest(pred_path)
                        or receipt.get('checkpoint_sha256') != sha
                        or receipt.get('metrics_sha256') != checkpoint_digest(metric_path)):
                    continue
                report = load_json(metric_path)
                if validate_metrics_report(report) or report.get('split') != 'val': continue
                if report.get('manifest_sha256') != summary.get('features', {}).get('manifest_sha256'): continue
                out.update(metrics=report, evaluation_receipt=receipt, evaluation_path=metric_path)
                out['available_artifacts'].extend([metric_path, receipt_path])
                break
            except (OSError, ValueError):
                continue
    return out


def evaluate_validation_results(root: Path, run_directory: str | Path, protocol: str):
    """Invoke existing Member 2 evaluation, without calibration or test access."""
    import json
    import subprocess
    import sys
    import uuid
    from .backend_status import RunResult, _command_env
    from avfuse.member2_data.predictions import by_config, join
    from .data_access import validate_metrics_report
    details = {'phase': 'evaluation_preflight', 'cli_launched': False, 'status': 'not_ready'}
    try:
        result = load_experiment_results(root, run_directory)
        if not result['completion_recorded'] or not result['predictions']:
            raise ValueError('Evaluation requires completed validation inference and valid predictions.')
        summary = result['execution_summary.json']
        if summary.get('data_origin') != 'real_cached_features':
            raise ValueError('Synthetic/development artifacts are not detector evaluation evidence.')
        manifest = authorized_path(root, summary['features']['manifest'])
        rows = load_jsonl(manifest)
        errors = validate_manifest(rows, root / 'contracts/schemas/manifest.schema.json')
        if errors or not rows or any(row.get('split') != 'val' for row in rows):
            raise ValueError('Evaluation requires a schema-valid validation-only manifest with labels.')
        if checkpoint_digest(manifest) != summary['features']['manifest_sha256']:
            raise ValueError('Manifest bytes differ from the inference run.')
        protocol_path = authorized_path(root, protocol)
        frozen = load_json(protocol_path)
        release = load_json(root / 'docs/member1/dayC/releases/M2-REL-20261113-01/m2_release.json')
        if (frozen.get('calibration_split') != 'val' or frozen.get('calibration_config') != 'reference'
                or frozen.get('threshold') != release['fixed_threshold']
                or frozen.get('manifest_sha256') != checkpoint_digest(manifest)
                or frozen.get('checkpoint_sha256') != summary['checkpoint_sha256']):
            raise ValueError('Frozen protocol must match this exact validation manifest/checkpoint and the original validation-fixed threshold. No recalibration is performed.')
        for block in by_config(result['predictions']).values():
            joined = join(rows, block, 'val')
            if {row['label'] for row in joined} != {0, 1}:
                raise ValueError('Label-grounded evaluation requires real and fake validation examples.')
        # Keep derived evaluation outside the finalized inference bundle.
        output = authorized_path(root, root / 'artifacts/member3/evaluations' / Path(run_directory).name / uuid.uuid4().hex)
        if not output.is_relative_to(root.resolve()):
            raise ValueError('Evaluation output must stay in the repository.')
        prediction_path = Path(result['run_dir']) / 'predictions.jsonl'
        command = [sys.executable, str(root / 'scripts/m2_evaluate.py'), 'evaluate', '--manifest', str(manifest),
                   '--predictions', str(prediction_path), '--protocol', str(protocol_path), '--split', 'val',
                   '--n-boot', '1000', '--seed', '0', '--out', str(output)]
        input_hashes = {str(path): checkpoint_digest(path) for path in (manifest, prediction_path, protocol_path)}
    except (OSError, ValueError, KeyError) as exc:
        return RunResult(2, f'Label-grounded evaluation not performed: {exc}', details)
    details.update(phase='evaluation', cli_launched=True)
    try:
        proc = subprocess.run(command, cwd=root, env=_command_env(root), capture_output=True, text=True,
                              timeout=1800, check=False)
        if proc.returncode:
            return RunResult(proc.returncode, proc.stdout + '\n' + proc.stderr, {**details, 'status': 'failed'})
        report = load_json(output / 'metrics.json')
        if (validate_metrics_report(report) or report.get('split') != 'val'
                or report.get('manifest_sha256') != input_hashes[str(manifest)]
                or report.get('threshold') != frozen['threshold']
                or report.get('n_boot') != 1000 or report.get('seed') != 0
                or set(report['configs']) != set(result['available_modes'])
                or any(checkpoint_digest(Path(path)) != sha for path, sha in input_hashes.items())):
            raise ValueError('Evaluator output or inputs changed; report cannot be associated with this run.')
        receipt = dict(run_id=result['run_id'], split='val', checkpoint_sha256=summary['checkpoint_sha256'],
                       predictions_sha256=input_hashes[str(prediction_path)], manifest_sha256=input_hashes[str(manifest)],
                       protocol_sha256=input_hashes[str(protocol_path)], metrics_sha256=checkpoint_digest(output / 'metrics.json'))
        (output / 'receipt.json').write_text(json.dumps(receipt, indent=2))
        return RunResult(0, 'Member 2 label-grounded validation evaluation completed.', {**details, 'status': 'complete', 'output': str(output)})
    except subprocess.TimeoutExpired:
        return RunResult(124, 'Evaluation exceeded 1800 seconds; evaluation completion is not established.', {**details, 'status': 'timeout'})
    except (OSError, ValueError) as exc:
        return RunResult(1, f'Evaluation report unavailable: {exc}', {**details, 'status': 'failed'})


def available_validation_manifests(root: Path, paths: list[Path]) -> list[Path]:
    """Register only actual, schema-valid LAV-DF validation manifests; no asset inference."""
    available = []
    for path in paths:
        try:
            path = authorized_path(root, path)
            rows = load_jsonl(path)
            if (rows and all(row.get('split') == 'val' and str(row.get('dataset', '')).lower() == 'lavdf' for row in rows)
                    and not validate_manifest(rows, root / 'contracts/schemas/manifest.schema.json')):
                available.append(path)
        except (OSError, ValueError):
            continue
    return available
