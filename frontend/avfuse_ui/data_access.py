from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    from jsonschema import Draft202012Validator
except Exception:  # pragma: no cover - the project lists jsonschema as a base dependency
    Draft202012Validator = None

from avfuse.shared.contracts import PRECISION_MODES

CONFIGS = tuple(PRECISION_MODES)
FAKE_TYPES = ("audio_fake", "video_fake", "both_fake")
MANIFEST_TYPES = ("real", *FAKE_TYPES)


def parse_jsonl_text(text: str) -> list[dict[str, Any]]:
    if len(text.encode("utf-8")) > 32 * 1024 * 1024:
        raise ValueError("JSONL exceeds 32 MiB limit")
    rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at line {line_no}: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL line {line_no} must contain an object")
        rows.append(value)
    return rows


def load_json(path: Path) -> dict[str, Any]:
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("JSON exceeds 32 MiB limit")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected a JSON object in {path.name}")
    return obj


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("JSONL exceeds 32 MiB limit")
    return parse_jsonl_text(path.read_text(encoding="utf-8"))


def schema_errors(rows: Iterable[dict[str, Any]], schema_path: Path) -> list[str]:
    if Draft202012Validator is None:
        return ["jsonschema is not installed; schema validation was skipped"]
    if not schema_path.is_file():
        return [f"Schema file not found: {schema_path}"]
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    errors: list[str] = []
    for idx, row in enumerate(rows, start=1):
        for err in validator.iter_errors(row):
            path = ".".join(str(part) for part in err.absolute_path)
            errors.append(f"row {idx}{' / ' + path if path else ''}: {err.message}")
            if len(errors) >= 50:
                errors.append("Further validation errors omitted")
                return errors
    return errors


def validate_manifest(rows: list[dict[str, Any]], schema_path: Path) -> list[str]:
    errors = schema_errors(rows, schema_path)
    seen: set[str] = set()
    source_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    identity_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    for idx, row in enumerate(rows, start=1):
        clip = row.get("clip_id")
        if isinstance(clip, str):
            if clip in seen:
                errors.append(f"row {idx}: duplicate clip_id {clip!r}")
            seen.add(clip)
        if all(k in row for k in ("dataset", "source_id", "split")):
            source_splits[(str(row["dataset"]), str(row["source_id"]))].add(str(row["split"]))
        if all(k in row for k in ("dataset", "identity_id", "split")):
            identity_splits[(str(row["dataset"]), str(row["identity_id"]))].add(str(row["split"]))
    for label, groups in (("source", source_splits), ("identity", identity_splits)):
        leaked = [key for key, splits in groups.items() if len(splits) > 1]
        if leaked:
            errors.append(f"{len(leaked)} {label}-group(s) occur in multiple splits; examples: {leaked[:3]}")
    if not rows:
        errors.append("Manifest is empty")
    return errors


def validate_predictions(rows: list[dict[str, Any]], schema_path: Path) -> list[str]:
    errors = schema_errors(rows, schema_path)
    seen: set[tuple[Any, Any, Any]] = set()
    checkpoints: set[Any] = set()
    for idx, row in enumerate(rows, start=1):
        key = tuple(str(row.get(k)) for k in ("run_id", "split", "checkpoint_sha256", "config", "clip_id"))
        if key in seen:
            errors.append(f"row {idx}: duplicate prediction for run/config/clip {key}")
        seen.add(key)
        if "checkpoint_sha256" in row:
            checkpoints.add(str(row["checkpoint_sha256"]))
        if "label" in row or "ground_truth" in row:
            errors.append(f"row {idx}: prediction rows must not contain ground-truth labels")

    return errors


def validate_benchmarks(rows: list[dict[str, Any]], schema_path: Path, *, run_dir: Path | None = None) -> list[str]:
    from avfuse.member3_runtime.artifacts import validate_benchmark_records
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row.get("run_id"))].append(row)
    errors = []
    if not rows:
        return ["No benchmark records processed"]
    for run_id, block in grouped.items():
        try:
            validate_benchmark_records(block, run_id=run_id, run_dir=run_dir, schema_path=schema_path)
        except (ValueError, RuntimeError, OSError) as exc:
            errors.append(f"Verification incomplete or failed: {exc}")
    return errors


def validate_metrics_report(obj: dict[str, Any]) -> list[str]:
    if not isinstance(obj, dict):
        return ["Expected a metrics object"]
    errors: list[str] = []
    if obj.get("split") not in ("val", "test"):
        errors.append("metrics.json must declare split='val' or split='test'")
    configs = obj.get("configs")
    if not isinstance(configs, dict) or not configs:
        errors.append("metrics.json has no non-empty 'configs' object")
        return errors
    for cfg, block in configs.items():
        if cfg not in CONFIGS:
            errors.append(f"Unknown precision configuration: {cfg}")
        if not isinstance(block, dict):
            errors.append(f"Configuration {cfg} is not an object")
            continue
        for metric in ("auroc", "auprc", "bce"):
            m = block.get(metric)
            if not isinstance(m, dict) or not all(k in m for k in ("value", "ci_low", "ci_high")):
                errors.append(f"{cfg}.{metric} is missing value/ci_low/ci_high")
            elif any(not isinstance(m[k], (int, float)) or isinstance(m[k], bool) or
                     not __import__("math").isfinite(m[k]) for k in ("value", "ci_low", "ci_high")):
                errors.append(f"{cfg}.{metric} must contain finite numeric values")
            elif m["ci_low"] > m["ci_high"]:
                errors.append(f"{cfg}.{metric} confidence interval is reversed")
        if not isinstance(block.get("per_type"), dict):
            errors.append(f"{cfg}.per_type must be an object")
    for cfg, block in configs.items():
        if not isinstance(block, dict):
            continue
        per_type = block.get("per_type", {})
        if isinstance(per_type, dict) and any(not isinstance(value, dict) for value in per_type.values()):
            errors.append(f"{cfg}.per_type entries must be objects")
    paired = obj.get("paired", [])
    if not isinstance(paired, list) or any(not isinstance(item, dict) or
            not isinstance(item.get("metrics", {}), dict) or
            any(not isinstance(value, dict) for value in item.get("metrics", {}).values())
            for item in paired):
        errors.append("paired must contain comparison objects with metric objects")
    if obj.get("split") == "test" and not isinstance(obj.get("test_accesses_so_far"), int):
        errors.append("Test report should expose test_accesses_so_far")
    return errors


def manifest_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_split = Counter(str(r.get("split", "unknown")) for r in rows)
    by_type: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        by_type[str(row.get("split", "unknown"))][str(row.get("manipulation_type", "unknown"))] += 1
    source_groups = {(str(r.get("dataset")), str(r.get("source_id"))) for r in rows if r.get("source_id") is not None}
    identity_groups = {(str(r.get("dataset")), str(r.get("identity_id"))) for r in rows if r.get("identity_id") is not None}
    return {
        "clips": len(rows),
        "splits": dict(by_split),
        "types": {split: dict(counts) for split, counts in by_type.items()},
        "source_groups": len(source_groups),
        "identity_groups": len(identity_groups),
        "feature_paths_present": sum(bool(r.get("feature_path")) for r in rows),
    }


def prediction_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("run_id")), str(row.get("checkpoint_sha256")), str(row.get("split")), str(row.get("config")))].append(row)
    result = []
    for (run_id, checkpoint, split, config), block in sorted(grouped.items()):
        scores = [float(r["score_fake"]) for r in block if isinstance(r.get("score_fake"), (int, float))]
        result.append({
            "run_id": run_id, "checkpoint_sha256": checkpoint,
            "split": split,
            "config": config,
            "clips": len(block),
            "mean_score_fake": sum(scores) / len(scores) if scores else None,
            "min_score_fake": min(scores) if scores else None,
            "max_score_fake": max(scores) if scores else None,
        })
    return result


def discover_files(root: Path) -> dict[str, list[Path]]:
    data = root / "data"
    artifacts = root / "artifacts"
    manifests = sorted({p for p in data.rglob("*manifest*.jsonl") if p.is_file()}) if data.exists() else []
    prediction_paths: set[Path] = set()
    benchmark_paths: set[Path] = set()
    if artifacts.exists():
        prediction_paths.update(p for p in artifacts.rglob("*.jsonl")
                                if p.is_file() and (p.parent.name == "predictions" or p.name == "predictions.jsonl" or p.name.startswith("predictions_")))
        benchmark_paths.update(p for p in artifacts.rglob("*.jsonl")
                               if p.is_file() and (p.parent.name == "benchmarks" or p.name == "benchmark_records.jsonl"))
    run_roots = [artifacts / "runs", artifacts / "member3" / "runs"]
    run_dirs: set[Path] = set()
    env_reports: set[Path] = set()
    stage_profiles: set[Path] = set()
    for run_root in run_roots:
        if run_root.exists():
            run_dirs.update(p.parent for p in run_root.rglob("run_manifest.json") if p.is_file())
            run_dirs.update(p for p in run_root.iterdir() if p.is_dir())
            env_reports.update(p for p in run_root.rglob("environment_report.json") if p.is_file())
            stage_profiles.update(p for p in run_root.rglob("stage_profile.json") if p.is_file())
    discovered = {
        "manifests": manifests,
        "predictions": sorted(prediction_paths),
        "benchmarks": sorted(benchmark_paths),
        "metrics": sorted(artifacts.rglob("metrics.json")) if artifacts.exists() else [],
        "run_dirs": sorted(run_dirs),
        "environment_reports": sorted(env_reports),
        "stage_profiles": sorted(stage_profiles),
        "shortcut_audits": sorted(artifacts.glob("reports/**/shortcut_audit.json")) if artifacts.exists() else [],
        "train_overlap_reports": sorted(artifacts.glob("reports/**/train_overlap.json")) if artifacts.exists() else [],
        "prediction_checks": sorted(artifacts.glob("reports/**/check*.json")) if artifacts.exists() else [],
        "shortcut_controls": sorted(artifacts.glob("reports/**/shortcut_control.json")) if artifacts.exists() else [],
        "ablation_reports": sorted((artifacts / "day2").glob("ablation_report.json")) if (artifacts / "day2").exists() else [],
        "attention_audits": sorted((artifacts / "day2").glob("attention_audit.json")) if (artifacts / "day2").exists() else [],
        "attention_memory_reports": sorted((artifacts / "dayA").glob("attention_memory.json")) if (artifacts / "dayA").exists() else [],
        "padding_audits": sorted((artifacts / "dayA").glob("padding_mask_audit.json")) if (artifacts / "dayA").exists() else [],
    }
    return {kind: [path for path in paths if path.resolve().is_relative_to(root.resolve())]
            for kind, paths in discovered.items()}


def dimodif_npz_inspection_path(
    root: Path,
    feature_root: str | Path,
    split: Any,
    clip_id: Any,
) -> Path:
    """Resolve the explicit DiMoDif NPZ layout without leaving the repository.

    The Member 3 loader uses <feature_root>/<split>/<clip_id>/features.npz
    for dimodif_npz and does not consult the manifest's planned .pt path.
    """
    if split not in ("val", "test"):
        raise ValueError("DiMoDif NPZ inspection supports only the val or test split")
    if not isinstance(clip_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", clip_id):
        raise ValueError("clip_id contains characters that are unsafe for a feature path")
    if not str(feature_root).strip():
        raise ValueError("Enter a non-empty DiMoDif feature root")

    repo_root = Path(root).expanduser().resolve()
    base = Path(feature_root).expanduser()
    if not base.is_absolute():
        base = repo_root / base
    target = (base / str(split) / clip_id / "features.npz").resolve()
    authorized_path(repo_root, base)
    if not target.is_relative_to(base.resolve()):
        raise ValueError("Feature symlink escapes its selected root")
    return authorized_path(repo_root, target)

MAX_ARTIFACT_BYTES = 32 * 1024 * 1024


def authorized_path(root: Path, value: str | Path) -> Path:
    """Server operator allowlists roots via AVFUSE_ASSET_ROOTS, never via browser."""
    import os
    path = Path(value).expanduser()
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    roots = [root.resolve(), *[Path(x).expanduser().resolve() for x in
             os.environ.get("AVFUSE_ASSET_ROOTS", "").split(os.pathsep) if x]]
    if not any(path.is_relative_to(base) for base in roots):
        raise ValueError("Path outside server-authorized roots (AVFUSE_ASSET_ROOTS)")
    return path


def feature_file_summary(path: Path) -> dict[str, Any]:
    from avfuse.member3_runtime.feature_loader import load_dimodif_features, load_shared_pt_features
    import zipfile
    if path.stat().st_size > MAX_ARTIFACT_BYTES:
        raise ValueError("Feature inspection limit is 32 MiB")
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            if sum(item.file_size for item in archive.infolist()) > MAX_ARTIFACT_BYTES:
                raise ValueError("Expanded feature archive exceeds 32 MiB")
    if path.suffix.lower() == ".npz":
        tensors = load_dimodif_features(path)
        keys = ["video_features", "audio_features"]
        fmt = "dimodif_npz"
    elif path.suffix.lower() == ".pt":
        tensors = load_shared_pt_features(path)
        keys = list(tensors)
        fmt = "shared_pt"
    else:
        raise ValueError("Only .npz and safe .pt caches are supported")
    return {"contract_compatible": True, "loader_format": fmt, "keys": keys,
            "shapes": {k.replace("_tokens", "_features") if fmt == "dimodif_npz" else k: list(v.shape) for k, v in tensors.items()},
            "tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in tensors.items()},
            "note": "Actual backend loader validation; no inference performed."}


def feature_file_summary_bytes(content: bytes, filename: str) -> dict[str, Any]:
    import tempfile
    if len(content) > MAX_ARTIFACT_BYTES:
        raise ValueError("Upload exceeds 32 MiB inspection limit")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / ("cache" + Path(filename).suffix.lower())
        path.write_bytes(content)
        return feature_file_summary(path)


def saved_result_table(root: Path, filename: str) -> list[dict[str, str]]:
    import csv
    allowed = {"test_overall_metrics.csv", "test_manipulation_metrics.csv",
               "test_paired_comparisons.csv", "validation_overall_metrics.csv",
               "cpu_model_runtime.csv", "cpu_model_timing_samples.csv"}
    if filename not in allowed:
        raise ValueError("Unknown saved result table")
    with (root / "docs/results" / filename).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if not re.fullmatch(r"[a-f0-9]{64}", row.get("source_sha256", "")):
            raise ValueError("Missing original result provenance")
    return rows


def shortcut_audit_summary(report: dict[str, Any]) -> dict[str, Any]:
    flags = report.get("flags", [])
    splits = report.get("splits", {})
    feature_count = sum(len(features) for groups in splits.values() if isinstance(groups, dict)
                        for features in groups.values() if isinstance(features, dict))
    return {"passed": report.get("passed"), "flag_count": len(flags),
            "split_count": len(splits), "feature_measurements": feature_count,
            "rule": report.get("rule", "threshold not specified"), "flags": flags}


def documented_validation_table(markdown: str) -> list[dict[str, Any]]:
    """Parse the repo's compact VAL_4MODE_RESULTS.md table if present."""
    rows = []
    pattern = re.compile(r"\|\s*(reference|low_precision|restore_value_output|restore_query)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|")
    for line in markdown.splitlines():
        match = pattern.search(line)
        if match:
            rows.append({"config": match.group(1), "auroc": float(match.group(2)),
                         "auprc": float(match.group(3)), "recall": float(match.group(4))})
    return rows


def documented_test_results(markdown: str) -> dict[str, Any]:
    """Parse Member 2's published TEST_RESULTS.md summary without treating it as a fresh evaluation."""
    rows: list[dict[str, Any]] = []
    mode_pattern = re.compile(
        r"^\|\s*(reference|low_precision|restore_value_output|restore_query)\s*\|"
        r"\s*([0-9.]+)\s*\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]\s*\|"
        r"\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|$"
    )
    for line in markdown.splitlines():
        match = mode_pattern.match(line.strip())
        if not match:
            continue
        rows.append({
            "config": match.group(1),
            "auroc": float(match.group(2)),
            "auroc_ci_low": float(match.group(3)),
            "auroc_ci_high": float(match.group(4)),
            "auprc": float(match.group(5)),
            "bce": float(match.group(6)),
            "recall": float(match.group(7)),
            "real_fpr": float(match.group(8)),
        })

    metadata: dict[str, Any] = {}
    tick = r"\x60"
    run_match = re.search(r"Predictions:.*?run\s+" + tick + r"([^" + tick + r"]+)" + tick, markdown)
    if run_match:
        metadata["run_id"] = run_match.group(1)
    manifest_match = re.search(
        r"Manifest:\s*" + tick + r"([^" + tick + r"]+)" + tick + r"\s*\(sha\s+" + tick + r"([^" + tick + r"]+)" + tick + r"\)",
        markdown,
    )
    if manifest_match:
        metadata["manifest_name"] = manifest_match.group(1)
        metadata["manifest_sha_prefix"] = manifest_match.group(2)
    threshold_match = re.search(r"Threshold:\s*([0-9.]+)", markdown)
    if threshold_match:
        metadata["threshold"] = float(threshold_match.group(1))
    cohort_match = re.search(
        tick + r"split == test" + tick + r":\s*([\d,]+)\s+real,\s*([\d,]+)\s+audio_fake,\s*"
        r"([\d,]+)\s+video_fake,\s*([\d,]+)\s+both_fake,\s*([\d,]+)\s+source groups",
        markdown,
    )
    if cohort_match:
        counts = [int(value.replace(",", "")) for value in cohort_match.groups()]
        metadata["split_counts"] = dict(zip(("real", "audio_fake", "video_fake", "both_fake"), counts[:4]))
        metadata["total_test_videos"] = sum(counts[:4])
        metadata["source_groups"] = counts[4]
    bootstrap_match = re.search(r"Bootstrap:\s*([\d,]+)\s+resamples,\s*seed\s+(\d+)", markdown)
    if bootstrap_match:
        metadata["bootstrap_resamples"] = int(bootstrap_match.group(1).replace(",", ""))
        metadata["bootstrap_seed"] = int(bootstrap_match.group(2))
    seal_match = re.search(r"Test seal:\s*opened once on\s+(\d{4}-\d{2}-\d{2})", markdown)
    if seal_match:
        metadata["test_opened_date"] = seal_match.group(1)
    ledger_match = re.search(r"ledger has\s+(\d+)\s+lines", markdown)
    if ledger_match:
        metadata["ledger_lines"] = int(ledger_match.group(1))
    attempts_match = re.search(r"inference took\s+(\d+)\s+attempts", markdown)
    if attempts_match:
        metadata["inference_attempts"] = int(attempts_match.group(1))

    per_type: list[dict[str, Any]] = []
    in_per_type = False
    for line in markdown.splitlines():
        if "Reference, per fake type vs real:" in line:
            in_per_type = True
            continue
        if in_per_type and not line.strip():
            break
        if in_per_type:
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) == 3 and cells[0] in FAKE_TYPES:
                try:
                    per_type.append({
                        "manipulation_type": cells[0],
                        "auroc": float(cells[1]),
                        "recall": float(cells[2]),
                    })
                except ValueError:
                    continue

    paired_bce: list[dict[str, Any]] = []
    in_paired = False
    paired_pattern = re.compile(
        r"^-\s*([a-z_]+)\s+vs\s+([a-z_]+):\s*([+-]?[0-9.]+)\s*"
        r"\[\s*([+-]?[0-9.]+)\s*,\s*([+-]?[0-9.]+)\s*\]$"
    )
    for line in markdown.splitlines():
        if line.strip() == "Paired BCE differences (same videos):":
            in_paired = True
            continue
        if in_paired and not line.strip().startswith("- "):
            if paired_bce:
                break
        if in_paired:
            match = paired_pattern.match(line.strip())
            if match:
                paired_bce.append({
                    "contrast": f"{match.group(1)} vs {match.group(2)}",
                    "delta_bce": float(match.group(3)),
                    "ci_low": float(match.group(4)),
                    "ci_high": float(match.group(5)),
                })

    return {
        "rows": rows,
        "metadata": metadata,
        "per_type": per_type,
        "paired_bce": paired_bce,
    }


def documented_benchmark_summary(markdown: str) -> dict[str, Any] | None:
    """Parse only the compact typical-workload benchmark figures stated in the Day 9 report."""
    match = re.search(
        r"Verified typical p50 ms:\s*"
        r"reference\s+([0-9.]+);\s*"
        r"low_precision\s+([0-9.]+);\s*"
        r"restore_value_output\s+([0-9.]+);\s*"
        r"restore_query\s+([0-9.]+)",
        markdown,
    )
    if not match:
        return None
    p50 = {
        config: float(value)
        for config, value in zip(CONFIGS, match.groups())
    }
    value_match = re.search(r"Typical E_value\s*=\s*([+\-−]?[0-9.]+)\s*ms", markdown)
    query_match = re.search(r"Typical E_query\s*=\s*([+\-−]?[0-9.]+)\s*ms", markdown)

    def parse_signed(value: str) -> float:
        return float(value.replace("−", "-"))

    return {
        "workload": "typical",
        "scope": "cached_feature_model",
        "device": "cpu",
        "backend": "sdpa",
        "p50_ms": p50,
        "extra_cost_ms": {
            "value_output_vs_low_precision": parse_signed(value_match.group(1)) if value_match else None,
            "query_vs_low_precision": parse_signed(query_match.group(1)) if query_match else None,
        },
    }


def documented_shortcut_table(markdown: str) -> list[dict[str, Any]]:
    """Parse the repository's committed whole-video shortcut summary, not raw audit JSON."""
    rows: list[dict[str, Any]] = []
    pattern = re.compile(r"^\|\s*([^|]+?)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([^|]+?)\s*\|$")
    for line in markdown.splitlines():
        normalized = line.replace("**", "").strip()
        match = pattern.match(normalized)
        if match and match.group(1).strip().lower() != "feature":
            name = re.sub(r"\*", "", match.group(1)).strip()
            rows.append({"feature": name, "val_shortcut_auroc": float(match.group(2)),
                         "test_shortcut_auroc": float(match.group(3)),
                         "documented_verdict": re.sub(r"\*", "", match.group(4)).strip()})
    return rows


def safe_read_jsonl_files(paths: Iterable[Path]) -> list[dict[str, Any]]:
    return [row for path in paths for row in load_jsonl(path)]


def human_bytes(value: int | float | None) -> str:
    if value is None:
        return "Not measured"
    value = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:,.1f} {unit}"
        value /= 1024
    return f"{value:,.1f} GiB"
