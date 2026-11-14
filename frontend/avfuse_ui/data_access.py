from __future__ import annotations

import json
import re
from io import BytesIO
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    from jsonschema import Draft202012Validator
except Exception:  # pragma: no cover - the project lists jsonschema as a base dependency
    Draft202012Validator = None

CONFIGS = (
    "reference",
    "low_precision",
    "restore_value_output",
    "restore_query",
)
FAKE_TYPES = ("audio_fake", "video_fake", "both_fake")
MANIFEST_TYPES = ("real", *FAKE_TYPES)


def parse_jsonl_text(text: str) -> list[dict[str, Any]]:
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
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected a JSON object in {path.name}")
    return obj


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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
        key = (row.get("run_id"), row.get("config"), row.get("clip_id"))
        if key in seen:
            errors.append(f"row {idx}: duplicate prediction for run/config/clip {key}")
        seen.add(key)
        if "checkpoint_sha256" in row:
            checkpoints.add(row["checkpoint_sha256"])
        if "label" in row or "ground_truth" in row:
            errors.append(f"row {idx}: prediction rows must not contain ground-truth labels")
    if len(checkpoints) > 1:
        errors.append("More than one checkpoint SHA-256 appears in this prediction batch")
    return errors


def validate_benchmarks(rows: list[dict[str, Any]], schema_path: Path) -> list[str]:
    return schema_errors(rows, schema_path)


def validate_metrics_report(obj: dict[str, Any]) -> list[str]:
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
        if not isinstance(block.get("per_type"), dict):
            errors.append(f"{cfg}.per_type must be an object")
    if obj.get("split") == "test" and not isinstance(obj.get("test_accesses_so_far"), int):
        errors.append("Test report should expose test_accesses_so_far")
    return errors


def manifest_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_split = Counter(str(r.get("split", "unknown")) for r in rows)
    by_type: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        by_type[str(row.get("split", "unknown"))][str(row.get("manipulation_type", "unknown"))] += 1
    source_groups = {(r.get("dataset"), r.get("source_id")) for r in rows if r.get("source_id") is not None}
    identity_groups = {(r.get("dataset"), r.get("identity_id")) for r in rows if r.get("identity_id") is not None}
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
        grouped[(str(row.get("split")), str(row.get("config")))].append(row)
    result = []
    for (split, config), block in sorted(grouped.items()):
        scores = [float(r["score_fake"]) for r in block if isinstance(r.get("score_fake"), (int, float))]
        result.append({
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
            run_dirs.update(p for p in run_root.glob("run-*") if p.is_dir())
            env_reports.update(p for p in run_root.rglob("environment_report.json") if p.is_file())
            stage_profiles.update(p for p in run_root.rglob("stage_profile.json") if p.is_file())
    return {
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


def _summarize_feature_payload(payload: Any, suffix: str) -> dict[str, Any]:
    if suffix == ".npz":
        expected = {"audio_tokens", "video_tokens"}
        shared_keys = expected.issubset(payload["keys"])
        dimodif_keys = {"audio_features", "video_features"}.issubset(payload["keys"])
        compatible = shared_keys or dimodif_keys
        return {"format": "NumPy archive", "keys": payload["keys"], "shapes": payload["shapes"],
                "dtypes": payload["dtypes"], "contract_compatible": compatible,
                "loader_format": "shared_pt-compatible NPZ keys" if shared_keys else
                                 ("dimodif_npz" if dimodif_keys else "unrecognized"),
                "note": ("The Member 3 feature loader accepts these keys with an explicit feature format." if compatible else
                         "Required keys were not found. Expected audio_tokens/video_tokens or audio_features/video_features.")}
    import torch
    if not isinstance(payload, dict):
        raise ValueError(".pt feature cache must contain a tensor dictionary")
    expected = ("audio_tokens", "video_tokens")
    missing = [k for k in expected if k not in payload]
    if missing:
        raise ValueError(f"Feature cache missing keys: {missing}")
    summary = {}
    for key in expected:
        tensor = payload[key]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{key} is not a tensor")
        summary[key] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                        "finite": bool(torch.isfinite(tensor).all())}
    return {"format": "PyTorch tensor dictionary", "keys": list(payload),
            "tensors": summary, "contract_compatible": True,
            "note": "Inspected only; no model inference was run."}


def feature_file_summary(path: Path) -> dict[str, Any]:
    """Inspect supported local cache files without running a model."""
    suffix = path.suffix.lower()
    if suffix == ".npz":
        import numpy as np
        with np.load(path, allow_pickle=False) as archive:
            payload = {"keys": list(archive.files),
                       "shapes": {k: list(archive[k].shape) for k in archive.files},
                       "dtypes": {k: str(archive[k].dtype) for k in archive.files}}
        return _summarize_feature_payload(payload, suffix)
    if suffix == ".pt":
        import torch
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError as exc:
            raise RuntimeError("This PyTorch version does not support safe weights_only loading") from exc
        return _summarize_feature_payload(payload, suffix)
    raise ValueError("Only .pt token caches and .npz feature archives can be inspected")


def feature_file_summary_bytes(content: bytes, filename: str) -> dict[str, Any]:
    """Inspect an uploaded feature file in memory; it is never persisted by this app."""
    suffix = Path(filename).suffix.lower()
    stream = BytesIO(content)
    if suffix == ".npz":
        import numpy as np
        with np.load(stream, allow_pickle=False) as archive:
            payload = {"keys": list(archive.files),
                       "shapes": {k: list(archive[k].shape) for k in archive.files},
                       "dtypes": {k: str(archive[k].dtype) for k in archive.files}}
        return _summarize_feature_payload(payload, suffix)
    if suffix == ".pt":
        import torch
        try:
            payload = torch.load(stream, map_location="cpu", weights_only=True)
        except TypeError as exc:
            raise RuntimeError("This PyTorch version does not support safe weights_only loading") from exc
        return _summarize_feature_payload(payload, suffix)
    raise ValueError("Only .pt and .npz feature files can be inspected")


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
