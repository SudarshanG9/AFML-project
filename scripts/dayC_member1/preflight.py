"""Read-only C0 baseline checks with a new, exclusive report as the only output.

No inference, calibration, held-out enumeration, cache loading or Git writes.
Later phase arguments fail closed until those phases are implemented/reviewed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys

from jsonschema.exceptions import ValidationError

from avfuse.member1_fusion.dayc_guards import (
    ensure_inside_repo, read_strict_json, require_allowed_split, require_new_file,
    require_sha256, sha256_file, write_new_json,
)

MODES = ["reference", "low_precision", "restore_value_output", "restore_query"]
CATEGORIES = {"real": 300, "audio_fake": 300, "video_fake": 300, "both_fake": 300}
CHECKPOINT = "e1e59006d1bb6ad36a3859af5486a6649ed41bba23258b7c41e24f873be7948e"
CORE_HASHES = {
    "source_precision": "a0e54283deb806e5541b80f162354c22ed85f48582917e603eed10690a997cd9",
    "source_attention": "8ef4f1f85c48963d65dc99ec415ab34a170d493fe51336635d221c396fee8578",
    "source_dimodif_intervention": "3a0fe003ca3081884f5b38b362145df43a8d3cd3cdd9bee4a0db05e69bb131cf",
    "source_dimodif_adapter": "256e5620a00f8c4cb846c30100d1870d2c4c9db5c59f7f6a0d1d25eb8a130ae0",
}
INPUT_NAMES = {
    "checkpoint", "model_config", "upstream_source", "canonical_manifest",
    "validation_manifest", "ready_val_manifest", "calibration", "dayb_audit",
    "dayb_context", "environment_reference", "precision_modes",
    "source_dayB_inference", "source_dayB_model_benchmark",
    *("predictions_" + mode for mode in MODES[:3]), *CORE_HASHES,
}


def _read_val_manifest(path: Path) -> list[dict]:
    from avfuse.member2_data.manifest import verify_manifest
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                # Strict JSON also rejects duplicate keys/nonfinite numbers.
                from avfuse.member1_fusion.dayc_guards import _unique_object
                row = json.loads(line, object_pairs_hook=_unique_object)
                json.dumps(row, allow_nan=False)
                require_allowed_split(row.get("split"), phase="baseline", test_approved=False)
                rows.append(row)
    verify_manifest(rows)
    if len(rows) != 1200 or Counter(r["manipulation_type"] for r in rows) != CATEGORIES:
        raise ValueError("Validation requires exactly 1,200 clips and 300 per category")
    return rows


def _reconcile_val_manifests(validation: list[dict], ready: list[dict]) -> set[str]:
    baseline = {row["clip_id"]: row for row in validation}
    if len(baseline) != 1200 or {r["clip_id"] for r in ready} != set(baseline):
        raise ValueError("Ready and validation clip ID sets differ")
    for row in ready:
        for key in ("dataset", "source_id", "identity_id", "split", "label", "manipulation_type"):
            if row[key] != baseline[row["clip_id"]][key]:
                raise ValueError(f"Ready manifest changes {key} for {row['clip_id']}")
    return set(baseline)


def _check_feature_paths(root: Path, ready: list[dict]) -> dict:
    unresolved, resolved = [], 0
    for row in ready:
        require_allowed_split(row.get("split"), phase="baseline", test_approved=False)
        try:
            path = ensure_inside_repo(root, Path(row["feature_path"]), must_exist=True)
            if not path.is_file() or row["clip_id"] not in path.parts:
                raise ValueError("Feature path is not a regular file linked to its clip ID")
            resolved += 1
        except (OSError, ValueError) as exc:
            unresolved.append({"clip_id": row["clip_id"], "feature_path": row["feature_path"],
                               "reason": type(exc).__name__})
    return {"checked_validation_paths": len(ready), "existing_local_paths": resolved,
            "unresolved_paths": unresolved, "feature_file_contents_read": False,
            "external_paths_opened": False}


def _check_predictions(path: Path, mode: str, ids: set[str], checkpoint: str) -> dict:
    from avfuse.member2_data.predictions import read_predictions
    rows = read_predictions(path)
    if len(rows) != 1200 or len({r["clip_id"] for r in rows}) != 1200:
        raise ValueError("Prediction count or uniqueness differs from 1,200")
    if {r["clip_id"] for r in rows} != ids:
        raise ValueError("Prediction and validation manifest IDs differ")
    if len({r["run_id"] for r in rows}) != 1:
        raise ValueError("Prediction file combines multiple runs")
    for row in rows:
        require_allowed_split(row["split"], phase="baseline", test_approved=False)
        if row["config"] != mode or row["checkpoint_sha256"] != checkpoint:
            raise ValueError("Prediction config or checkpoint differs")
        if not math.isfinite(row["score_fake"]) or not 0 <= row["score_fake"] <= 1:
            raise ValueError("Prediction score must be finite in [0,1]")
    return {"rows": len(rows), "unique_ids": len(ids), "config": mode,
            "split": "val", "checkpoint_sha256": checkpoint, "labels_in_scores": False}


def _environment() -> dict:
    import torch
    versions = {name: importlib.metadata.version(name) for name in
                ("numpy", "jsonschema", "torch", "scikit-learn", "matplotlib", "pytest")}
    return {"python": sys.version, "python_executable": sys.executable,
            "platform": platform.platform(), "versions": versions,
            "device": "cpu", "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "mha_fastpath_enabled": torch.backends.mha.get_fastpath_enabled(),
            "sdpa_available": hasattr(torch.nn.functional, "scaled_dot_product_attention")}


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True,
                            env=dict(os.environ, GIT_OPTIONAL_LOCKS="0"), check=True)
    return result.stdout.rstrip("\n")


def build_baseline_report(root: Path, spec_path: Path, *, phase: str = "baseline") -> dict:
    root = Path(root).resolve(strict=True)
    report = {"schema_id": "avfuse-member1-dayc-preflight/1.0", "phase": phase,
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "repo_root": str(root),
              "status": "BLOCKED", "checks": [], "blockers": [], "hashes": {},
              "verified_scope": "file_bytes; NOT a whole-feature-cache digest",
              "model_inference_performed": False, "test_feature_access": False,
              "canonical_manifest_rows_inspected": False, "human_approval_generated": False}

    def check(name, operation):
        try:
            value = operation()
            evidence = str(value) if isinstance(value, Path) else value
            if isinstance(value, list):
                evidence = {"count": len(value), "details": "validated in memory; rows not duplicated in report"}
            report["checks"].append({"name": name, "status": "PASS", "evidence": evidence})
            return value
        except (OSError, ValueError, TypeError, KeyError, ImportError,
                subprocess.CalledProcessError, ValidationError) as exc:
            report["checks"].append({"name": name, "status": "BLOCKED",
                                     "reason": f"{type(exc).__name__}: {exc}"})
            report["blockers"].append(name)
            return None

    # No later-phase artifacts or real data are opened by this C0 implementation.
    if phase != "baseline":
        report["blockers"].append("Later phase requires its human gate and reviewed implementation")
        return report

    def root_check():
        for rel in (".git", "pyproject.toml", "src", "scripts", "tests/member1"):
            ensure_inside_repo(root, Path(rel), must_exist=True)
        return {"branch": _git(root, "branch", "--show-current"),
                "head": _git(root, "rev-parse", "HEAD"),
                "status_short": _git(root, "status", "--short")}
    git = check("repository", root_check)
    spec_path = check("spec_path", lambda: ensure_inside_repo(root, spec_path, must_exist=True))
    if spec_path is None:
        return report
    spec = check("spec_json", lambda: read_strict_json(spec_path))
    if not isinstance(spec, dict):
        report["blockers"].append("Specification must be a JSON object")
        return report
    report["spec_sha256"] = sha256_file(spec_path)

    def spec_check():
        required = {"original_days": [8, 9, 10], "ownership": "member1_only",
                    "attention_backend": "sdpa", "batch_size": 1,
                    "precision": "symmetric_dynamic_per_tensor_fake_INT8_in_floating_point",
                    "quantized_boundaries": ["Q", "K", "V", "attention_output"],
                    "restore_value_output": {"Q": "fake_INT8", "K": "fake_INT8",
                                             "V": "FP32", "attention_output": "FP32"},
                    "restore_query": {"Q": "FP32", "K": "fake_INT8",
                                      "V": "fake_INT8", "attention_output": "fake_INT8"},
                    "bootstrap_seed": 0, "bootstrap_replicates": 1000}
        if any(spec.get(key) != value for key, value in required.items()):
            raise ValueError("Scientific definitions differ from the Day C runbook")
        if (spec["schema_id"] != "avfuse-member1-dayc-spec/1.0"
                or spec["modes_ordered"] != MODES or spec["checkpoint_sha256"] != CHECKPOINT
                or spec["validation_expected_total"] != 1200
                or spec["validation_expected_by_type"] != CATEGORIES
                or set(spec["local_inputs"]) != INPUT_NAMES):
            raise ValueError("Specification does not match C0 inputs and four-mode cohort contract")
        if git is None or any(git[k] != spec["expected_baseline"][k] for k in ("branch", "head")):
            raise ValueError("Branch/HEAD differs from the reviewed baseline")
        return "Four modes, 1,200 validation clips, expected branch/HEAD"
    if check("specification", spec_check) is None:
        return report

    paths = {}
    for name, entry in spec["local_inputs"].items():
        def hash_input(name=name, entry=entry):
            path = ensure_inside_repo(root, Path(entry["path"]), must_exist=True)
            if entry["verified_scope"] != "file_bytes":
                raise ValueError("Input scope must explicitly be file_bytes")
            require_sha256(path, entry["sha256"])
            paths[name] = path
            report["hashes"][name] = {"path": str(path.relative_to(root)),
                                      "sha256": entry["sha256"], "verified_scope": "file_bytes"}
            return report["hashes"][name]
        check("input_" + name, hash_input)

    def accepted_provenance():
        audit = read_strict_json(paths["dayb_audit"])
        context = read_strict_json(paths["dayb_context"])
        if audit["technical_audit"] != "PASS":
            raise ValueError("Prior Day B technical audit did not pass")
        for name, digest in CORE_HASHES.items():
            require_sha256(paths[name], digest)
        require_sha256(paths["checkpoint"], CHECKPOINT)
        assets = {Path(asset["path"]).name: asset["sha256"] for asset in context["runtime_assets"]}
        for name in ("checkpoint", "model_config", "upstream_source"):
            require_sha256(paths[name], assets[paths[name].name])
        for mode in MODES[:3]:
            require_sha256(paths["predictions_" + mode], audit["prediction_sha256"][mode])
        require_sha256(paths["calibration"],
                       "358c037242f6980fed2a8d0ff0ef830f17b4789676b391c8f6dff245842b37dd")
        environment = read_strict_json(paths["environment_reference"])
        require_sha256(paths["canonical_manifest"], environment["canonical_manifest_sha256"])
        require_sha256(paths["validation_manifest"], environment["validation_manifest_sha256"])
        require_sha256(paths["ready_val_manifest"],
                       "c79ad908517878ebf0f3aa0e0dac053a01d9d5c4eb264583db26096985890481")
        return {"technical_audit": "PASS", "human_acknowledgement_pending":
                audit["acknowledgement_pending"], "code_and_weights_match_DayB": True}
    prior = check("accepted_dayb_provenance", accepted_provenance)
    if prior and prior["human_acknowledgement_pending"]:
        report["blockers"].append("Day B scientific acknowledgement / C0 review pending")

    def precision_check():
        definitions = read_strict_json(paths["precision_modes"])
        expected = {"reference": dict.fromkeys(("Q", "K", "V", "attention_output"), "FP32"),
                    "low_precision": dict.fromkeys(("Q", "K", "V", "attention_output"), "fake_INT8"),
                    "restore_value_output": spec["restore_value_output"],
                    "restore_query": spec["restore_query"]}
        for mode in MODES:
            if {key: definitions[mode][key] for key in expected[mode]} != expected[mode]:
                raise ValueError(f"Precision definition mismatch: {mode}")
        return expected
    check("four_precision_definitions", precision_check)
    validation = check("validation_manifest", lambda: _read_val_manifest(paths["validation_manifest"]))
    ready = check("ready_validation_manifest", lambda: _read_val_manifest(paths["ready_val_manifest"]))
    ids = None
    if validation is not None and ready is not None:
        ids = check("validation_cohort_reconciliation", lambda: sorted(_reconcile_val_manifests(validation, ready)))
        from avfuse.member2_data.splits import split_audit
        report["validation_split_audit"] = split_audit(validation)
        report["cross_split_audit"] = {
            "status": "prior_Member2_audit_only_not_independently_recomputed_in_C0",
            "reason": "Full canonical rows remain unopened; val-only audit cannot prove cross-split separation"}
    if ready is not None:
        report["cache_readiness"] = _check_feature_paths(root, ready)
        if report["cache_readiness"]["unresolved_paths"]:
            report["blockers"].append("Ready manifest does not resolve all 1,200 project-local caches")
    if ids is not None:
        for mode in MODES[:3]:
            check("predictions_" + mode, lambda mode=mode: _check_predictions(
                paths["predictions_" + mode], mode, set(ids), CHECKPOINT))
        report["validation_cohort"] = {"total": len(ids), "by_type": dict(CATEGORIES)}

    def calibration_check():
        calibration = read_strict_json(paths["calibration"])
        if (calibration["calibration_split"] != "val"
                or calibration["calibration_config"] != "reference"
                or calibration["checkpoint_sha256"] != CHECKPOINT
                or calibration["manifest_sha256"] != report["hashes"]["canonical_manifest"]["sha256"]
                or calibration["threshold"] != 0.6627789735794068):
            raise ValueError("Provisional Day B calibration provenance differs")
        return {"threshold": calibration["threshold"], "status": "PROVISIONAL_NOT_DAY9_FREEZE"}
    check("calibration", calibration_check)
    check("query_output_absent", lambda: require_new_file(
        ensure_inside_repo(root, Path("artifacts/predictions/restore_query_val.jsonl"))) or "absent")

    local_environment = check("local_environment", _environment)
    if local_environment:
        report["environment"] = local_environment
        def matching_environment():
            accepted = read_strict_json(paths["environment_reference"])["reference_environment"]
            mismatches = {key: {"accepted": accepted[key], "local": local_environment[key]}
                          for key in ("python", "platform", "versions", "torch_threads",
                                      "torch_interop_threads", "cuda_available")
                          if accepted[key] != local_environment[key]}
            report["runtime_differences"] = mismatches
            if mismatches:
                raise ValueError("Local runtime differs from approved Day B CPU runtime; human review required")
            return "Accepted runtime matches; no inference authorized by this check"
        check("runtime_comparability", matching_environment)
    report["blockers"].append("C0 human acknowledgement not yet recorded; no Day 8 authorization")
    report["blockers"] = list(dict.fromkeys(report["blockers"]))
    report["status"] = "BLOCKED" if report["blockers"] else "PASS"
    return report



def _new_report_path(root: Path, requested: Path) -> Path:
    output = ensure_inside_repo(root, requested)
    try:
        output.relative_to(root / "artifacts/dayC/member1/preflight")
    except ValueError as exc:
        raise PermissionError("Preflight output belongs under artifacts/dayC/member1/preflight") from exc
    original = requested if requested.is_absolute() else root / requested
    # Check the supplied final name as well as the resolved target: an internal
    # dangling symlink is an existing destination, not a new output name.
    require_new_file(original)
    require_new_file(output)
    return output


def build_day9_report(root,*,proposal_path=None,preparation_authorization=None):
    """Metadata/validation only. Missing research/owner inputs are reported together."""
    from avfuse.member1_fusion.dayc_freeze import check_preparation_authorization,prior_test_artifacts,verify_registry
    report={'phase':'day9','status':'BLOCKED','blockers':[],'test_rows_read':False,'test_features_read':False,
            'freeze_written':False,'policy_selected':False,'checks':{}}
    try:
        context_path=root/'configs/dayC/member1/C8A_context.json';context=read_strict_json(context_path)
        if preparation_authorization:
            auth=read_strict_json(ensure_inside_repo(root,preparation_authorization,must_exist=True))
            check_preparation_authorization(auth,sha256_file(context_path));report['source_preparation_authorized']=True
        validation_item=context['experiment_spec']['local_inputs']['validation_manifest']
        path=ensure_inside_repo(root,validation_item['path'],must_exist=True);require_sha256(path,validation_item['sha256'])
        validation=_read_val_manifest(path);ids={r['clip_id'] for r in validation}
        for mode in MODES:
            prediction=ensure_inside_repo(root,Path('artifacts/predictions')/(mode+'_val.jsonl'),must_exist=True)
            report['checks'][mode]=_check_predictions(prediction,mode,ids,CHECKPOINT)
        report['prior_test_outputs_or_attempts']=prior_test_artifacts(root)
        if report['prior_test_outputs_or_attempts']:report['blockers'].append('Existing held-out attempt or predictions require lead incident review')
        if proposal_path:
            proposal=read_strict_json(ensure_inside_repo(root,proposal_path,must_exist=True))
            report['blockers']+=proposal.get('blockers',[])
            verify_registry(root,proposal.get('evidence_hashes',{}),complete=False)
            if proposal.get('ready_for_freeze') is not True:report['blockers'].append('Proposal is not scientifically freeze-ready')
        else:
            report['blockers']+=['Actual B2 statistics/research proposal pending',
                'WAITING FOR MEMBER 2 final threshold/test-seal release',
                'WAITING FOR MEMBER 3 benchmark/runtime release','Lead C9 proposal-bound freeze approval absent']
        report['source_preparation_status']='IMPLEMENTED; joint B2/B3 review pending'
    except (OSError,ValueError,KeyError) as exc:
        report['blockers'].append(f'{type(exc).__name__}: {exc}')
    report['blockers']=list(dict.fromkeys(report['blockers']))
    report['status']='BLOCKED' if report['blockers'] else 'READY_FOR_C9_REVIEW'
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("baseline", "day8", "day9", "day10-readonly"), required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=Path("configs/dayC/member1/experiment_spec.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument('--proposal',type=Path)
    parser.add_argument('--preparation-authorization',type=Path)
    args = parser.parse_args(argv)
    if not args.repo_root.is_absolute():
        parser.error("--repo-root must be absolute")
    root = args.repo_root.resolve(strict=True)
    if Path(__file__).resolve().parents[2] != root:
        raise PermissionError("Run the preflight belonging to the specified repository")
    output = _new_report_path(root, args.out)
    if args.phase=='day9':
        report=build_day9_report(root,proposal_path=args.proposal,preparation_authorization=args.preparation_authorization)
    elif args.phase=='day10-readonly':
        report={'phase':'day10-readonly','status':'BLOCKED','blockers':['Use verify_day10_release --dry-run with actual frozen identities after C9'],
                'test_rows_read':False,'test_features_read':False,'attempt_created':False}
    else:
        report = build_baseline_report(root, args.spec, phase=args.phase)
    if not output.parent.is_dir():
        output.parent.mkdir(parents=True, exist_ok=True)
    digest = write_new_json(output, report)
    print(f"{report['status']}: {output}\nSHA256 {digest}")
    for blocker in report["blockers"]:
        print("BLOCKER:", blocker)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
