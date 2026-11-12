"""Guarded validation pilot; BLOCKED reports are not experimental results."""
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import argparse
import math
import torch
from jsonschema.exceptions import ValidationError
from avfuse.member1_fusion.dayc_guards import (
    ensure_inside_repo, read_strict_json, require_human_approval, require_new_file,
    require_sha256, require_frozen_inputs, sha256_file, write_new_json,
)
from avfuse.member1_fusion.dayc_precision_audit import (
    check_model_invariants, reference_parity_on_validation_feature,
    validate_precision_trace, _exclusive_audit,
)
from scripts.dayC_member1 import preflight


def _c0_release(root):
    approvals = root / "artifacts/dayC/member1/approvals"
    for request in sorted(approvals.glob("C0_request_*.json"), reverse=True):
        request = ensure_inside_repo(root, request, must_exist=True)
        pending = read_strict_json(request)
        context_path = ensure_inside_repo(root, Path(pending["context_path"]), must_exist=True)
        require_sha256(context_path, pending["context_sha256"])
        context = read_strict_json(context_path)
        if context.get("gate") != "C0" or context.get("repo_root") != str(root):
            continue
        for group in ("review_files", "evidence"):
            for rel, digest in context[group].items():
                require_sha256(ensure_inside_repo(root, Path(rel), must_exist=True), digest)
        for ack in sorted(approvals.glob("C0_ack_*.json"), reverse=True):
            try:
                ack = ensure_inside_repo(root, ack, must_exist=True)
                require_human_approval(ack, "C0", context_sha256=pending["context_sha256"])
            except (ValueError, PermissionError):
                continue
            return {"path": str(ack.relative_to(root)), "sha256": sha256_file(ack),
                    "context_sha256": pending["context_sha256"], "reviewed_head": context["baseline_head"]}
    raise PermissionError("No human C0 acknowledgement matches the current reviewed file bytes")


def pilot_entry(root, spec_path, upstream, requested_feature):
    """Metadata only: all 1,200 caches and the accepted runtime must be ready."""
    report = {"status": "BLOCKED", "blockers": [], "feature_contents_read": False,
              "model_constructed": False, "test_access": False}
    try:
        report["c0"] = _c0_release(root)
        spec = read_strict_json(ensure_inside_repo(root, spec_path, must_exist=True))
        paths = {name: ensure_inside_repo(root, Path(item["path"]), must_exist=True)
                 for name, item in spec["local_inputs"].items()}
        hashes = {name: item["sha256"] for name, item in spec["local_inputs"].items()}
        require_frozen_inputs(hashes, paths)
        require_sha256(paths["checkpoint"], preflight.CHECKPOINT)
        report["input_paths"] = {name: str(p.relative_to(root)) for name, p in paths.items()}
        report["input_hashes"] = hashes
        report["current_head"] = preflight._git(root, "rev-parse", "HEAD")
        if preflight._git(root, "branch", "--show-current") != spec["expected_baseline"]["branch"]:
            raise PermissionError("Branch differs from reviewed baseline")
        if ensure_inside_repo(root, upstream, must_exist=True) != paths["upstream_source"].parent.parent:
            raise PermissionError("Upstream model root differs from approved checkpoint/config/source")
        validation = preflight._read_val_manifest(paths["validation_manifest"])
        ready = preflight._read_val_manifest(paths["ready_val_manifest"])
        ids = preflight._reconcile_val_manifests(validation, ready)
        for mode in preflight.MODES[:3]:
            preflight._check_predictions(paths["predictions_" + mode], mode, ids, preflight.CHECKPOINT)
        report["cache"] = preflight._check_feature_paths(root, ready)
        if report["cache"]["existing_local_paths"] != 1200 or report["cache"]["unresolved_paths"]:
            report["blockers"].append("Full 1,200 project-local validation cache is unavailable")
        report["environment"] = preflight._environment()
        accepted = read_strict_json(paths["environment_reference"])["reference_environment"]
        report["runtime_differences"] = {
            key: {"accepted": accepted[key], "local": report["environment"][key]}
            for key in ("python", "platform", "versions", "torch_threads", "torch_interop_threads", "cuda_available")
            if accepted[key] != report["environment"][key]}
        if report["runtime_differences"]:
            report["blockers"].append("Runtime differs from accepted Day B CPU environment")
        if report["blockers"]:
            return report
        feature = ensure_inside_repo(root, requested_feature, must_exist=True)
        matches = [r for r in ready if ensure_inside_repo(root, Path(r["feature_path"])) == feature]
        if len(matches) != 1 or matches[0]["split"] != "val":
            raise PermissionError("Feature is not uniquely bound to the approved validation cohort")
        require_new_file(root / "artifacts/predictions/restore_query_val.jsonl")
        report.update(status="READY", split="val", clip_id=matches[0]["clip_id"],
                      feature_path=str(feature.relative_to(root)), feature_sha256=sha256_file(feature))
    except (OSError, ValueError, KeyError, TypeError, ImportError, ValidationError) as exc:
        report["blockers"].append(f"{type(exc).__name__}: {exc}")
    return report


def _run_pilot(root, upstream, release):
    if release.get("status") != "READY" or release.get("split") != "val":
        raise PermissionError("Pilot entry is not READY for validation")
    from scripts.dayB_inference import create_adapter, load_features
    paths = {key: ensure_inside_repo(root, Path(rel), must_exist=True)
             for key, rel in release["input_paths"].items()}
    require_frozen_inputs(release["input_hashes"], paths)
    feature = ensure_inside_repo(root, Path(release["feature_path"]), must_exist=True)
    require_sha256(feature, release["feature_sha256"])
    adapter = create_adapter(SimpleNamespace(upstream=str(upstream), device="cpu"))
    features = load_features(feature)
    initial = check_model_invariants(adapter, checkpoint_sha256=preflight.CHECKPOINT)
    snapshots = {key: value.detach().clone() for key, value in features.items()}
    parity = reference_parity_on_validation_feature(adapter, features)
    modes = {}
    with _exclusive_audit(adapter.model), torch.inference_mode():
        for mode in ("low_precision", "restore_value_output", "restore_query"):
            trace = []
            output = adapter.forward_with_mode(features, mode=mode, trace=trace)
            trace_audit = validate_precision_trace(trace, mode=mode)
            if output["score_fake"].numel() != 1 or not torch.isfinite(output["logits"]).all():
                raise ValueError("Pilot requires finite logits and batch size one")
            score = float(output["score_fake"].reshape(-1)[0].item())
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Pilot fake probability is invalid")
            modes[mode] = {"score_fake": score, "trace": trace, "trace_audit": trace_audit}
            state = check_model_invariants(adapter, checkpoint_sha256=preflight.CHECKPOINT)
            if state["state_dict_sha256"] != initial["state_dict_sha256"]:
                raise RuntimeError("Model state changed during pilot")
            if set(features) != set(snapshots) or any(not torch.equal(features[k], v) for k, v in snapshots.items()):
                raise RuntimeError("Cached input tensors changed during pilot")
    require_frozen_inputs(release["input_hashes"], paths)
    require_sha256(feature, release["feature_sha256"])
    return {"status": "PILOT_COMPLETE_PENDING_C8A", "clip_id": release["clip_id"], "split": "val",
            "checkpoint_sha256": preflight.CHECKPOINT, "parity": parity, "modes": modes,
            "query_minus_low_precision_score": modes["restore_query"]["score_fake"] - modes["low_precision"]["score_fake"],
            "state_and_inputs_unchanged": True,
            "scope": "one-clip diagnostics; no population accuracy or H3 conclusion",
            "limitation": "Five-layer intervention is cumulative; downstream activations can differ"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("repo-root", "upstream", "features", "out"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=Path("configs/dayC/member1/experiment_spec.json"))
    args = parser.parse_args(argv)
    if not args.repo_root.is_absolute():
        parser.error("--repo-root must be absolute")
    root = args.repo_root.resolve(strict=True)
    if Path(__file__).resolve().parents[2] != root:
        raise PermissionError("Runner does not belong to the specified clone")
    output = ensure_inside_repo(root, args.out)
    try:
        output.relative_to(root / "artifacts/dayC/member1/day8")
    except ValueError as exc:
        raise PermissionError("Pilot reports belong under artifacts/dayC/member1/day8") from exc
    require_new_file(args.out if args.out.is_absolute() else root / args.out)
    require_new_file(output)
    release = pilot_entry(root, args.spec, args.upstream, args.features)
    code = {str(p.relative_to(root)): sha256_file(p) for p in
            (Path(__file__), root / "src/avfuse/member1_fusion/dayc_precision_audit.py")}
    report = {"schema_id": "avfuse-member1-dayc-query-pilot/1.0", "status": "BLOCKED",
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "entry": release,
              "real_model_inference_started": False, "source_hashes": code}
    exit_code = 2
    if release["status"] == "READY":
        report["real_model_inference_started"] = True
        try:
            report["pilot"] = _run_pilot(root, ensure_inside_repo(root, args.upstream), release)
            for rel, digest in code.items():
                require_sha256(root / rel, digest)
            report["status"], exit_code = "PILOT_COMPLETE_PENDING_C8A", 0
        except Exception as exc:
            report.update(status="FAILED_STOP_NO_AUTOMATIC_RETRY", error=f"{type(exc).__name__}: {exc}")
            exit_code = 1
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = write_new_json(output, report)
    print(f"{report['status']}: {output}\nSHA256 {digest}")
    for blocker in release["blockers"]:
        print("BLOCKER:", blocker)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
