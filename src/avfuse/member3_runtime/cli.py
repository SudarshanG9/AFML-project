"""Member 3 command-line interface (Segment F).

    python -m avfuse.member3_runtime.cli <command> ...

Commands: ``validate-config``, ``run``, ``predict``, ``benchmark``, ``inspect`` and the legacy
``--smoke`` diagnostic. The CLI only wires existing pieces together: Segment E orchestration,
the feature loaders, Member 2's prediction validator and test seal, and Segment D run
provenance. It contains no model logic and no way around the sealed test split.

Exit codes (stable, documented in docs/member3/CLI.md):
    0 success (complete)        1 runtime failure          2 invalid usage/configuration
    3 blocked (unsupported mode) 4 incomplete (partial/subset/unfinalized; not releasable)
    5 sealed-test access denied  6 missing input           7 checkpoint hash mismatch
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from avfuse.shared.contracts import PRECISION_MODES

EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_BLOCKED = 0, 1, 2, 3
EXIT_INCOMPLETE, EXIT_SEALED, EXIT_MISSING, EXIT_CHECKPOINT = 4, 5, 6, 7
EXIT_NAMES = {0: "OK", 1: "FAILED", 2: "USAGE", 3: "BLOCKED", 4: "INCOMPLETE",
              5: "SEALED", 6: "MISSING_INPUT", 7: "CHECKPOINT_MISMATCH"}

DEFAULT_CONFIG = "configs/benchmark.example.json"
DEFAULT_RUNS_ROOT = "artifacts/member3/runs"
CHECKPOINT_NAMES = ("lavdf", "avdeepfake1m")
FEATURE_FORMATS = ("dimodif_npz", "shared_pt")
_DEVICE = re.compile(r"^(cpu|cuda(:[0-9]+)?|mps)$")


class CliError(Exception):
    """An error with a defined exit code and an actionable message."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def _usage(message: str) -> CliError:
    return CliError(EXIT_USAGE, message)


def _repo_root() -> Path:
    from .artifacts import REPO_ROOT
    return REPO_ROOT


# ------------------------------------------------------------------------------ parsing
def _add_common(p: argparse.ArgumentParser, *, split: bool = True) -> None:
    g = p.add_argument_group("model and inputs")
    g.add_argument("--upstream-root",
                   help="original DiMoDif checkout (contains src/models.py, ckpt/)")
    g.add_argument("--checkpoint-name", choices=CHECKPOINT_NAMES,
                   help="checkpoint to load; never defaulted silently")
    g.add_argument("--manifest", help="whole-video manifest JSONL (manifest_whole.jsonl)")
    g.add_argument("--feature-root", help="root of <split>/<video_id>/features.npz")
    g.add_argument("--feature-format", choices=FEATURE_FORMATS, default="dimodif_npz")
    g.add_argument("--device", default="cpu", help="cpu | cuda[:N] | mps (no silent fallback)")
    g.add_argument("--modes", nargs="+", default=list(PRECISION_MODES), metavar="MODE",
                   help=f"subset of {', '.join(PRECISION_MODES)} (default: all four)")
    g.add_argument("--allow-partial", action="store_true",
                   help="development aid: run supported modes only; result is never releasable")
    g.add_argument("--max-clips", type=int,
                   help="development subset; result is never releasable")
    c = p.add_argument_group("checkpoint identity")
    c.add_argument("--expected-checkpoint-sha256", help="fail before inference on mismatch")
    c.add_argument("--require-lavdf-hash", action="store_true",
                   help="require Member 2's agreed LAV-DF checkpoint SHA-256")
    if split:
        s = p.add_argument_group("split and sealed test access")
        s.add_argument("--split", choices=("val", "test"), default="val")
        s.add_argument("--protocol", help="frozen protocol file (test only)")
        s.add_argument("--unseal-test", action="store_true",
                       help="authorize test access (logged)")
        s.add_argument("--purpose", help="why the test split is opened (test only; logged)")


def _add_run_opts(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("run directory and provenance")
    g.add_argument("--config", default=DEFAULT_CONFIG, help="benchmark config JSON")
    g.add_argument("--runs-root", default=DEFAULT_RUNS_ROOT)
    g.add_argument("--run-id")
    g.add_argument("--experiment-id")
    g.add_argument("--overwrite", action="store_true")
    g.add_argument("--no-hash-cache", action="store_true",
                   help="skip hashing feature-cache contents (slow on 3,200 videos)")
    g.add_argument("--allow-unset-inputs", action="store_true",
                   help="development/synthetic only: unhashed inputs are recorded as not_provided")
    b = p.add_argument_group("benchmark settings")
    b.add_argument("--warmup", type=int, default=10)
    b.add_argument("--repeats", type=int, default=50)
    b.add_argument("--percentile-method", choices=("linear", "lower", "nearest_rank"),
                   default="linear")
    b.add_argument("--no-memory", action="store_true")
    b.add_argument("--benchmark-clip-id")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m avfuse.member3_runtime.cli",
        description="AVFuse-CausalQ Member 3 runtime. Cached-feature inference and benchmarking; "
                    "not raw-video end-to-end, and no accuracy is computed here.",
        epilog="Exit codes: 0 ok, 1 failed, 2 usage, 3 blocked/unsupported, 4 incomplete, "
               "5 sealed test denied, 6 missing input, 7 checkpoint mismatch.")
    p.add_argument("--smoke", action="store_true",
                   help="legacy synthetic wiring check (toy model; not a result)")
    sub = p.add_subparsers(dest="command")

    v = sub.add_parser("validate-config", help="check arguments, paths and config; run nothing")
    _add_common(v)
    v.add_argument("--config", default=DEFAULT_CONFIG)
    v.add_argument("--json", action="store_true")

    r = sub.add_parser("run", help="predictions + cached-feature benchmark in one Segment D run")
    _add_common(r)
    _add_run_opts(r)
    r.add_argument("--benchmark", dest="benchmark", action="store_true", default=None)
    r.add_argument("--no-benchmark", dest="benchmark", action="store_false")
    r.add_argument("--json", action="store_true")

    pr = sub.add_parser("predict", help="write a prediction JSONL only (no run directory)")
    _add_common(pr)
    pr.add_argument("--output", required=True)
    pr.add_argument("--run-id", default=None)
    pr.add_argument("--overwrite", action="store_true")
    pr.add_argument("--json", action="store_true")

    b = sub.add_parser("benchmark", help="cached-feature model benchmark on one validation sample")
    _add_common(b, split=False)
    _add_run_opts(b)
    b.add_argument("--json", action="store_true")

    i = sub.add_parser("inspect", help="verify and summarise an existing run directory")
    i.add_argument("run_dir")
    i.add_argument("--json", action="store_true")
    return p


# -------------------------------------------------------------------------- validation
def validate_args(args: argparse.Namespace) -> None:
    """Usage errors (exit 2), sealing (5), then missing inputs (6). Loads no model or feature."""
    modes = list(args.modes)
    bad = [m for m in modes if m not in PRECISION_MODES]
    if bad:
        raise _usage(f"unknown mode(s) {bad}; choose from {list(PRECISION_MODES)}")
    if len(set(modes)) != len(modes):
        raise _usage("--modes contains duplicates")
    if not _DEVICE.match(args.device):
        raise _usage(f"--device must be cpu, cuda[:N] or mps (got {args.device!r}); 'auto' is not "
                     "accepted so the executing device is always explicit")
    for name in ("warmup", "repeats", "max_clips"):
        value = getattr(args, name, None)
        floor = 0 if name == "warmup" else 1
        if value is not None and value < floor:
            raise _usage(f"--{name.replace('_', '-')} must be >= {floor}")
    if args.require_lavdf_hash and args.expected_checkpoint_sha256:
        raise _usage("use either --expected-checkpoint-sha256 or --require-lavdf-hash, not both")
    sha = args.expected_checkpoint_sha256
    if sha is not None and not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise _usage("--expected-checkpoint-sha256 must be 64 lowercase hex characters")
    for name in ("upstream_root", "checkpoint_name", "manifest", "feature_root"):
        if not getattr(args, name):
            raise _usage(f"--{name.replace('_', '-')} is required")
    if args.split == "test":
        if not (args.protocol and args.unseal_test and args.purpose and args.purpose.strip()):
            raise CliError(
                EXIT_SEALED,
                "the test split is sealed. A held-out run needs --protocol <frozen protocol file>, "
                "--unseal-test and a non-empty --purpose (the access is logged). Calibrate and "
                "freeze on val first.")
        if not Path(args.protocol).is_file():
            raise CliError(EXIT_SEALED, f"--protocol file not found: {args.protocol}")
        if getattr(args, "benchmark", None) is True:
            raise _usage("benchmarking is validation-only; do not combine --benchmark with "
                         "--split test")
    elif args.protocol or args.unseal_test or args.purpose:
        raise _usage("--protocol/--unseal-test/--purpose apply only to --split test; remove them "
                     "for validation runs so test access can never be implied")
    if not Path(args.manifest).is_file():
        raise CliError(EXIT_MISSING, f"manifest not found: {args.manifest}")
    if not Path(args.feature_root).is_dir():
        raise CliError(EXIT_MISSING, f"feature root not found: {args.feature_root}")
    if not Path(args.upstream_root).is_dir():
        raise CliError(EXIT_MISSING, f"DiMoDif upstream root not found: {args.upstream_root}")


def _checkpoint_path(args: argparse.Namespace) -> Path:
    from avfuse.member1_fusion.dimodif_adapter import CHECKPOINTS
    return Path(args.upstream_root) / "ckpt" / "dfd" / f"{CHECKPOINTS[args.checkpoint_name]}.pth"


def _check_config(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute() and not p.is_file():
        p = _repo_root() / path
    if not p.is_file():
        raise CliError(EXIT_MISSING, f"benchmark config not found: {path}")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise _usage(f"benchmark config is not valid JSON: {exc}") from exc


# ------------------------------------------------------------------ model / run helpers
def _build_model(args: argparse.Namespace):
    """Real adapter factory (patched in tests; the CLI never builds a mock itself)."""
    from .experiment import build_dimodif_adapter
    return build_dimodif_adapter(args.upstream_root, args.checkpoint_name, args.device)


def _clips_and_provider(args: argparse.Namespace, split: str):
    from .feature_loader import make_feature_provider
    from .predictions import load_clip_refs
    return (load_clip_refs(args.manifest, split),
            make_feature_provider(args.feature_format, args.feature_root))


def _preflight(model: Any, args: argparse.Namespace, modes: Sequence[str]) -> None:
    """Before a TEST run, prove on a VALIDATION clip that every requested mode is supported, so a
    blocked configuration never consumes (and logs) a test access."""
    from .predictions import load_clip_refs
    from .runner import SWEEP_BLOCKED, run_precision_sweep
    try:
        val = load_clip_refs(args.manifest, "val")
    except ValueError as exc:
        raise CliError(
            EXIT_BLOCKED,
            "cannot safely open the sealed test split because validation preflight "
            "could not load validation clips; fix the validation manifest first.",
        ) from exc
    if not val:
        raise CliError(
            EXIT_BLOCKED,
            "cannot safely open the sealed test split because the manifest contains "
            "no validation clips for preflight.",
        )
    _, provider = _clips_and_provider(args, "val")
    sweep = run_precision_sweep(model, provider(val[0]), modes, allow_partial=args.allow_partial)
    if sweep.status == SWEEP_BLOCKED:
        raise CliError(EXIT_BLOCKED, "preflight on a validation clip found unsupported mode(s): "
                       f"{sweep.summary()['unsupported_modes']}. The sealed test split was NOT "
                       "opened and nothing was logged.")


def _make_ctx(args: argparse.Namespace, split: str, modes: Sequence[str]):
    from .artifacts import create_run_context
    config = _check_config(args.config)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    config["experiment_id"] = args.experiment_id or f"m3-{split}-{stamp}"
    config.update(device=args.device, warmup=args.warmup, repeats=args.repeats, batch_size=1,
                  configs=list(modes), percentile_method=args.percentile_method)
    config.setdefault("output", {})["overwrite"] = bool(args.overwrite)
    if args.no_hash_cache:
        config.setdefault("reproducibility", {})["hash_input_cache"] = False
    root = Path(args.feature_root)
    cache_dir = root / split if args.feature_format == "dimodif_npz" else root
    if not cache_dir.is_dir():
        raise CliError(EXIT_MISSING,
                       f"feature directory for split {split!r} not found: {cache_dir}")
    return create_run_context(
        config, runs_root=args.runs_root, repo_root=_repo_root(), run_id=args.run_id,
        checkpoint_path=_checkpoint_path(args), cache_dir=cache_dir,
        allow_unset_inputs=args.allow_unset_inputs)


def _feature_description(args: argparse.Namespace) -> dict[str, Any]:
    from . import provenance as prov
    return {"format": args.feature_format, "feature_root": str(args.feature_root),
            "manifest": str(args.manifest),
            "manifest_sha256": prov.sha256_file(Path(args.manifest)),
            "checkpoint_name": args.checkpoint_name}


def _emit(args: argparse.Namespace, payload: dict[str, Any], lines: Sequence[str]) -> None:
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        print("\n".join(lines))


def _exit_for_status(status: str, finalized: bool = True) -> int:
    if status == "complete":
        return EXIT_OK if finalized else EXIT_INCOMPLETE
    return {"blocked": EXIT_BLOCKED, "partial": EXIT_INCOMPLETE}.get(status, EXIT_FAILED)


def _summary_lines(title: str, s: dict[str, Any]) -> list[str]:
    pred = s.get("predictions", {})
    lines = [
        (f"{title}: status={s['status']}  finalized={s.get('finalized')}  "
         f"four_mode_complete={s.get('four_mode_complete')}"),
        f"  run_id={s['run_id']}  split={s['split']}  modes={s['requested_modes']}"]
    for mode, why in pred.get("unsupported_modes", {}).items():
        lines.append(f"  UNSUPPORTED {mode}: {why}  (no prediction generated)")
    if s.get("dataset_complete") is False:
        lines.append("  INCOMPLETE: subset of the split processed; not a completed run")
    lines.append("  Cached-feature inference only; no accuracy computed or claimed.")
    return lines


# ----------------------------------------------------------------------------- commands
def cmd_validate_config(args: argparse.Namespace) -> int:
    validate_args(args)
    config = _check_config(args.config)
    from jsonschema import Draft202012Validator

    from .artifacts import CONFIG_SCHEMA_PATH
    schema = json.loads(CONFIG_SCHEMA_PATH.read_text(encoding="utf-8"))
    problems = [e.message[:160] for e in Draft202012Validator(schema).iter_errors(config)]
    if problems:
        raise _usage("benchmark config invalid: " + "; ".join(problems[:5]))
    ckpt = _checkpoint_path(args)
    if not ckpt.is_file() or not ckpt.with_suffix(".json").is_file():
        raise CliError(EXIT_MISSING, f"checkpoint or model config missing next to: {ckpt}")
    from .predictions import load_clip_refs
    n = len(load_clip_refs(args.manifest, args.split))
    root = Path(args.feature_root)
    base = root / args.split if args.feature_format == "dimodif_npz" else root
    if not base.is_dir():
        raise CliError(EXIT_MISSING, f"feature directory not found: {base}")
    payload = {"ok": True, "split": args.split, "modes": list(args.modes),
               "clips_in_split": n, "checkpoint_file": str(ckpt),
               "test_access": "authorized by flags (not yet opened)" if args.split == "test"
               else "not requested",
               "note": "Configuration and paths only; no model was loaded and no feature was read."}
    _emit(args, payload, [f"OK: configuration valid ({n} {args.split} clips, modes {args.modes})",
                          payload["note"]])
    return EXIT_OK


def _prepare(args: argparse.Namespace):
    """Shared by run/predict/benchmark: build the model and check checkpoint identity."""
    from .predictions import check_checkpoint_hash
    model = _build_model(args)
    check_checkpoint_hash(model.checkpoint_sha256, args.expected_checkpoint_sha256,
                          args.require_lavdf_hash)
    return model


def cmd_run(args: argparse.Namespace) -> int:
    validate_args(args)
    if args.benchmark is None:
        args.benchmark = args.split == "val"
    from .experiment import execute_experiment
    model = _prepare(args)
    modes = tuple(args.modes)
    if args.split == "test":
        _preflight(model, args, modes)
    clips, provider = _clips_and_provider(args, args.split)
    ctx = _make_ctx(args, args.split, modes)
    summary = execute_experiment(
        model, ctx, clips, provider, checkpoint_sha256=model.checkpoint_sha256, split=args.split,
        modes=modes, allow_partial=args.allow_partial, max_clips=args.max_clips,
        device=args.device, warmup=args.warmup, repeats=args.repeats,
        percentile_method=args.percentile_method, measure_memory=not args.no_memory,
        benchmark=args.benchmark, benchmark_clip_id=args.benchmark_clip_id,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        require_lavdf_hash=args.require_lavdf_hash, protocol_path=args.protocol,
        unseal=args.unseal_test, purpose=args.purpose,
        feature_description=_feature_description(args),
        data_origin="development_unhashed_inputs" if args.allow_unset_inputs
        else "real_cached_features")
    summary["run_dir"] = str(ctx.run_dir)
    _emit(args, summary, [*_summary_lines("run", summary), f"  run_dir={ctx.run_dir}"])
    return _exit_for_status(summary["status"], summary.get("finalized", False))


def cmd_predict(args: argparse.Namespace) -> int:
    validate_args(args)
    from .artifacts import generate_run_id
    from .predictions import generate_predictions
    model = _prepare(args)
    modes = tuple(args.modes)
    if args.split == "test":
        _preflight(model, args, modes)
    clips, provider = _clips_and_provider(args, args.split)
    report = generate_predictions(
        model, clips, provider, run_id=args.run_id or generate_run_id(),
        checkpoint_sha256=model.checkpoint_sha256, output_path=args.output, split=args.split,
        modes=modes, allow_partial=args.allow_partial, overwrite=args.overwrite,
        max_clips=args.max_clips, expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        require_lavdf_hash=args.require_lavdf_hash, protocol_path=args.protocol,
        unseal=args.unseal_test, purpose=args.purpose)
    lines = [(f"predict: status={report.status}  published={report.published}  "
              f"rows={report.n_rows}  four_mode_complete={report.four_mode_complete}  "
              f"full_split={report.full_split}")]
    lines += [f"  UNSUPPORTED {m}: {w}  (no prediction generated)"
              for m, w in report.unsupported_modes.items()]
    lines += [f"  note: {n}" for n in report.notes]
    _emit(args, report.to_dict(), lines)
    if report.status == "complete" and not report.full_split:
        return EXIT_INCOMPLETE
    return _exit_for_status(report.status)


def cmd_benchmark(args: argparse.Namespace) -> int:
    args.split, args.protocol, args.unseal_test, args.purpose = "val", None, False, None
    validate_args(args)
    from .model_benchmark import benchmark_precision_modes
    model = _prepare(args)
    modes = tuple(args.modes)
    clips, provider = _clips_and_provider(args, "val")
    sample = next((c for c in clips if args.benchmark_clip_id in (None, c.clip_id)), None)
    if sample is None:
        raise _usage(f"--benchmark-clip-id {args.benchmark_clip_id!r} is not a validation clip")
    ctx = _make_ctx(args, "val", modes)
    res = benchmark_precision_modes(
        model, provider(sample), ctx=ctx, device=args.device, modes=modes, warmup=args.warmup,
        repeats=args.repeats, percentile_method=args.percentile_method,
        measure_memory=not args.no_memory, allow_partial=args.allow_partial,
        checkpoint_sha256=model.checkpoint_sha256)
    payload = {"status": res.status, "run_id": ctx.run_id, "run_dir": str(ctx.run_dir),
               "benchmarked_modes": list(res.benchmarked_modes), "scope": "cached_feature_model",
               "input_boundary": "cached_encoder_tokens", "sample_clip_id": sample.clip_id,
               "four_mode_complete": res.four_mode_complete, "sweep": res.sweep.summary(),
               "split": "val", "finalized": False}
    if res.status == "complete":
        ctx.write_json_artifact("benchmark_summary.json", payload,
                                artifact_type="avfuse_execution_summary")
        ctx.finalize()
        ctx.verify()
        payload["finalized"] = True
    lines = [(f"benchmark: status={res.status}  modes={list(res.benchmarked_modes)}  "
              f"finalized={payload['finalized']}"),
             "  Cached-feature model latency only; not raw-video end-to-end."]
    lines += [f"  UNSUPPORTED {m}: {w}  (no record written)"
              for m, w in payload["sweep"]["unsupported_modes"].items()]
    _emit(args, payload, lines)
    return _exit_for_status(res.status, payload["finalized"])


def cmd_inspect(args: argparse.Namespace) -> int:
    from .artifacts import MANIFEST_NAME, verify_run_integrity
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise CliError(EXIT_MISSING, f"run directory not found: {run_dir}")
    summary: dict[str, Any] = {}
    for name in ("execution_summary.json", "benchmark_summary.json", "failure_summary.json"):
        f = run_dir / name
        if f.is_file():
            try:
                summary = json.loads(f.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                summary = {"status": "unreadable_summary"}
            break
    finalized = (run_dir / MANIFEST_NAME).is_file()
    integrity = verify_run_integrity(run_dir) if finalized else None
    status = summary.get("status", "unknown")
    pred = summary.get("predictions", {})
    complete_four = bool(finalized and integrity and integrity["ok"] and status == "complete"
                         and summary.get("four_mode_complete")
                         and summary.get("dataset_complete", True)
                         and summary.get("split") == "val")
    payload = {
        "run_dir": str(run_dir), "finalized": finalized, "status": status,
        "integrity_ok": None if integrity is None else integrity["ok"],
        "four_mode_complete": summary.get("four_mode_complete"),
        "dataset_complete": summary.get("dataset_complete"),
        "split": summary.get("split"), "unsupported_modes": pred.get("unsupported_modes", {}),
        "complete_four_mode_validation_run": complete_four,
        "note": "Implementation-level status only. Not a release approval and not evidence of "
                "detector accuracy or speedup; see docs/member3/RELEASE_CHECKLIST.md."}
    lines = [
        (f"inspect {run_dir}: status={status} finalized={finalized} "
         f"integrity_ok={payload['integrity_ok']}"),
        (f"  four_mode_complete={payload['four_mode_complete']} "
         f"dataset_complete={payload['dataset_complete']} split={payload['split']}"),
        f"  complete_four_mode_validation_run={complete_four}", f"  {payload['note']}"]
    _emit(args, payload, lines)
    if (integrity is not None and not integrity["ok"]) or status == "failed":
        return EXIT_FAILED
    if status == "blocked":
        return EXIT_BLOCKED
    return EXIT_OK if status == "complete" and finalized else EXIT_INCOMPLETE


def cmd_smoke() -> int:
    import torch

    from avfuse.member1_fusion.diagnostic_model import DiagnosticModel

    from .benchmark import benchmark_callable
    from .runner import run_four_modes
    print("DIAGNOSTIC ONLY: synthetic toy-model smoke test; not a research result.")
    torch.manual_seed(7)
    device = torch.device("cpu")
    model = DiagnosticModel(16).to(device).eval()
    features = {"audio_tokens": torch.randn(1, 12, 16, device=device),
                "video_tokens": torch.randn(1, 8, 16, device=device)}
    print("DIAGNOSTIC ONLY: synthetic tokens + randomly initialized toy model")
    for mode, score in run_four_modes(model, features).items():
        timing = benchmark_callable(
            lambda mode=mode: model.predict_from_cached_features(features, mode),
            device, warmup=2, repeats=5)
        print(f"{mode:22s} fake-score={float(score.item()):.4f}  p50={timing['p50_ms']:.3f} ms")
    print("No forensic accuracy or INT8 speedup is claimed by this diagnostic.")
    return EXIT_OK


_COMMANDS = {"validate-config": cmd_validate_config, "run": cmd_run, "predict": cmd_predict,
             "benchmark": cmd_benchmark, "inspect": cmd_inspect}


def _classify(exc: BaseException) -> tuple[int, str]:
    from avfuse.member2_data.seal import SealedTestError

    from .artifacts import ArtifactExistsError
    from .experiment import MissingInputError
    from .model_benchmark import CheckpointIdentityError
    from .predictions import CheckpointMismatchError, TestAccessError
    if isinstance(exc, CliError):
        return exc.code, str(exc)
    if isinstance(exc, (SealedTestError, TestAccessError)):
        return EXIT_SEALED, str(exc)
    if isinstance(exc, (CheckpointMismatchError, CheckpointIdentityError)):
        return EXIT_CHECKPOINT, str(exc)
    if isinstance(exc, (ArtifactExistsError, FileExistsError)):
        return EXIT_USAGE, f"{exc}. Use a new --run-id/--output or pass --overwrite."
    if isinstance(exc, (MissingInputError, FileNotFoundError)):
        return EXIT_MISSING, str(exc)
    return EXIT_FAILED, f"{type(exc).__name__}: {exc}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)  # argparse itself exits 2 on bad usage
    if args.smoke and not args.command:
        return cmd_smoke()
    if not args.command:
        parser.print_help(sys.stderr)
        return EXIT_USAGE
    try:
        return _COMMANDS[args.command](args)
    except Exception as exc:  # noqa: BLE001  classified below; never reported as success
        code, message = _classify(exc)
        print(f"ERROR[{EXIT_NAMES[code]}]: {message}", file=sys.stderr)
        return code


if __name__ == "__main__":
    sys.exit(main())
