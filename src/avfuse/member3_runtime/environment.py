"""Hardware and software environment diagnostics for Member 3.

This module audits the machine that will run benchmarks. It produces a
machine-readable report (JSON) and a readable terminal summary.

What the precision checks mean
------------------------------
The FP32 / FP16 / BF16 / SDPA checks execute **one tiny tensor operation** per
device and compare it with a CPU float64 reference. A ``succeeded`` status means
only that this one operation ran and produced plausible numbers. It does NOT
mean the research model, or any benchmark, is supported or fast in that dtype,
and the checks measure no performance. INT8 is never tested: the project's INT8
mode is simulated symmetric fake quantization in floating point, not an INT8
hardware kernel.

Design rules
------------
* Only the standard library is required at import time. ``torch`` and ``psutil``
  are imported lazily and optionally.
* Every section is collected independently, so one failing probe (for example a
  broken accelerator driver) is recorded in the report instead of crashing it.
* Nothing here is a benchmark. No timings are recorded.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPORT_TYPE = "avfuse_environment_report"
REPORT_VERSION = "1.0"
DEFAULT_OUTPUT = Path("artifacts") / "runs" / "environment_report.json"

# Status vocabulary for every probe and operation check.
SUCCEEDED = "succeeded"
FAILED = "failed"
UNSUPPORTED = "unsupported"
NOT_TESTED = "not_tested"
NOT_APPLICABLE = "not_applicable"

DISCLAIMERS = (
    ("Precision/SDPA checks run one small tensor operation per device. A 'succeeded' "
     "status does not show that the research model supports that precision on that device."),
    "No timing or performance is measured or implied by this report.",
    ("INT8 in this project is simulated symmetric fake quantization in floating point. "
     "No INT8 hardware acceleration is tested or claimed."),
    "Only cuda:0 (when present) and the single MPS device are exercised by the operation checks.",
)


class CheckFailed(Exception):
    """Raised by a precision check when the operation ran but gave a wrong result."""


# --------------------------------------------------------------------------- helpers
def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _error_entry(section: str, exc: BaseException) -> dict[str, str]:
    return {"section": section, "error_type": type(exc).__name__, "message": str(exc)[:500]}


def _load_torch() -> tuple[Any | None, str | None]:
    """Import torch lazily. Returns ``(module, None)`` or ``(None, error_message)``.

    ``OSError`` is caught as well as ``ImportError`` because broken DLL loading on
    Windows surfaces as ``OSError``.
    """
    try:
        import torch

        return torch, None
    except Exception as exc:  # noqa: BLE001 - any import-time failure means "unavailable"
        return None, f"{type(exc).__name__}: {str(exc)[:300]}"


def _run_git(args: list[str], cwd: Path) -> str | None:
    try:
        proc = subprocess.run(  # fixed argument list, no shell
            ["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


# --------------------------------------------------------------------- basic sections
def collect_python() -> dict[str, Any]:
    return {
        "version": platform.python_version(),
        "executable": sys.executable,
        "implementation": platform.python_implementation(),
    }


def collect_os() -> dict[str, Any]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "architecture": platform.machine(),
        "platform": platform.platform(),
    }


def _linux_cpu_model() -> str | None:
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.lower().startswith(("model name", "hardware", "cpu model")):
                    return line.split(":", 1)[1].strip() or None
    except OSError:
        return None
    return None


def collect_cpu() -> dict[str, Any]:
    """CPU description and core counts. Physical cores need optional ``psutil``."""
    description = _linux_cpu_model() if sys.platform.startswith("linux") else None
    if not description:
        description = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER") or None
    physical: int | None = None
    physical_source = "unavailable: psutil not installed (pip install psutil)"
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
        physical_source = "psutil" if physical is not None else "psutil returned None"
    except ImportError:
        pass
    return {
        "description": description,
        "logical_cores": os.cpu_count(),
        "physical_cores": physical,
        "physical_cores_source": physical_source,
    }


def _memory_windows() -> tuple[int, int]:
    import ctypes

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
        raise OSError("GlobalMemoryStatusEx failed")
    return int(status.ullTotalPhys), int(status.ullAvailPhys)


def _memory_proc_meminfo() -> tuple[int, int | None]:
    values: dict[str, int] = {}
    with open("/proc/meminfo", encoding="utf-8") as fh:
        for line in fh:
            key, _, rest = line.partition(":")
            parts = rest.split()
            if parts and parts[0].isdigit():
                values[key] = int(parts[0]) * 1024  # reported in kB
    return values["MemTotal"], values.get("MemAvailable")


def collect_memory() -> dict[str, Any]:
    """Total and available RAM in bytes.

    Order: ``psutil`` (optional) -> Windows ``GlobalMemoryStatusEx`` -> Linux
    ``/proc/meminfo`` -> POSIX ``sysconf`` (total only). Unknown values are ``None``.
    """
    try:
        import psutil

        vm = psutil.virtual_memory()
        return {"total_bytes": int(vm.total), "available_bytes": int(vm.available),
                "source": "psutil"}
    except ImportError:
        pass
    attempts: list[tuple[str, Callable[[], tuple[int, int | None]]]] = []
    if os.name == "nt":
        attempts.append(("windows_GlobalMemoryStatusEx", _memory_windows))
    attempts.append(("/proc/meminfo", _memory_proc_meminfo))

    def _sysconf() -> tuple[int, int | None]:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"), None

    attempts.append(("posix_sysconf_total_only", _sysconf))
    for source, fn in attempts:
        try:
            total, available = fn()
            return {"total_bytes": total, "available_bytes": available, "source": source}
        except (OSError, KeyError, ValueError, AttributeError):
            continue
    return {"total_bytes": None, "available_bytes": None, "source": "unavailable"}


def collect_git(repo_root: Path) -> dict[str, Any]:
    """Current commit and whether the work tree has uncommitted/untracked changes."""
    commit = _run_git(["rev-parse", "HEAD"], repo_root)
    status = _run_git(["status", "--porcelain"], repo_root) if commit else None
    return {
        "commit": commit,
        "dirty": (bool(status) if status is not None else None),
        "source": "git" if commit else "unavailable (not a git checkout or git not installed)",
    }


# ------------------------------------------------------------------------- torch probes
def _collect_cuda(torch: Any) -> dict[str, Any]:
    info: dict[str, Any] = {
        "available": False,
        "runtime_version": getattr(torch.version, "cuda", None),
        "device_count": 0,
        "devices": [],
        "error": None,
    }
    try:
        info["available"] = bool(torch.cuda.is_available())
        if not info["available"]:
            return info
        info["device_count"] = int(torch.cuda.device_count())
        for index in range(info["device_count"]):
            try:
                props = torch.cuda.get_device_properties(index)
                info["devices"].append({
                    "index": index,
                    "name": props.name,
                    "total_memory_bytes": int(props.total_memory),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multi_processor_count": int(props.multi_processor_count),
                })
            except Exception as exc:  # noqa: BLE001 - one bad device must not hide the rest
                info["devices"].append({"index": index, "error": f"{type(exc).__name__}: {exc}"})
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    return info


def _collect_mps(torch: Any) -> dict[str, Any]:
    info: dict[str, Any] = {
        "built": False,
        "available": False,
        "device_name": None,
        "host_architecture": platform.machine(),
        "note": "PyTorch does not expose an MPS device name; host architecture is reported instead.",
        "error": None,
    }
    try:
        info["built"] = bool(torch.backends.mps.is_built())
        info["available"] = bool(torch.backends.mps.is_available())
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    return info


def collect_torch(torch: Any | None, import_error: str | None) -> dict[str, Any]:
    if torch is None:
        return {"available": False, "version": None, "import_error": import_error,
                "num_threads": None, "sdpa_api_available": False,
                "cuda": None, "mps": None}
    try:
        num_threads: int | None = int(torch.get_num_threads())
    except Exception:  # noqa: BLE001
        num_threads = None
    return {
        "available": True,
        "version": str(torch.__version__),
        "import_error": None,
        "num_threads": num_threads,
        "sdpa_api_available": hasattr(torch.nn.functional, "scaled_dot_product_attention"),
        "cuda": _collect_cuda(torch),
        "mps": _collect_mps(torch),
    }


# ------------------------------------------------------------------- operation checks
def _sync(torch: Any, device: str) -> None:
    """Block until queued work finishes so asynchronous errors surface inside the check."""
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    elif device == "mps":
        torch.mps.synchronize()


def _matmul_check(torch: Any, device: str, dtype_name: str, atol: float) -> str:
    dtype = getattr(torch, dtype_name)
    base = (torch.arange(16, dtype=torch.float32).reshape(4, 4) / 16.0).to(device=device, dtype=dtype)
    out = base @ base
    _sync(torch, device)
    if out.dtype != dtype:
        raise CheckFailed(f"output dtype {out.dtype} != requested {dtype}")
    got = out.detach().to("cpu", torch.float64)
    if not bool(torch.isfinite(got).all()):
        raise CheckFailed("output contains non-finite values")
    reference = base.detach().to("cpu", torch.float64)
    reference = reference @ reference
    max_err = float((got - reference).abs().max())
    if max_err > atol:
        raise CheckFailed(f"max abs error {max_err:.3g} exceeds {atol:g} vs CPU float64 reference")
    return f"4x4 {dtype_name} matmul matched CPU float64 reference (max abs err {max_err:.2e})"


def _check_fp32(torch: Any, device: str) -> str:
    return _matmul_check(torch, device, "float32", 1e-4)


def _check_fp16(torch: Any, device: str) -> str:
    return _matmul_check(torch, device, "float16", 1e-2)


def _check_bf16(torch: Any, device: str) -> str:
    return _matmul_check(torch, device, "bfloat16", 5e-2)


def _check_sdpa(torch: Any, device: str) -> str:
    functional = torch.nn.functional
    if not hasattr(functional, "scaled_dot_product_attention"):
        raise NotImplementedError("torch.nn.functional.scaled_dot_product_attention is missing")
    shape = (1, 2, 4, 8)  # (batch, heads, tokens, head_dim); arbitrary tiny test size
    idx = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
    q, k, v = torch.sin(idx * 0.1), torch.cos(idx * 0.07), torch.sin(idx * 0.03 + 1.0)
    out = functional.scaled_dot_product_attention(q.to(device), k.to(device), v.to(device))
    _sync(torch, device)
    got = out.detach().to("cpu", torch.float64)
    if not bool(torch.isfinite(got).all()):
        raise CheckFailed("output contains non-finite values")
    qd, kd, vd = q.double(), k.double(), v.double()
    weights = torch.softmax(qd @ kd.transpose(-1, -2) / math.sqrt(shape[-1]), dim=-1)
    max_err = float((got - weights @ vd).abs().max())
    if max_err > 1e-4:
        raise CheckFailed(f"max abs error {max_err:.3g} exceeds 1e-4 vs manual softmax attention")
    return f"float32 SDPA matched manual softmax attention (max abs err {max_err:.2e})"


#: Operation name -> callable(torch, device) returning a detail string or raising.
#: Module-level so tests can substitute failing checks.
PRECISION_CHECKS: dict[str, Callable[[Any, str], str]] = {
    "fp32": _check_fp32,
    "fp16": _check_fp16,
    "bf16": _check_bf16,
    "sdpa": _check_sdpa,
}

_UNSUPPORTED_MARKERS = (
    "not implemented", "not supported", "unsupported", "does not support",
    "not linked with support", "not built", "no kernel image",
)


def classify_exception(exc: BaseException) -> str:
    """Map an exception to ``unsupported`` (op not available) or ``failed`` (anything else)."""
    if isinstance(exc, NotImplementedError):
        return UNSUPPORTED
    if isinstance(exc, (RuntimeError, TypeError)):
        text = str(exc).lower()
        if any(marker in text for marker in _UNSUPPORTED_MARKERS):
            return UNSUPPORTED
    return FAILED


def _result(status: str, detail: str | None = None, error_type: str | None = None) -> dict[str, Any]:
    return {"status": status, "detail": detail, "error_type": error_type}


def run_precision_checks(torch: Any, device: str) -> dict[str, Any]:
    """Run every registered check on ``device``. Never raises."""
    operations: dict[str, Any] = {}
    for name, check in PRECISION_CHECKS.items():
        try:
            operations[name] = _result(SUCCEEDED, check(torch, device))
        except Exception as exc:  # noqa: BLE001 - classification is the whole point
            operations[name] = _result(classify_exception(exc), str(exc)[:300], type(exc).__name__)
    return {"device": device, "tested": True, "reason_not_tested": None, "operations": operations}


def _not_tested(device: str, reason: str) -> dict[str, Any]:
    operations = {name: _result(NOT_TESTED, reason) for name in PRECISION_CHECKS}
    return {"device": device, "tested": False, "reason_not_tested": reason,
            "operations": operations}


def collect_precision_checks(torch: Any | None, torch_info: dict[str, Any]) -> dict[str, Any]:
    """Operation checks for cpu, cuda (device 0) and mps; unavailable ones are ``not_tested``."""
    if torch is None:
        reason = "PyTorch is not importable"
        return {"cpu": _not_tested("cpu", reason), "cuda": _not_tested("cuda:0", reason),
                "mps": _not_tested("mps", reason)}
    cuda = (torch_info.get("cuda") or {})
    mps = (torch_info.get("mps") or {})
    return {
        "cpu": run_precision_checks(torch, "cpu"),
        "cuda": (run_precision_checks(torch, "cuda:0") if cuda.get("available")
                 else _not_tested("cuda:0", "CUDA is not available")),
        "mps": (run_precision_checks(torch, "mps") if mps.get("available")
                else _not_tested("mps", "MPS is not available")),
    }


# ------------------------------------------------------------------------ backend info
def _check_sync(torch: Any, kind: str) -> dict[str, Any]:
    """Probe whether the device synchronization call exists and runs."""
    module = getattr(torch, kind, None)
    api_present = module is not None and hasattr(module, "synchronize")
    if not api_present:
        return {"api_present": False, **_result(UNSUPPORTED, f"torch.{kind}.synchronize missing")}
    try:
        module.synchronize()
        return {"api_present": True, **_result(SUCCEEDED, f"torch.{kind}.synchronize() ran")}
    except Exception as exc:  # noqa: BLE001
        return {"api_present": True,
                **_result(classify_exception(exc), str(exc)[:300], type(exc).__name__)}


def collect_backend(torch: Any | None, torch_info: dict[str, Any]) -> dict[str, Any]:
    devices = ["cpu"]
    cuda = torch_info.get("cuda") or {}
    mps = torch_info.get("mps") or {}
    devices += [f"cuda:{i}" for i in range(int(cuda.get("device_count") or 0))]
    if mps.get("available"):
        devices.append("mps")
    skip = _result(NOT_TESTED, "device not available")
    sync: dict[str, Any] = {
        "cpu": {"api_present": False, **_result(NOT_APPLICABLE, "CPU timing needs no synchronize")},
        "cuda": {"api_present": False, **skip},
        "mps": {"api_present": False, **skip},
    }
    if torch is not None:
        if cuda.get("available"):
            sync["cuda"] = _check_sync(torch, "cuda")
        if mps.get("available"):
            sync["mps"] = _check_sync(torch, "mps")
    return {
        "available_devices": devices,
        "sdpa_api_available": bool(torch_info.get("sdpa_api_available")),
        "synchronization": sync,
    }


# ------------------------------------------------------------------------ the report
def collect_environment_report(repo_root: Path | None = None) -> dict[str, Any]:
    """Build the full report. Each section is isolated; failures go into ``errors``."""
    root = repo_root if repo_root is not None else Path.cwd()
    errors: list[dict[str, str]] = []

    def safe(section: str, fn: Callable[[], Any], fallback: Any) -> Any:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - isolation of probes is the requirement
            errors.append(_error_entry(section, exc))
            return fallback

    torch, import_error = _load_torch()
    torch_info = safe("torch", lambda: collect_torch(torch, import_error),
                      {"available": torch is not None, "version": None, "import_error": import_error,
                       "num_threads": None, "sdpa_api_available": False, "cuda": None, "mps": None})
    not_collected = {"error": "section failed; see 'errors'"}
    report = {
        "report_type": REPORT_TYPE,
        "report_version": REPORT_VERSION,
        "generated_at_utc": _utc_now(),
        "python": safe("python", collect_python, not_collected),
        "os": safe("os", collect_os, not_collected),
        "cpu": safe("cpu", collect_cpu, not_collected),
        "memory": safe("memory", collect_memory, not_collected),
        "git": safe("git", lambda: collect_git(root), not_collected),
        "torch": torch_info,
        "backend": safe("backend", lambda: collect_backend(torch, torch_info), not_collected),
        "precision_checks": safe("precision_checks",
                                 lambda: collect_precision_checks(torch, torch_info), not_collected),
        "disclaimers": list(DISCLAIMERS),
    }
    report["errors"] = errors
    return report


def write_report(report: dict[str, Any], output: Path) -> Path:
    """Write ``report`` as UTF-8 JSON, creating parent directories. Raises ``OSError``."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return output


# ------------------------------------------------------------------ terminal rendering
def _fmt_bytes(value: int | None) -> str:
    return "unknown" if value is None else f"{value / 2**30:.2f} GiB ({value} bytes)"


def format_report(report: dict[str, Any]) -> str:
    """Render the report as plain ASCII text (safe for Windows consoles)."""
    py, os_, cpu, mem = (report.get(k, {}) for k in ("python", "os", "cpu", "memory"))
    torch_info, backend = report.get("torch", {}), report.get("backend", {})
    git = report.get("git", {})
    lines = [
        "AVFuse-CausalQ environment report",
        f"  generated (UTC) : {report.get('generated_at_utc')}",
        "",
        "[Python / OS]",
        f"  python          : {py.get('version')} ({py.get('implementation')})",
        f"  executable      : {py.get('executable')}",
        f"  os              : {os_.get('platform')}",
        f"  architecture    : {os_.get('architecture')}",
        "",
        "[CPU / memory]",
        f"  cpu             : {cpu.get('description') or 'unknown'}",
        f"  logical cores   : {cpu.get('logical_cores')}",
        f"  physical cores  : {cpu.get('physical_cores')} ({cpu.get('physical_cores_source')})",
        f"  total RAM       : {_fmt_bytes(mem.get('total_bytes'))}",
        f"  available RAM   : {_fmt_bytes(mem.get('available_bytes'))}  [source: {mem.get('source')}]",
        f"  git commit      : {git.get('commit') or 'unavailable'} (dirty: {git.get('dirty')})",
        "",
        "[PyTorch]",
    ]
    if not torch_info.get("available"):
        lines.append(f"  NOT AVAILABLE: {torch_info.get('import_error')}")
    else:
        cuda, mps = torch_info.get("cuda") or {}, torch_info.get("mps") or {}
        lines += [
            f"  version         : {torch_info.get('version')}  (threads: {torch_info.get('num_threads')})",
            f"  SDPA API        : {torch_info.get('sdpa_api_available')}",
            f"  CUDA available  : {cuda.get('available')}  (runtime: {cuda.get('runtime_version')})",
        ]
        for dev in cuda.get("devices", []):
            lines.append(f"    cuda:{dev.get('index')} {dev.get('name', dev.get('error'))}, "
                         f"{_fmt_bytes(dev.get('total_memory_bytes'))}, "
                         f"capability {dev.get('compute_capability')}")
        lines.append(f"  MPS available   : {mps.get('available')}  (built: {mps.get('built')})")
    lines += ["", "[Backend]",
              f"  devices         : {', '.join(backend.get('available_devices', [])) or 'none'}"]
    for kind, entry in (backend.get("synchronization") or {}).items():
        lines.append(f"  sync {kind:<9}: {entry.get('status')}")
    lines += ["", "[Small-operation checks]  (one tiny op per device; NOT model support, NOT timing)"]
    for key, entry in (report.get("precision_checks") or {}).items():
        if not isinstance(entry, dict) or "operations" not in entry:
            lines.append(f"  {key}: unavailable")
            continue
        lines.append(f"  {key} ({entry.get('device')}):")
        for name, res in entry["operations"].items():
            note = f"  -- {res['detail']}" if res.get("status") not in (SUCCEEDED,) and res.get("detail") else ""
            lines.append(f"    {name:<5}: {res.get('status')}{note}")
    if report.get("errors"):
        lines += ["", "[Non-fatal probe errors]"]
        lines += [f"  {e['section']}: {e['error_type']}: {e['message']}" for e in report["errors"]]
    lines += ["", "[Notes]"] + [f"  - {text}" for text in report.get("disclaimers", [])]
    return "\n".join(lines)


# ------------------------------------------------------------------------------- CLI
def build_parser(default_output: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit the Python/OS/CPU/PyTorch environment and write a JSON report. "
                    "Works on CPU-only machines; CUDA and MPS are optional.")
    parser.add_argument("--output", "-o", type=Path, default=default_output,
                        help=f"JSON report path (default: {default_output})")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="do not print the terminal report")
    return parser


def main(argv: Sequence[str] | None = None, *, default_output: Path | None = None,
         repo_root: Path | None = None) -> int:
    """CLI entry point. Returns 0 on success, 1 only if the report cannot be produced/written."""
    args = build_parser(default_output or DEFAULT_OUTPUT).parse_args(argv)
    try:
        report = collect_environment_report(repo_root)
    except Exception as exc:  # noqa: BLE001 - last-resort guard
        print(f"ERROR: could not collect environment report: {exc}", file=sys.stderr)
        return 1
    if not args.quiet:
        print(format_report(report))
    try:
        written = write_report(report, args.output)
    except OSError as exc:
        print(f"ERROR: could not write report to {args.output}: {exc}", file=sys.stderr)
        return 1
    print(f"\nJSON report written to: {written}")
    return 0
