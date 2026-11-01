"""Environment diagnostics tests. None of these need a GPU or any particular machine."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from avfuse.member3_runtime import environment as env

REPO_ROOT = Path(__file__).resolve().parents[2]
TOP_LEVEL_KEYS = {
    "report_type", "report_version", "generated_at_utc", "python", "os", "cpu", "memory", "git",
    "torch", "backend", "precision_checks", "disclaimers", "errors",
}
OPERATIONS = {"fp32", "fp16", "bf16", "sdpa"}
VALID_STATUSES = {env.SUCCEEDED, env.FAILED, env.UNSUPPORTED, env.NOT_TESTED}


@pytest.fixture(scope="module")
def report():
    return env.collect_environment_report(REPO_ROOT)


def test_report_has_required_fields(report):
    assert TOP_LEVEL_KEYS <= set(report)
    assert report["report_type"] == env.REPORT_TYPE
    assert {"version", "executable", "implementation"} <= set(report["python"])
    assert {"system", "release", "version", "architecture", "platform"} <= set(report["os"])
    assert {"description", "logical_cores", "physical_cores"} <= set(report["cpu"])
    assert {"total_bytes", "available_bytes", "source"} <= set(report["memory"])
    assert {"available", "version", "cuda", "mps", "sdpa_api_available"} <= set(report["torch"])
    assert "cpu" in report["backend"]["available_devices"]
    assert set(report["backend"]["synchronization"]) == {"cpu", "cuda", "mps"}


def test_utc_timestamp_format(report):
    assert report["generated_at_utc"].endswith("Z")
    from datetime import datetime
    datetime.strptime(report["generated_at_utc"], "%Y-%m-%dT%H:%M:%S.%fZ")


def test_report_is_json_serializable_and_round_trips(report):
    assert json.loads(json.dumps(report)) == report


def test_cpu_checks_cover_all_operations_with_valid_statuses(report):
    cpu = report["precision_checks"]["cpu"]
    assert cpu["tested"] is True
    assert set(cpu["operations"]) == OPERATIONS
    for result in cpu["operations"].values():
        assert result["status"] in VALID_STATUSES
        assert result["status"] != env.NOT_TESTED
    # fp32 matmul and SDPA must work on any CPU build of a supported PyTorch.
    assert cpu["operations"]["fp32"]["status"] == env.SUCCEEDED
    assert cpu["operations"]["sdpa"]["status"] == env.SUCCEEDED


def _all_keys(node):
    if isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from _all_keys(value)
    elif isinstance(node, list):
        for item in node:
            yield from _all_keys(item)


def test_report_contains_no_timing_or_int8_claim(report):
    keys = {k.lower() for k in _all_keys(report)}
    assert not any(k.endswith("_ms") or "latency" in k or "duration" in k for k in keys)
    assert any("int8" in d.lower() and "simulated" in d.lower() for d in report["disclaimers"])
    assert not any("int8" in k for k in report["precision_checks"]["cpu"]["operations"])


def test_missing_accelerators_are_not_tested_not_errors(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    report = env.collect_environment_report(REPO_ROOT)
    assert report["backend"]["available_devices"] == ["cpu"]
    for key in ("cuda", "mps"):
        entry = report["precision_checks"][key]
        assert entry["tested"] is False and entry["reason_not_tested"]
        assert {r["status"] for r in entry["operations"].values()} == {env.NOT_TESTED}
        assert report["backend"]["synchronization"][key]["status"] == env.NOT_TESTED
    assert report["torch"]["cuda"]["available"] is False
    assert report["errors"] == []


def test_torch_missing_still_produces_report(monkeypatch):
    monkeypatch.setattr(env, "_load_torch", lambda: (None, "ImportError: simulated"))
    report = env.collect_environment_report(REPO_ROOT)
    assert report["torch"]["available"] is False
    assert report["torch"]["import_error"] == "ImportError: simulated"
    assert report["backend"]["available_devices"] == ["cpu"]
    for entry in report["precision_checks"].values():
        assert {r["status"] for r in entry["operations"].values()} == {env.NOT_TESTED}
    json.dumps(report)
    assert "NOT AVAILABLE" in env.format_report(report)


def test_failed_and_unsupported_optional_operations_are_recorded(monkeypatch):
    pytest.importorskip("torch")

    def boom(torch, device):
        raise RuntimeError("driver exploded")

    def not_impl(torch, device):
        raise RuntimeError('"addmm" not implemented for \'BFloat16\'')

    def wrong(torch, device):
        raise env.CheckFailed("numerical mismatch")

    monkeypatch.setitem(env.PRECISION_CHECKS, "fp16", boom)
    monkeypatch.setitem(env.PRECISION_CHECKS, "bf16", not_impl)
    monkeypatch.setitem(env.PRECISION_CHECKS, "sdpa", wrong)
    report = env.collect_environment_report(REPO_ROOT)
    ops = report["precision_checks"]["cpu"]["operations"]
    assert ops["fp16"]["status"] == env.FAILED and ops["fp16"]["error_type"] == "RuntimeError"
    assert ops["bf16"]["status"] == env.UNSUPPORTED
    assert ops["sdpa"]["status"] == env.FAILED
    assert ops["fp32"]["status"] == env.SUCCEEDED  # one failure does not stop the others
    assert report["errors"] == []


@pytest.mark.parametrize("exc,expected", [
    (NotImplementedError("x"), env.UNSUPPORTED),
    (RuntimeError("PyTorch is not linked with support for mps devices"), env.UNSUPPORTED),
    (RuntimeError("something else"), env.FAILED),
    (ValueError("bad"), env.FAILED),
    (env.CheckFailed("mismatch"), env.FAILED),
])
def test_exception_classification(exc, expected):
    assert env.classify_exception(exc) == expected


def test_broken_section_is_isolated(monkeypatch):
    def broken():
        raise OSError("memory probe broke")

    monkeypatch.setattr(env, "collect_memory", broken)
    report = env.collect_environment_report(REPO_ROOT)
    assert report["python"]["version"]
    assert "error" in report["memory"]
    assert [e["section"] for e in report["errors"]] == ["memory"]
    assert report["errors"][0]["error_type"] == "OSError"


def test_main_creates_missing_output_directories(tmp_path, capsys):
    out = tmp_path / "does" / "not" / "exist" / "env.json"
    assert env.main(["--output", str(out)], repo_root=REPO_ROOT) == 0
    assert out.is_file()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert TOP_LEVEL_KEYS <= set(data)
    assert "environment report" in capsys.readouterr().out


def test_main_custom_output_path_and_default_output(tmp_path):
    default = tmp_path / "artifacts" / "runs" / "environment_report.json"
    assert env.main(["--quiet"], default_output=default, repo_root=REPO_ROOT) == 0
    assert default.is_file()
    custom = tmp_path / "my_environment.json"
    assert env.main(["--quiet", "--output", str(custom)], default_output=default,
                    repo_root=REPO_ROOT) == 0
    assert custom.is_file()


def test_main_returns_nonzero_when_output_cannot_be_written(tmp_path, capsys):
    blocker = tmp_path / "file_not_dir"
    blocker.write_text("x", encoding="utf-8")
    assert env.main(["--quiet", "--output", str(blocker / "env.json")], repo_root=REPO_ROOT) == 1
    assert "ERROR" in capsys.readouterr().err


def test_format_report_is_ascii_text(report):
    text = env.format_report(report)
    text.encode("ascii")  # must be printable on any Windows console code page
    assert "Small-operation checks" in text and "NOT model support" in text


def test_script_runs_end_to_end(tmp_path):
    out = tmp_path / "nested" / "script_env.json"
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "check_environment.py"),
         "--quiet", "--output", str(out)],
        capture_output=True, text=True, timeout=300, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(out.read_text(encoding="utf-8"))["report_type"] == env.REPORT_TYPE


def test_psutil_is_optional(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)  # makes `import psutil` raise ImportError
    cpu = env.collect_cpu()
    assert cpu["physical_cores"] is None and "psutil" in cpu["physical_cores_source"]
    assert cpu["logical_cores"] is None or cpu["logical_cores"] >= 1
    memory = env.collect_memory()
    assert memory["source"] != "psutil"
    assert memory["total_bytes"] is None or memory["total_bytes"] > 0
    json.dumps({"cpu": cpu, "memory": memory})
