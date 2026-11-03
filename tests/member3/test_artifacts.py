"""Run identity, benchmark records, artifact writing and integrity (Segment D).

Synthetic temporary content only: no datasets, checkpoints or GPUs. Passing these tests shows
the software works; it is not a research result.
"""
import copy
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from avfuse.member3_runtime import artifacts as art
from avfuse.member3_runtime import provenance as prov
from avfuse.member3_runtime.benchmark import benchmark_callable, summarize_latencies
from avfuse.member3_runtime.stage_profiler import (
    StageSettings,
    profile_pipeline_stages,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = json.loads((REPO_ROOT / "contracts/schemas/benchmark.schema.json").read_text("utf-8"))
NOW = datetime(2026, 5, 1, 12, 30, 45, 123000, tzinfo=timezone.utc)
SHAPES = {"q": [1, 2, 3, 4], "k": [1, 2, 5, 4]}


def fake_report(**overrides) -> dict:
    report = {
        "report_type": "avfuse_environment_report", "report_version": "1.0",
        "generated_at_utc": "2026-05-01T12:30:45.000000Z",
        "python": {"version": "3.12.0"}, "os": {"platform": "Test-OS-1.0"},
        "cpu": {"description": "Test CPU", "logical_cores": 4, "physical_cores": 2},
        "git": {"commit": None, "dirty": None},
        "torch": {
            "available": True, "version": "2.5.0", "num_threads": 4,
            "cuda": {"available": True, "runtime_version": "12.1", "device_count": 1,
                     "devices": [{"index": 0, "name": "Test GPU", "total_memory_bytes": 1024,
                                  "compute_capability": "8.0", "multi_processor_count": 10}]},
            "mps": {"available": False, "host_architecture": "arm64"},
        },
        "errors": [],
    }
    report.update(overrides)
    return report


def make_config(**output) -> dict:
    cfg = json.loads((REPO_ROOT / "configs/benchmark.example.json").read_text("utf-8"))
    cfg["experiment_id"] = "test-experiment"
    cfg["output"].update(output)
    return cfg


def make_ctx(tmp_path, config=None, **kw):
    kw.setdefault("allow_unset_inputs", True)
    kw.setdefault("environment_report", fake_report())
    kw.setdefault("repo_root", tmp_path)  # not a git checkout -> git "unavailable"
    kw.setdefault("now", NOW)
    return art.create_run_context(config or make_config(), runs_root=tmp_path / "runs", **kw)


def measurement(device="cpu", samples=(1.0, 2.0, 3.0), method="linear", **overrides) -> dict:
    stats = summarize_latencies(list(samples), method)
    m = {
        "repeats": len(samples), "p50_ms": stats["p50_ms"], "p95_ms": stats["p95_ms"],
        "peak_memory_bytes": None, "device": device, "warmup": 2,
        "timing_samples_ms": list(samples), "mean_ms": stats["mean_ms"],
        "median_ms": stats["median_ms"], "min_ms": stats["min_ms"], "max_ms": stats["max_ms"],
        "percentile_method": method,
        "synchronization": "not_required_cpu" if device == "cpu" else "cuda_synchronize",
        "peak_memory_reserved_bytes": None, "memory_baseline_bytes": None,
        "memory_method": "not_measured", "memory_note": "Peak memory is not measured on CPU.",
    }
    m.update(overrides)
    return m


def record(ctx, config="reference", m=None, **kw) -> dict:
    fields = {"config": config, "scope": "operator", "backend": "sdpa", "batch_size": 1,
              "execution_dtype": "float32", "input_boundary": "synthetic_tensors",
              "input_shapes": SHAPES}
    fields.update(kw)
    return ctx.build_record(m or measurement(), **fields)


def four_records(ctx):
    return [record(ctx, c) for c in art.PRECISION_CONFIGS]


def all_files(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*"))
            if p.is_file()}


# =============================================================================== run identity
def test_run_ids_are_safe_unique_and_not_content_hashes():
    ids = {art.generate_run_id() for _ in range(2000)}
    assert len(ids) == 2000
    for rid in list(ids)[:50]:
        assert re.fullmatch(r"run-\d{8}T\d{6}Z-[0-9a-f]{16}", rid)
        assert art.validate_run_id(rid) == rid
    assert art.generate_run_id(NOW).startswith("run-20260501T123045Z-")


@pytest.mark.parametrize("bad", ["", "../x", "a/b", "a\\b", ".hidden", "a..b", "x" * 200,
                                 "with space", "C:evil", None])
def test_unsafe_run_ids_rejected(bad):
    with pytest.raises(ValueError):
        art.validate_run_id(bad)


def test_utc_timestamps_are_aware_and_end_in_z():
    assert art.utc_timestamp(NOW) == "2026-05-01T12:30:45.123Z"
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z", art.utc_timestamp())
    plus_two = datetime(2026, 5, 1, 14, 30, 45, tzinfo=timezone(timedelta(hours=2)))
    assert art.utc_timestamp(plus_two) == "2026-05-01T12:30:45.000Z"
    with pytest.raises(ValueError, match="Naive"):
        art.utc_timestamp(datetime(2026, 5, 1, 12, 0, 0))
    assert art.parse_utc_timestamp("2026-05-01T12:30:45Z").tzinfo is not None
    for bad in ("2026-05-01 12:30:45", "2026-05-01T12:30:45+00:00", "2026-13-45T00:00:00Z"):
        with pytest.raises(ValueError):
            art.parse_utc_timestamp(bad)


# ============================================================================== run context
def test_run_directory_and_initial_artifacts(tmp_path):
    ctx = make_ctx(tmp_path, run_id="run-abc")
    assert ctx.run_id == "run-abc" and ctx.run_dir == tmp_path / "runs" / "run-abc"
    assert sorted(all_files(ctx.run_dir)) == ["environment_report.json", "provenance.json",
                                              "resolved_config.json"]
    assert ctx.created_at_utc == "2026-05-01T12:30:45.123Z"
    prov_doc = json.loads((ctx.run_dir / "provenance.json").read_text())
    assert prov_doc["run_id"] == "run-abc" and prov_doc["artifact_type"] == "avfuse_run_provenance"
    assert "not a content hash" in prov_doc["run_id_note"]


def test_generated_run_id_is_used_everywhere(tmp_path):
    ctx = make_ctx(tmp_path)
    ctx.write_benchmark_records(four_records(ctx))
    ctx.finalize()
    docs = [json.loads((ctx.run_dir / n).read_text()) for n in
            ("provenance.json", "run_manifest.json", "integrity/artifact_hashes.json")]
    assert {d["run_id"] for d in docs} == {ctx.run_id}
    lines = (ctx.run_dir / "benchmark_records.jsonl").read_text().splitlines()
    assert {json.loads(line)["run_id"] for line in lines} == {ctx.run_id}


def test_existing_run_directory_is_rejected(tmp_path):
    ctx = make_ctx(tmp_path, run_id="run-dup")
    before = all_files(ctx.run_dir)
    with pytest.raises(art.ArtifactExistsError, match="already exists"):
        make_ctx(tmp_path, run_id="run-dup")
    assert all_files(ctx.run_dir) == before


def test_overwrite_true_reuses_run_directory(tmp_path):
    cfg = make_config(overwrite=True)
    make_ctx(tmp_path, cfg, run_id="run-ow")
    again = make_ctx(tmp_path, cfg, run_id="run-ow")
    assert (again.run_dir / "provenance.json").is_file()


def test_resolved_config_snapshot_and_hash(tmp_path):
    cfg = make_config()
    ctx = make_ctx(tmp_path, cfg)
    snapshot = json.loads((ctx.run_dir / "resolved_config.json").read_text())
    assert snapshot == cfg
    assert prov.sha256_json(snapshot) == ctx.config_sha256
    reordered = json.loads(json.dumps(cfg, sort_keys=True))
    assert prov.sha256_json(reordered) == ctx.config_sha256  # key order irrelevant
    assert ctx.provenance["hashes"]["config"]["sha256"] == ctx.config_sha256
    changed = make_config()
    changed["reproducibility"]["seed"] = 99
    assert prov.sha256_json(changed) != ctx.config_sha256


def test_snapshot_and_environment_report_can_be_disabled(tmp_path):
    cfg = make_config()
    cfg["reproducibility"].update(snapshot_config=False, record_environment_report=False)
    ctx = make_ctx(tmp_path, cfg)
    assert sorted(all_files(ctx.run_dir)) == ["provenance.json"]
    assert ctx.environment_report_ref is None
    assert record(ctx)["environment"]["environment_report_ref"] is None


def test_invalid_config_rejected_without_creating_run(tmp_path):
    cfg = make_config()
    cfg["output"]["timing_samples_storage"] = "somewhere_else"
    with pytest.raises(art.ArtifactError, match="Invalid benchmark config"):
        make_ctx(tmp_path, cfg)
    assert not (tmp_path / "runs").exists()


def test_environment_report_is_stored_and_referenced(tmp_path):
    ctx = make_ctx(tmp_path, environment_report=None)  # real collect_environment_report
    stored = json.loads((ctx.run_dir / "environment_report.json").read_text())
    assert stored["report_type"] == "avfuse_environment_report"
    assert ctx.environment_report_ref == "environment_report.json"
    env = record(ctx)["environment"]
    assert env["environment_report_ref"] == "environment_report.json"
    assert not Path(env["environment_report_ref"]).is_absolute()
    summary = ctx.provenance["environment"]
    assert summary["python_version"] and summary["torch_version"] and summary["os_platform"]
    assert any("not performance measurements" in d for d in ctx.provenance["disclaimers"])
    assert any("not hardware INT8" in d for d in ctx.provenance["disclaimers"])


def test_provenance_environment_summary_uses_report_without_inventing(tmp_path):
    ctx = make_ctx(tmp_path)
    summary = ctx.provenance["environment"]
    assert summary["cuda_runtime_version"] == "12.1"
    assert summary["cuda_device_names"] == ["Test GPU"]
    assert summary["mps_available"] is False
    assert "per benchmark record" in summary["note"]


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_git_commit_and_dirty_state_captured(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
        "GIT_CONFIG_GLOBAL": os.devnull,
           "GIT_CONFIG_SYSTEM": os.devnull}

    def git(*args):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, env=env)

    git("init", "-q")
    (repo / "f.txt").write_text("1")
    git("add", ".")
    git("commit", "-q", "-m", "init")
    clean = art.capture_git_provenance(repo)
    assert re.fullmatch(r"[0-9a-f]{40}", clean["commit"]) and clean["dirty"] is False
    assert clean["state"] == "recorded" and clean["warning"] is None
    (repo / "f.txt").write_text("changed")
    dirty = art.capture_git_provenance(repo)
    assert dirty["commit"] == clean["commit"] and dirty["dirty"] is True
    assert "uncommitted" in dirty["warning"]

    ctx = make_ctx(tmp_path, repo_root=repo)
    env_block = record(ctx)["environment"]
    assert env_block["git_commit"] == clean["commit"] and env_block["git_dirty"] is True
    assert ctx.provenance["git"]["dirty"] is True


def test_git_unavailable_is_explicit_null(tmp_path, monkeypatch):
    plain = tmp_path / "not_a_repo"
    plain.mkdir()
    info = art.capture_git_provenance(plain)
    assert info == {"state": "unavailable", "commit": None, "dirty": None,
                    "warning": "Git commit unavailable; the code version is not identified."}

    def no_git(*a, **k):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", no_git)
    assert art.capture_git_provenance(plain)["commit"] is None
    ctx = make_ctx(tmp_path, repo_root=plain)
    env_block = record(ctx)["environment"]
    assert env_block["git_commit"] is None and env_block["git_dirty"] is None


def test_record_git_commit_false_is_not_recorded(tmp_path):
    cfg = make_config()
    cfg["reproducibility"]["record_git_commit"] = False
    ctx = make_ctx(tmp_path, cfg)
    assert ctx.git["state"] == "not_recorded" and ctx.git["commit"] is None


def test_hashing_requested_without_input_fails_before_creating_run(tmp_path):
    with pytest.raises(prov.MissingInputError, match="Checkpoint"):
        make_ctx(tmp_path, allow_unset_inputs=False)
    with pytest.raises(prov.MissingInputError, match="does not exist"):
        make_ctx(tmp_path, checkpoint_path=tmp_path / "gone.pt")
    cfg = make_config()
    cfg["reproducibility"]["hash_checkpoint"] = False
    with pytest.raises(prov.MissingInputError, match="Input-cache"):
        make_ctx(tmp_path, cfg, allow_unset_inputs=False)
    assert not (tmp_path / "runs").exists()  # no half-made run directory


def test_unset_inputs_are_recorded_as_not_provided_never_hashed(tmp_path):
    ctx = make_ctx(tmp_path)
    hashes = ctx.provenance["hashes"]
    assert hashes["checkpoint"] == {"status": "not_provided"}
    assert hashes["feature_cache_content"] == {"status": "not_provided"}
    rec = record(ctx)
    assert rec["workload"]["checkpoint_sha256"] is None
    assert rec["workload"]["input_identity_sha256"] is None


def test_hashing_disabled_is_distinct_from_not_provided(tmp_path):
    cfg = make_config()
    cfg["reproducibility"].update(hash_checkpoint=False, hash_input_cache=False)
    ctx = make_ctx(tmp_path, cfg, allow_unset_inputs=False)
    assert ctx.provenance["hashes"]["checkpoint"] == {"status": "hashing_disabled"}
    assert ctx.provenance["hashes"]["feature_cache_content"] == {"status": "hashing_disabled"}


def test_checkpoint_and_cache_hashes_flow_into_records_without_copying_data(tmp_path):
    ckpt = tmp_path / "inputs" / "model.ckpt"
    ckpt.parent.mkdir()
    ckpt.write_bytes(b"synthetic checkpoint bytes")
    cache = tmp_path / "inputs" / "cache"
    cache.mkdir()
    (cache / "f1.bin").write_bytes(b"feat1")
    (cache / "f2.bin").write_bytes(b"feat2")
    manifest = tmp_path / "inputs" / "manifest.csv"
    manifest.write_bytes(b"id\n1\n")
    ctx = make_ctx(tmp_path, checkpoint_path=ckpt, cache_dir=cache, cache_manifest_path=manifest)
    rec = record(ctx)
    assert rec["workload"]["checkpoint_sha256"] == prov.sha256_file(ckpt)
    assert rec["workload"]["input_identity_sha256"] == prov.sha256_directory(cache)
    hashes = ctx.provenance["hashes"]
    assert hashes["cache_manifest"]["sha256"] == prov.sha256_file(manifest)
    assert hashes["cache_manifest"]["sha256"] != hashes["feature_cache_content"]["sha256"]
    # absolute paths never recorded; inputs never copied into the run
    assert str(tmp_path) not in json.dumps(ctx.provenance["hashes"])
    stored = all_files(ctx.run_dir)
    assert not any(name.endswith((".ckpt", ".bin", ".csv", ".npy", ".pt", ".mp4"))
                   for name in stored)
    assert sum(len(b) for b in stored.values()) < 200_000


def test_manifest_only_cache_yields_null_identity_with_explanation(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_bytes(b"id\n1\n")
    ctx = make_ctx(tmp_path, cache_manifest_path=manifest)
    assert ctx.provenance["hashes"]["feature_cache_content"]["status"] == "not_provided"
    assert "not verified" in ctx.provenance["hashes"]["feature_cache_content"]["note"]
    rec = record(ctx)
    assert rec["workload"]["input_identity_sha256"] is None
    ext = rec["extensions"]["provenance"]
    assert ext["cache_manifest_sha256"] == prov.sha256_file(manifest)
    assert "not verified" in ext["input_identity_note"]


# ============================================================================ record building
def schema_errors(rec) -> list:
    return list(Draft202012Validator(SCHEMA, format_checker=FormatChecker()).iter_errors(rec))


def test_record_is_valid_schema_v02_and_maps_measurement_fields(tmp_path):
    ctx = make_ctx(tmp_path)
    m = measurement()
    snapshot = copy.deepcopy(m)
    rec = record(ctx, m=m, timestamp_utc="2026-05-01T12:31:00.000Z")
    assert schema_errors(rec) == []
    assert m == snapshot  # input measurement untouched
    assert rec["schema_version"] == "0.2" and rec["run_id"] == ctx.run_id
    assert rec["experiment_id"] == "test-experiment"
    for key in ("device", "warmup", "repeats", "mean_ms", "median_ms", "p50_ms", "p95_ms",
                "timing_samples_ms", "peak_memory_bytes", "memory_method"):
        assert rec[key] == m[key]
    assert rec["measurement"] == {"percentile_method": "linear",
                                  "synchronization": "not_required_cpu"}
    for key in ("percentile_method", "synchronization", "min_ms", "max_ms", "memory_note",
                "peak_memory_reserved_bytes", "memory_baseline_bytes"):
        assert key not in rec  # never top-level
    assert rec["peak_memory_bytes"] is None and rec["memory_method"] == "not_measured"
    assert rec["extensions"]["benchmark_engine"]["memory_note"] == m["memory_note"]
    assert rec["timestamp_utc"].endswith("Z")


def test_builder_accepts_real_benchmark_callable_output(tmp_path):
    ctx = make_ctx(tmp_path)
    real = benchmark_callable(lambda: sum(range(50)), "cpu", warmup=1, repeats=7)
    rec = record(ctx, m=real)
    assert schema_errors(rec) == [] and len(rec["timing_samples_ms"]) == 7
    assert rec["peak_memory_bytes"] is None  # CPU: unavailable, never zero


@pytest.mark.parametrize("config,mode,simulated", [
    ("reference", "native_float", False), ("low_precision", "simulated_int8", True),
    ("restore_value_output", "simulated_int8", True), ("restore_query", "simulated_int8", True),
])
def test_precision_mapping_for_all_four_configs(tmp_path, config, mode, simulated):
    ctx = make_ctx(tmp_path)
    rec = record(ctx, config, execution_dtype="float16")
    p = rec["precision"]
    assert rec["config"] == config and p["mode"] == mode and p["simulated"] is simulated
    assert p["execution_dtype"] == "float16" and p["real_int8_kernel"] is False
    if simulated:
        assert "Simulated" in p["disclosure"] and "no integer INT8 kernel" in p["disclosure"]
        assert "not INT8 hardware speedups" in p["disclosure"]
    else:
        assert "natively" in p["disclosure"] and "INT8" not in p["disclosure"]
    assert schema_errors(rec) == []


def test_execution_dtype_accepts_torch_dtype_and_rejects_unknown(tmp_path):
    import torch

    ctx = make_ctx(tmp_path)
    assert record(ctx, execution_dtype=torch.bfloat16)["precision"]["execution_dtype"] == "bfloat16"
    with pytest.raises(art.BenchmarkValidationError):
        record(ctx, execution_dtype="int8")  # no real INT8 execution dtype exists in the schema
    with pytest.raises(ValueError, match="Unknown config"):
        record(ctx, "int8_real")


def test_cuda_record_maps_device_metadata_and_memory(tmp_path):
    ctx = make_ctx(tmp_path)
    m = measurement("cuda:0", peak_memory_bytes=4096, memory_method="cuda_max_memory_allocated",
                    peak_memory_reserved_bytes=8192, memory_baseline_bytes=1024, memory_note=None)
    rec = record(ctx, m=m)
    meta = rec["environment"]["device_metadata"]
    assert meta["type"] == "cuda" and meta["name"] == "Test GPU"
    assert meta["properties"]["compute_capability"] == "8.0"
    assert rec["measurement"]["synchronization"] == "cuda_synchronize"
    assert rec["peak_memory_bytes"] == 4096
    assert rec["extensions"]["benchmark_engine"] == {"peak_memory_reserved_bytes": 8192,
                                                     "memory_baseline_bytes": 1024,
                                                     "memory_note": None}


def test_device_names_are_never_inferred(tmp_path):
    ctx = make_ctx(tmp_path, environment_report=fake_report())
    mps = record(ctx, m=measurement("mps", synchronization="mps_synchronize"))
    meta = mps["environment"]["device_metadata"]
    assert meta["type"] == "mps" and meta["name"] is None
    assert meta["properties"] == {"host_architecture": "arm64"}
    # a cuda index not in the report -> no name rather than a guess
    unknown = record(ctx, m=measurement("cuda:3"))
    assert unknown["environment"]["device_metadata"]["name"] is None
    no_cpu_name = fake_report(cpu={"description": None})
    ctx2 = make_ctx(tmp_path / "b", environment_report=no_cpu_name)
    assert record(ctx2)["environment"]["device_metadata"]["name"] is None


def test_environment_fields_come_from_report(tmp_path):
    rec = record(make_ctx(tmp_path))
    env = rec["environment"]
    assert (env["python_version"], env["torch_version"], env["os_platform"]) == (
        "3.12.0", "2.5.0", "Test-OS-1.0")
    assert env["device_metadata"]["name"] == "Test CPU"
    broken = fake_report(torch={"available": False, "version": None})
    with pytest.raises(ValueError, match="lacks"):
        record(make_ctx(tmp_path / "c", environment_report=broken))


def test_missing_hashes_are_null_and_kinds_are_checked(tmp_path):
    ctx = make_ctx(tmp_path)
    assert record(ctx)["workload"]["checkpoint_sha256"] is None
    wrong = prov.hash_config({"a": 1})
    with pytest.raises(ValueError, match="checkpoint"):
        record(ctx, checkpoint=wrong)
    with pytest.raises(ValueError, match="input_identity"):
        record(ctx, input_identity=wrong)
    with pytest.raises(ValueError, match="reserved"):
        record(ctx, extensions={"benchmark_engine": {}})


def test_builder_rejects_incomplete_measurement(tmp_path):
    ctx = make_ctx(tmp_path)
    m = measurement()
    del m["synchronization"]
    with pytest.raises(ValueError, match="synchronization"):
        record(ctx, m=m)


# ======================================================================== cross-field validation
def valid_record(tmp_path):
    return record(make_ctx(tmp_path))


def expect_invalid(rec, fragment, **kw):
    with pytest.raises(art.BenchmarkValidationError) as info:
        art.validate_benchmark_records([rec], **kw)
    assert fragment in "\n".join(info.value.errors), info.value.errors


def test_sample_count_must_equal_repeats(tmp_path):
    rec = valid_record(tmp_path)
    rec["repeats"] = 4
    expect_invalid(rec, "3 timing samples but repeats == 4")


def test_p50_must_not_exceed_p95(tmp_path):
    rec = valid_record(tmp_path)
    rec["p50_ms"], rec["p95_ms"] = 3.0, 2.0
    expect_invalid(rec, "p50_ms")


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_samples_must_be_finite_and_non_negative(tmp_path, bad):
    rec = valid_record(tmp_path)
    rec["timing_samples_ms"][0] = bad
    expect_invalid(rec, "")


def test_statistics_must_match_samples_and_declared_method(tmp_path):
    ctx = make_ctx(tmp_path)
    samples = (1.0, 2.0, 3.0, 4.0, 10.0, 12.0)
    linear = measurement(samples=samples, method="linear")
    lower = measurement(samples=samples, method="lower")
    assert linear["p95_ms"] != lower["p95_ms"]
    rec = record(ctx, m=linear)
    rec["measurement"]["percentile_method"] = "lower"  # declared method contradicts the numbers
    expect_invalid(rec, "p95_ms")
    ok = record(ctx, m=lower)
    assert ok["measurement"]["percentile_method"] == "lower"
    tampered = valid_record(tmp_path / "t")
    tampered["mean_ms"] = 99.0
    expect_invalid(tampered, "mean_ms")


def test_device_and_synchronization_consistency(tmp_path):
    ctx = make_ctx(tmp_path)
    with pytest.raises(art.BenchmarkValidationError, match="ynchronization"):
        record(ctx, m=measurement("cuda:0", synchronization="not_required_cpu"))
    with pytest.raises(art.BenchmarkValidationError, match="ynchronization"):
        record(ctx, m=measurement("cpu", synchronization="cuda_synchronize"))
    rec = valid_record(tmp_path / "x")
    rec["environment"]["device_metadata"]["type"] = "cuda"
    expect_invalid(rec, "device_metadata")
    disabled = record(ctx, m=measurement("cpu", synchronization="disabled"))
    assert disabled["measurement"]["synchronization"] == "disabled"


def test_memory_method_and_value_consistency(tmp_path):
    ctx = make_ctx(tmp_path)
    cases = [
        (measurement(peak_memory_bytes=10), "memory_method"),  # value but not_measured
        (measurement("cuda:0", memory_method="cuda_max_memory_allocated"), "memory_method"),
        (measurement(peak_memory_bytes=10, memory_method="cuda_max_memory_allocated"),
         "memory_method"),  # CUDA memory claimed on a CPU device
    ]
    for m, fragment in cases:
        with pytest.raises(art.BenchmarkValidationError) as info:
            record(ctx, m=m)
        assert fragment in "\n".join(info.value.errors), (fragment, info.value.errors)


def test_zero_peak_cuda_memory_is_a_valid_measured_value(tmp_path):
    ctx = make_ctx(tmp_path)
    rec = record(
        ctx,
        m=measurement("cuda:0", peak_memory_bytes=0,
                      memory_method="cuda_max_memory_allocated"),
    )
    art.validate_benchmark_records([rec])
    assert rec["peak_memory_bytes"] == 0
    assert rec["memory_method"] == "cuda_max_memory_allocated"


def test_scope_boundary_backend_combinations(tmp_path):
    ctx = make_ctx(tmp_path)
    # The schema rejects these first, so the message names the offending field.
    for kw, fragment in [
        ({"backend": "none"}, "backend"),
        ({"input_boundary": "raw_media"}, "input_boundary"),
        ({"scope": "preprocessing_once"}, "backend"),
        ({"scope": "cached_feature_model", "input_boundary": "raw_media"},
         "input_boundary"),
    ]:
        with pytest.raises(art.BenchmarkValidationError) as info:
            record(ctx, **kw)
        assert fragment in "\n".join(info.value.errors), (kw, info.value.errors)
    ok = record(ctx, scope="cached_feature_model", backend="sdpa",
                input_boundary="cached_encoder_tokens", input_shapes=None)
    assert ok["scope"] == "cached_feature_model"
    pre = record(ctx, scope="preprocessing_once", backend="none", input_boundary="raw_media",
                 input_shapes=None)
    assert pre["backend"] == "none"


def test_cross_field_rules_hold_independently_of_the_schema(tmp_path):
    """Call the Python rules directly on schema-bypassing records, so they cannot be silently
    relying on the JSON Schema to catch the same problem."""
    base = valid_record(tmp_path)
    mutations = [
        (lambda r: r.update(scope="operator", backend="none"), "operator scope requires"),
        (lambda r: r["workload"].update(input_boundary="raw_media"), "raw_media"),
        (lambda r: r.update(scope="preprocessing_once"), "preprocessing_once requires"),
        (lambda r: r.update(peak_memory_bytes=5), "exactly when"),
        (lambda r: r.update(memory_method="cuda_max_memory_allocated", peak_memory_bytes=5),
         "non-CUDA"),
        (lambda r: r["measurement"].update(synchronization="cuda_synchronize"),
         "inconsistent with device"),
    ]
    for mutate, fragment in mutations:
        rec = copy.deepcopy(base)
        mutate(rec)
        errors: list[str] = []
        art._check_record(rec, "record[0]", rec["timing_samples_ms"], errors)
        assert any(fragment in e for e in errors), (fragment, errors)


def test_real_int8_kernel_and_mode_mapping_enforced(tmp_path):
    rec = valid_record(tmp_path)
    rec["precision"]["real_int8_kernel"] = True
    expect_invalid(rec, "real_int8_kernel")
    rec = valid_record(tmp_path / "a")
    rec["precision"].update(mode="simulated_int8", simulated=True,
                            disclosure="Simulated int8")
    expect_invalid(rec, "requires precision.mode 'native_float'")
    ctx = make_ctx(tmp_path / "b")
    lowp = record(ctx, "low_precision")
    lowp["precision"]["disclosure"] = "Runs fast in hardware."
    expect_invalid(lowp, "must state that INT8 is simulated")


def test_unknown_top_level_fields_rejected(tmp_path):
    rec = valid_record(tmp_path)
    rec["percentile_method"] = "linear"
    expect_invalid(rec, "percentile_method")
    rec2 = valid_record(tmp_path / "b")
    rec2["labels"] = [0, 1]
    expect_invalid(rec2, "labels")


def test_labels_and_predictions_never_allowed_in_extensions(tmp_path):
    ctx = make_ctx(tmp_path)
    with pytest.raises(art.BenchmarkValidationError, match="labels"):
        record(ctx, extensions={"debug": {"labels": [1, 0]}})
    assert "extensions" in record(ctx, extensions={"debug": {"note": "fine"}})


def test_timestamp_format_is_checked(tmp_path):
    for bad in ("2026-05-01 12:00:00", "2026-05-01T12:00:00+00:00", "2026-13-45T25:61:61Z",
                "2026-05-01T12:00:00"):
        rec = valid_record(tmp_path)
        rec["timestamp_utc"] = bad
        with pytest.raises(art.BenchmarkValidationError):
            art.validate_benchmark_records([rec])
    rec = valid_record(tmp_path)
    rec["timestamp_utc"] = "2026-05-01T12:00:00Z"
    art.validate_benchmark_records([rec])


def test_duplicates_and_mixed_runs_rejected(tmp_path):
    ctx = make_ctx(tmp_path)
    a = record(ctx)
    b = record(ctx, m=measurement(samples=(5.0, 6.0, 7.0)))  # same series, different numbers
    with pytest.raises(art.DuplicateRecordError, match="duplicates record"):
        art.validate_benchmark_records([a, b])
    other = copy.deepcopy(record(ctx, "low_precision"))
    other["run_id"] = "another-run"
    with pytest.raises(art.BenchmarkValidationError, match="mix several run_ids"):
        art.validate_benchmark_records([a, other])
    with pytest.raises(art.BenchmarkValidationError, match="!= expected"):
        art.validate_benchmark_records([a], run_id="expected-run")
    # distinct series (different batch size / shapes) are not duplicates
    c = record(ctx, batch_size=2)
    d = record(ctx, input_shapes={"q": [1, 2, 9, 4], "k": [1, 2, 5, 4]})
    art.validate_benchmark_records([a, c, d])


def test_all_problems_are_reported_not_just_the_first(tmp_path):
    rec = valid_record(tmp_path)
    rec["repeats"] = 9
    rec["p50_ms"], rec["p95_ms"] = 9.0, 1.0
    with pytest.raises(art.BenchmarkValidationError) as info:
        art.validate_benchmark_records([rec])
    assert len(info.value.errors) >= 2


# ========================================================================== artifact writers
def test_json_write_is_stable_utf8_and_creates_parents(tmp_path):
    dest = tmp_path / "a" / "b" / "c.json"
    info = art.write_json_artifact(dest, {"z": 1, "a": "caf\u00e9"})
    text = dest.read_text(encoding="utf-8")
    assert text == '{\n  "a": "caf\u00e9",\n  "z": 1\n}\n'
    assert info["size_bytes"] == len(text.encode()) and info["sha256"] == prov.sha256_file(dest)
    assert not list(dest.parent.glob(".*tmp"))


def test_json_write_rejects_non_finite_without_creating_file(tmp_path):
    dest = tmp_path / "bad.json"
    with pytest.raises(ValueError):
        art.write_json_artifact(dest, {"x": float("nan")})
    assert not dest.exists() and not list(tmp_path.iterdir())


def test_no_clobber_and_overwrite_behaviour(tmp_path):
    dest = tmp_path / "x.json"
    art.write_json_artifact(dest, {"v": 1})
    with pytest.raises(art.ArtifactExistsError):
        art.write_json_artifact(dest, {"v": 2})
    assert json.loads(dest.read_text()) == {"v": 1}
    art.write_json_artifact(dest, {"v": 3}, overwrite=True)
    assert json.loads(dest.read_text()) == {"v": 3}
    assert [p.name for p in tmp_path.iterdir()] == ["x.json"]


def test_failed_write_leaves_no_artifact_or_temp_file(tmp_path, monkeypatch):
    dest = tmp_path / "x.json"
    monkeypatch.setattr(os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        art.write_json_artifact(dest, {"v": 1})
    assert list(tmp_path.iterdir()) == []


def test_failed_overwrite_keeps_previous_valid_file(tmp_path, monkeypatch):
    dest = tmp_path / "x.json"
    art.write_json_artifact(dest, {"v": "old"})
    monkeypatch.setattr(os, "replace", lambda *a: (_ for _ in ()).throw(OSError("boom")))
    with pytest.raises(OSError):
        art.write_json_artifact(dest, {"v": "new"}, overwrite=True)
    assert json.loads(dest.read_text()) == {"v": "old"}
    assert [p.name for p in tmp_path.iterdir()] == ["x.json"]


def test_write_refuses_symlink_and_directory_destinations(tmp_path):
    target = tmp_path / "t.json"
    target.write_text("{}")
    link = tmp_path / "link.json"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unsupported")
    with pytest.raises(art.UnsafePathError):
        art.write_json_artifact(link, {"v": 1}, overwrite=True)
    with pytest.raises(art.ArtifactError, match="directory"):
        art.write_json_artifact(tmp_path, {"v": 1}, overwrite=True)


def test_jsonl_one_valid_record_per_line(tmp_path):
    ctx = make_ctx(tmp_path)
    result = ctx.write_benchmark_records(four_records(ctx))
    raw = (ctx.run_dir / "benchmark_records.jsonl").read_bytes()
    assert raw.endswith(b"\n") and result["record_count"] == 4
    lines = raw.decode("utf-8").splitlines()
    assert len(lines) == 4
    parsed = [json.loads(line) for line in lines]
    assert [p["config"] for p in parsed] == list(art.PRECISION_CONFIGS)  # order preserved
    for p, line in zip(parsed, lines, strict=True):
        assert schema_errors(p) == []
        assert line == art.serialize_jsonl_line(p) and "\n" not in line
    assert "timing_samples_ms" in parsed[0] and "timing_samples_ref" not in parsed[0]
    assert not (ctx.run_dir / "timing_samples").exists()


def test_invalid_record_in_batch_means_nothing_is_written(tmp_path):
    ctx = make_ctx(tmp_path, make_config(timing_samples_storage="separate_file"))
    records = four_records(ctx)
    records[3]["p50_ms"], records[3]["p95_ms"] = 9.0, 1.0
    before = all_files(ctx.run_dir)
    with pytest.raises(art.BenchmarkValidationError):
        ctx.write_benchmark_records(records)
    assert all_files(ctx.run_dir) == before
    assert not (ctx.run_dir / "timing_samples").exists()
    with pytest.raises(ValueError, match="No benchmark records"):
        ctx.write_benchmark_records([])


def test_duplicate_records_not_silently_inserted(tmp_path):
    ctx = make_ctx(tmp_path)
    r = record(ctx)
    with pytest.raises(art.DuplicateRecordError):
        ctx.write_benchmark_records([r, copy.deepcopy(r)])
    assert not (ctx.run_dir / "benchmark_records.jsonl").exists()


def test_jsonl_no_overwrite_and_overwrite(tmp_path):
    ctx = make_ctx(tmp_path)
    ctx.write_benchmark_records(four_records(ctx)[:2])
    first = (ctx.run_dir / "benchmark_records.jsonl").read_bytes()
    with pytest.raises(art.ArtifactExistsError):
        art.write_benchmark_records(four_records(ctx), ctx.run_dir, run_id=ctx.run_id)
    assert (ctx.run_dir / "benchmark_records.jsonl").read_bytes() == first

    ow = make_ctx(tmp_path / "ow", make_config(overwrite=True))
    ow.write_benchmark_records(four_records(ow)[:2])
    ow.write_benchmark_records(four_records(ow))
    assert len((ow.run_dir / "benchmark_records.jsonl").read_text().splitlines()) == 4


def test_inline_storage_keeps_samples_in_the_record(tmp_path):
    ctx = make_ctx(tmp_path, make_config(timing_samples_storage="inline"))
    result = ctx.write_benchmark_records([record(ctx)])
    assert result["sample_paths"] == []
    line = json.loads((ctx.run_dir / "benchmark_records.jsonl").read_text())
    assert line["timing_samples_ms"] == [1.0, 2.0, 3.0] and "timing_samples_ref" not in line


def test_separate_timing_samples_use_relative_deterministic_refs(tmp_path):
    ctx = make_ctx(tmp_path, make_config(timing_samples_storage="separate_file"))
    recs = four_records(ctx)
    result = ctx.write_benchmark_records(recs)
    lines = [json.loads(x) for x in
             (ctx.run_dir / "benchmark_records.jsonl").read_text().splitlines()]
    refs = [line["timing_samples_ref"] for line in lines]
    assert len(set(refs)) == 4 and sorted(refs) == result["sample_paths"]
    for line, original, ref in zip(lines, recs, refs, strict=True):
        assert "timing_samples_ms" not in line
        assert re.fullmatch(r"timing_samples/operator-[a-z_]+-[0-9a-f]{16}\.json", ref)
        assert not Path(ref).is_absolute() and str(tmp_path) not in ref
        stored = json.loads((ctx.run_dir / ref).read_text())
        assert stored["timing_samples_ms"] == original["timing_samples_ms"]
        assert stored["artifact_type"] == "avfuse_timing_samples" and stored["repeats"] == 3
        assert stored["run_id"] == ctx.run_id and ref.endswith(stored["record_id"] + ".json")
        assert ref == f"timing_samples/{art.record_id_for(original)}.json"
    art.validate_benchmark_records(lines, run_dir=ctx.run_dir)  # refs resolve and match stats
    assert schema_errors(lines[0]) == []


def test_exactly_one_timing_sample_representation_is_required(tmp_path):
    ctx = make_ctx(tmp_path)
    inline = record(ctx)
    art.validate_benchmark_records([inline])

    ref = f"timing_samples/{art.record_id_for(inline)}.json"
    sample_artifact = art._timing_artifact(ctx.run_id, art.record_id_for(inline),
                                          inline["timing_samples_ms"])
    art.write_json_artifact(ctx.run_dir / ref, sample_artifact)
    external = copy.deepcopy(inline)
    external.pop("timing_samples_ms")
    external["timing_samples_ref"] = ref
    art.validate_benchmark_records([external], run_dir=ctx.run_dir)
    with pytest.raises(art.BenchmarkValidationError, match="run_dir is required"):
        art.validate_benchmark_records([external])

    both = copy.deepcopy(external)
    both["timing_samples_ms"] = list(inline["timing_samples_ms"])
    with pytest.raises(art.BenchmarkValidationError, match="exactly one"):
        art.validate_benchmark_records([both], run_dir=ctx.run_dir)

    neither = copy.deepcopy(inline)
    neither.pop("timing_samples_ms")
    with pytest.raises(art.BenchmarkValidationError):
        art.validate_benchmark_records([neither], run_dir=ctx.run_dir)

    missing = copy.deepcopy(external)
    missing["timing_samples_ref"] = "timing_samples/missing.json"
    with pytest.raises(art.BenchmarkValidationError, match="does not exist"):
        art.validate_benchmark_records([missing], run_dir=ctx.run_dir)


def test_sample_reference_validation(tmp_path):
    ctx = make_ctx(tmp_path, make_config(timing_samples_storage="separate_file"))
    ctx.write_benchmark_records([record(ctx)])
    line = json.loads((ctx.run_dir / "benchmark_records.jsonl").read_text())
    ref_path = ctx.run_dir / line["timing_samples_ref"]

    changed = json.loads(ref_path.read_text())
    changed["timing_samples_ms"] = [100.0, 200.0, 300.0]
    ref_path.write_text(json.dumps(changed))
    with pytest.raises(art.BenchmarkValidationError, match="mean_ms"):
        art.validate_benchmark_records([line], run_dir=ctx.run_dir)
    changed["timing_samples_ms"] = [1.0, 2.0]
    ref_path.write_text(json.dumps(changed))
    with pytest.raises(art.BenchmarkValidationError, match="timing samples but repeats"):
        art.validate_benchmark_records([line], run_dir=ctx.run_dir)
    ref_path.unlink()
    with pytest.raises(art.BenchmarkValidationError, match="does not exist"):
        art.validate_benchmark_records([line], run_dir=ctx.run_dir)
    for bad in ("/abs/samples.json", "../samples.json", "timing_samples/../../x.json",
                "C:\\samples.json"):
        broken = dict(line, timing_samples_ref=bad)
        with pytest.raises(art.UnsafePathError):
            art.validate_benchmark_records([broken])


def test_separate_storage_never_overwrites_existing_sample_file(tmp_path):
    ctx = make_ctx(tmp_path, make_config(timing_samples_storage="separate_file"))
    recs = four_records(ctx)
    victim = ctx.run_dir / "timing_samples" / f"{art.record_id_for(recs[2])}.json"
    victim.parent.mkdir()
    victim.write_text('{"keep": "me"}')
    with pytest.raises(art.ArtifactExistsError):
        ctx.write_benchmark_records(recs)
    assert victim.read_text() == '{"keep": "me"}'
    assert [p.name for p in victim.parent.iterdir()] == [victim.name]  # nothing else created
    assert not (ctx.run_dir / "benchmark_records.jsonl").exists()


def test_failure_after_samples_removes_them_and_writes_no_jsonl(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path, make_config(timing_samples_storage="separate_file"))
    real = art.atomic_write_bytes

    def fail_on_jsonl(dest, data, **kw):
        if str(dest).endswith(".jsonl"):
            raise OSError("simulated crash while writing records")
        return real(dest, data, **kw)

    monkeypatch.setattr(art, "atomic_write_bytes", fail_on_jsonl)
    with pytest.raises(OSError, match="simulated crash"):
        ctx.write_benchmark_records(four_records(ctx))
    leftover = [p for p in ctx.run_dir.rglob("*") if p.is_file() and "timing" in p.parts[-2:][0]]
    assert leftover == []
    assert not (ctx.run_dir / "benchmark_records.jsonl").exists()
    assert "benchmark_records.jsonl" not in ctx.artifacts  # nothing registered


def test_failed_overwrite_restores_previous_run_outputs(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path, make_config(timing_samples_storage="separate_file", overwrite=True))
    ctx.write_benchmark_records(four_records(ctx))
    before = all_files(ctx.run_dir)
    real = art.atomic_write_bytes
    new_records = [record(ctx, c, m=measurement(samples=(9.0, 8.0, 7.0)))
                   for c in art.PRECISION_CONFIGS]

    def fail_on_jsonl(dest, data, **kw):
        should_fail = (
            str(dest).endswith(".jsonl")
            and kw.get("overwrite") is True
            and b'"mean_ms":8.0' in data
        )
        if should_fail:
            raise OSError("crash")
        return real(dest, data, **kw)

    monkeypatch.setattr(art, "atomic_write_bytes", fail_on_jsonl)
    with pytest.raises(OSError, match="crash"):
        art.write_benchmark_records(new_records, ctx.run_dir, run_id=ctx.run_id,
                                    timing_samples_storage="separate_file", overwrite=True)
    assert all_files(ctx.run_dir) == before  # prior valid outputs restored byte-for-byte


def test_filename_and_path_safety(tmp_path):
    for bad in ("/abs/x.json", "../x.json", "a/../b", "a\\b", "", "a//b", "./a", "C:\\x",
                "a/./b", "x\x00y"):
        with pytest.raises(art.UnsafePathError):
            art.validate_relative_path(bad)
    assert art.validate_relative_path("timing_samples/a.json") == "timing_samples/a.json"
    ctx = make_ctx(tmp_path)
    with pytest.raises(art.UnsafePathError):
        art.write_benchmark_records([record(ctx)], ctx.run_dir, filename="../evil.jsonl")
    assert not (tmp_path / "runs" / "evil.jsonl").exists()


def test_symlinked_run_subdirectory_is_refused(tmp_path):
    ctx = make_ctx(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (ctx.run_dir / "timing_samples").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unsupported")
    with pytest.raises(art.UnsafePathError):
        art.write_benchmark_records(
            [record(ctx)], ctx.run_dir, run_id=ctx.run_id, timing_samples_storage="separate_file")
    assert list(outside.iterdir()) == []


# ================================================================ manifest and integrity
def finished_run(tmp_path, **kw):
    cfg = make_config(timing_samples_storage="separate_file")
    ctx = make_ctx(tmp_path, cfg, **kw)
    ctx.write_benchmark_records(four_records(ctx))
    profile = profile_pipeline_stages(
        {"metadata_indexing": lambda: None}, device="cpu", input_kind="synthetic")
    ctx.write_stage_profile(profile)
    ctx.finalize()
    return ctx


def test_manifest_contents_and_no_absolute_paths(tmp_path):
    ctx = finished_run(tmp_path)
    text = (ctx.run_dir / "run_manifest.json").read_text()
    manifest = json.loads(text)
    assert str(tmp_path) not in text and "\\" not in text
    paths = [e["path"] for e in manifest["artifacts"]]
    # config, environment, provenance, records, four samples and stage profile
    assert paths == sorted(paths) and len(paths) == 9
    for e in manifest["artifacts"]:
        assert set(e) == {"path", "artifact_type", "size_bytes", "sha256", "created_at_utc",
                          "required"}
        assert not Path(e["path"]).is_absolute() and ".." not in e["path"].split("/")
        assert e["size_bytes"] == (ctx.run_dir / e["path"]).stat().st_size
        assert e["sha256"] == prov.sha256_file(ctx.run_dir / e["path"])
        assert e["created_at_utc"].endswith("Z")
    by_path = {e["path"]: e for e in manifest["artifacts"]}
    assert by_path["stage_profile.json"]["required"] is False
    assert by_path["benchmark_records.jsonl"]["required"] is True
    assert by_path["benchmark_records.jsonl"]["artifact_type"] == "avfuse_benchmark_records"


def test_manifest_excludes_itself_and_anchor_holds_its_hash(tmp_path):
    ctx = finished_run(tmp_path)
    manifest = json.loads((ctx.run_dir / "run_manifest.json").read_text())
    listed = {e["path"] for e in manifest["artifacts"]}
    assert "run_manifest.json" not in listed and "integrity/artifact_hashes.json" not in listed
    anchor = json.loads((ctx.run_dir / "integrity/artifact_hashes.json").read_text())
    assert anchor["run_manifest"]["sha256"] == prov.sha256_file(ctx.run_dir / "run_manifest.json")
    assert set(anchor["artifacts"]) == listed
    assert "SHA-256 is stored in" in manifest["self_hash_policy"]
    for reserved in ("run_manifest.json", "integrity/artifact_hashes.json"):
        with pytest.raises(art.ArtifactError):
            ctx.register(reserved, "x")


def test_verification_passes_and_does_not_mutate(tmp_path):
    ctx = finished_run(tmp_path)
    stats_before = {p: (p.stat().st_size, p.stat().st_mtime_ns)
                    for p in ctx.run_dir.rglob("*") if p.is_file()}
    bytes_before = all_files(ctx.run_dir)
    report = art.verify_run_integrity(ctx.run_dir)
    assert report["ok"] is True and report["report_type"] == "avfuse_integrity_report"
    assert report["manifest_status"] == "valid" and report["anchor_status"] == "valid"
    assert report["counts"] == {"valid": 9, "missing": 0, "modified": 0, "unreadable": 0,
                                "invalid_entries": 0}
    assert report["missing"] == report["modified"] == report["reference_problems"] == []
    assert report["run_id"] == ctx.run_id
    json.dumps(report)  # machine-readable
    assert all_files(ctx.run_dir) == bytes_before
    assert {p: (p.stat().st_size, p.stat().st_mtime_ns)
            for p in ctx.run_dir.rglob("*") if p.is_file()} == stats_before
    assert ctx.verify()["ok"] is True


def test_missing_required_artifact_is_an_integrity_failure(tmp_path):
    ctx = finished_run(tmp_path)
    sample = next((ctx.run_dir / "timing_samples").iterdir())
    sample.unlink()
    report = art.verify_run_integrity(ctx.run_dir)
    rel = f"timing_samples/{sample.name}"
    assert report["ok"] is False
    assert report["missing"] == [{"path": rel, "required": True}]
    assert rel not in report["valid"] and report["modified"] == []
    assert report["reference_problems"] and "does not exist" in report["reference_problems"][0][
        "reason"]


def test_missing_optional_artifact_is_reported_but_not_fatal(tmp_path):
    ctx = finished_run(tmp_path)
    (ctx.run_dir / "stage_profile.json").unlink()
    report = art.verify_run_integrity(ctx.run_dir)
    assert report["missing"] == [{"path": "stage_profile.json", "required": False}]
    assert report["ok"] is True


def test_modified_artifact_is_detected_and_not_repaired(tmp_path):
    ctx = finished_run(tmp_path)
    target = ctx.run_dir / "resolved_config.json"
    original = target.read_bytes()
    target.write_bytes(original.replace(b"test-experiment", b"test-experimenT"))  # same size
    report = art.verify_run_integrity(ctx.run_dir)
    assert report["ok"] is False
    (entry,) = report["modified"]
    assert entry["path"] == "resolved_config.json" and entry["reasons"] == ["hash_mismatch"]
    assert entry["expected_sha256"] != entry["actual_sha256"]
    assert "resolved_config.json" not in report["valid"]
    assert target.read_bytes() != original  # corruption is reported, never auto-repaired
    assert target.read_bytes().count(b"experimenT") == 1


def test_size_change_is_reported_with_both_reasons(tmp_path):
    ctx = finished_run(tmp_path)
    target = ctx.run_dir / "provenance.json"
    target.write_bytes(target.read_bytes() + b"\n\n")
    (entry,) = art.verify_run_integrity(ctx.run_dir)["modified"]
    assert entry["reasons"] == ["size_mismatch", "hash_mismatch"]
    assert entry["actual_size_bytes"] == entry["expected_size_bytes"] + 2


def test_incorrect_size_alone_is_detected(tmp_path):
    ctx = finished_run(tmp_path)
    path = ctx.run_dir / "run_manifest.json"
    manifest = json.loads(path.read_text())
    for e in manifest["artifacts"]:
        if e["path"] == "provenance.json":
            e["size_bytes"] += 1
    path.write_text(json.dumps(manifest))
    report = art.verify_run_integrity(ctx.run_dir)
    (entry,) = report["modified"]
    assert entry["reasons"] == ["size_mismatch"]
    assert report["anchor_status"] == "mismatch" and report["ok"] is False


def test_path_traversal_and_absolute_manifest_entries_rejected(tmp_path):
    ctx = finished_run(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"secret")
    path = ctx.run_dir / "run_manifest.json"
    manifest = json.loads(path.read_text())
    digest = prov.sha256_file(outside)
    bad_paths = ["../../outside.txt", str(outside), "/etc/passwd", "C:\\Windows\\x",
                 "a/../../outside.txt", "sub\\file", "", None, 5]
    for bad in bad_paths:
        manifest["artifacts"].append({"path": bad, "artifact_type": "x", "size_bytes": 6,
                                      "sha256": digest, "created_at_utc": "x", "required": True})
    path.write_text(json.dumps(manifest))
    report = art.verify_run_integrity(ctx.run_dir)
    assert len(report["invalid_entries"]) == len(bad_paths)
    assert report["ok"] is False
    assert not any("outside" in v for v in report["valid"])
    assert outside.read_bytes() == b"secret"


def test_symlinked_manifest_entry_is_invalid(tmp_path):
    ctx = finished_run(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"x")
    link = ctx.run_dir / "sneaky.json"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unsupported")
    manifest_path = ctx.run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"].append({"path": "sneaky.json", "artifact_type": "x", "size_bytes": 1,
                                  "sha256": prov.sha256_file(outside), "created_at_utc": "x",
                                  "required": True})
    manifest_path.write_text(json.dumps(manifest))
    report = art.verify_run_integrity(ctx.run_dir)
    assert [e["path"] for e in report["invalid_entries"]] == ["sneaky.json"]


def test_manifest_cannot_list_itself_or_duplicates(tmp_path):
    ctx = finished_run(tmp_path)
    path = ctx.run_dir / "run_manifest.json"
    manifest = json.loads(path.read_text())
    for rel in ("run_manifest.json", "integrity/artifact_hashes.json", "provenance.json"):
        manifest["artifacts"].append(dict(manifest["artifacts"][0], path=rel))
    path.write_text(json.dumps(manifest))
    report = art.verify_run_integrity(ctx.run_dir)
    assert len(report["invalid_entries"]) == 3 and report["ok"] is False


def test_missing_or_corrupt_manifest_and_anchor_do_not_raise(tmp_path):
    ctx = finished_run(tmp_path)
    anchor = ctx.run_dir / "integrity/artifact_hashes.json"
    saved = anchor.read_bytes()
    anchor.unlink()
    report = art.verify_run_integrity(ctx.run_dir)
    assert report["anchor_status"] == "missing" and report["ok"] is False
    anchor.write_bytes(saved)
    assert art.verify_run_integrity(ctx.run_dir)["ok"] is True
    tampered = json.loads(saved)
    tampered["run_manifest"]["sha256"] = "0" * 64
    anchor.write_text(json.dumps(tampered))
    assert art.verify_run_integrity(ctx.run_dir)["anchor_status"] == "mismatch"

    manifest = ctx.run_dir / "run_manifest.json"
    manifest.write_text("{not json")
    report = art.verify_run_integrity(ctx.run_dir)
    assert report["manifest_status"] == "unreadable" and report["ok"] is False
    manifest.write_text("[]")
    assert art.verify_run_integrity(ctx.run_dir)["manifest_status"] == "not_an_object"
    manifest.write_text('{"artifact_type": "other"}')
    assert art.verify_run_integrity(ctx.run_dir)["manifest_status"] == "invalid_structure"
    manifest.unlink()
    assert art.verify_run_integrity(ctx.run_dir)["manifest_status"] == "missing"


def test_malformed_nested_integrity_anchors_return_structured_reports(tmp_path):
    ctx = finished_run(tmp_path)
    anchor_path = ctx.run_dir / "integrity/artifact_hashes.json"
    valid = json.loads(anchor_path.read_text(encoding="utf-8"))
    malformed_values = [
        None,
        "invalid",
        [],
        dict(valid, artifact_type="wrong_type"),
        dict(valid, hash_algorithm="sha512"),
        dict(valid, run_manifest=None),
        dict(valid, run_manifest="invalid"),
        dict(valid, run_manifest={"path": "run_manifest.json"}),
        dict(valid, artifacts=None),
        dict(valid, artifacts=[]),
        dict(valid, artifacts={"file.json": 7}),
    ]
    for malformed in malformed_values:
        anchor_path.write_text(json.dumps(malformed), encoding="utf-8")
        report = art.verify_run_integrity(ctx.run_dir)
        assert report["ok"] is False
        assert report["anchor_status"] == "invalid_structure"
        json.dumps(report)  # always machine-readable
    anchor_path.write_text(json.dumps(valid), encoding="utf-8")
    assert art.verify_run_integrity(ctx.run_dir)["ok"] is True


def test_anchor_manifest_digest_mismatch_is_reported_without_raising(tmp_path):
    ctx = finished_run(tmp_path)
    anchor_path = ctx.run_dir / "integrity/artifact_hashes.json"
    anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
    anchor["run_manifest"]["sha256"] = "0" * 64
    anchor_path.write_text(json.dumps(anchor), encoding="utf-8")
    report = art.verify_run_integrity(ctx.run_dir)
    assert report["anchor_status"] == "mismatch"
    assert report["ok"] is False


def test_unlisted_files_are_informational(tmp_path):
    ctx = finished_run(tmp_path)
    (ctx.run_dir / "stray_notes.txt").write_text("extra")
    report = art.verify_run_integrity(ctx.run_dir)
    assert report["unlisted_files"] == ["stray_notes.txt"] and report["ok"] is True


def test_finalize_is_once_only_and_blocks_later_writes(tmp_path):
    ctx = finished_run(tmp_path)
    with pytest.raises(art.RunFinalizedError):
        ctx.finalize()
    with pytest.raises(art.RunFinalizedError):
        ctx.write_json_artifact("late.json", {}, artifact_type="x")
    with pytest.raises(art.RunFinalizedError):
        ctx.write_benchmark_records([record(ctx)])


def test_finalize_refuses_when_a_registered_artifact_vanished(tmp_path):
    ctx = make_ctx(tmp_path)
    (ctx.run_dir / "provenance.json").unlink()
    with pytest.raises(art.ArtifactError, match="missing"):
        ctx.finalize()
    assert not (ctx.run_dir / "run_manifest.json").exists()  # no falsely successful manifest
    assert not (ctx.run_dir / "integrity").exists()
    assert ctx.finalized is False


def test_finalize_without_overwrite_refuses_existing_manifest(tmp_path):
    ctx = make_ctx(tmp_path)
    (ctx.run_dir / "run_manifest.json").write_text("{}")
    with pytest.raises(art.ArtifactExistsError):
        ctx.finalize()
    assert (ctx.run_dir / "run_manifest.json").read_text() == "{}"


def test_finalize_rolls_back_manifest_when_anchor_write_fails_without_overwrite(tmp_path):
    ctx = make_ctx(tmp_path, make_config(overwrite=False))
    anchor_path = ctx.run_dir / art.ANCHOR_NAME
    anchor_path.parent.mkdir(parents=True)
    anchor_path.write_bytes(b'{"preserve": "anchor"}')
    before = all_files(ctx.run_dir)

    # The manifest write succeeds; no-clobber on the existing anchor then fails.
    with pytest.raises(art.ArtifactExistsError, match="artifact_hashes"):
        ctx.finalize()
    assert all_files(ctx.run_dir) == before
    assert ctx.finalized is False


def test_finalize_restores_previous_manifest_and_anchor_when_overwrite_fails(
    tmp_path, monkeypatch
):
    ctx = make_ctx(tmp_path, make_config(overwrite=True))
    manifest_path = ctx.run_dir / art.MANIFEST_NAME
    anchor_path = ctx.run_dir / art.ANCHOR_NAME
    anchor_path.parent.mkdir(parents=True)
    manifest_path.write_bytes(b'{"previous": "manifest"}')
    anchor_path.write_bytes(b'{"previous": "anchor"}')
    before = all_files(ctx.run_dir)

    original_write = art.write_json_artifact

    def fail_after_anchor_write(path, value, **kwargs):
        result = original_write(path, value, **kwargs)
        if Path(path) == anchor_path:
            raise OSError("injected post-write anchor failure")
        return result

    monkeypatch.setattr(art, "write_json_artifact", fail_after_anchor_write)
    with pytest.raises(OSError, match="post-write"):
        ctx.finalize()
    assert all_files(ctx.run_dir) == before
    assert ctx.finalized is False

    monkeypatch.setattr(art, "write_json_artifact", original_write)
    ctx.finalize()
    assert ctx.finalized is True
    assert ctx.verify()["ok"] is True


# ======================================================================== stage-profile artifact
def real_profile():
    return profile_pipeline_stages(
        {"metadata_indexing": lambda: None, "combined_audio_video_decode": lambda: None},
        device="cpu", input_kind="synthetic",
        stage_settings={"metadata_indexing": StageSettings(warmup=0, repeats=2),
                        "combined_audio_video_decode": StageSettings(warmup=0, repeats=2)})


def test_stage_profile_artifact_marker_and_unchanged_content(tmp_path):
    ctx = make_ctx(tmp_path)
    profile = real_profile()
    snapshot = copy.deepcopy(profile)
    ctx.write_stage_profile(profile)
    stored = json.loads((ctx.run_dir / "stage_profile.json").read_text())
    assert stored["artifact_type"] == "avfuse_stage_profile" and stored["artifact_version"] == "1.0"
    assert stored["run_id"] == ctx.run_id
    assert stored["profile"] == snapshot == profile  # embedded unchanged, input unmutated
    assert ctx.artifacts["stage_profile.json"]["artifact_type"] == "avfuse_stage_profile"
    assert ctx.artifacts["stage_profile.json"]["required"] is False


def test_stage_profile_semantics_preserved(tmp_path):
    ctx = make_ctx(tmp_path)
    ctx.write_stage_profile(real_profile())
    profile = json.loads((ctx.run_dir / "stage_profile.json").read_text())["profile"]
    stages = {s["stage"]: s for s in profile["stages"]}
    for name in ("video_decode", "audio_decode", "split_sampling"):
        assert stages[name]["status"] == "not_provided"  # unmeasured stays unmeasured
        assert not any(k in stages[name] for k in
                       ("p50_ms", "p95_ms", "timing_samples_ms", "mean_ms", "median_ms"))
    assert stages["video_decode"]["covered_only_by_composite"] == ["combined_audio_video_decode"]
    assert profile["composite_measurements"] == ["combined_audio_video_decode"]
    assert stages["combined_audio_video_decode"]["stage_kind"] == "composite"  # stays distinct
    assert stages["metadata_indexing"]["stage_kind"] == "individual"
    assert profile["end_to_end_measured"] is False and profile["complete"] is False


def test_stage_profile_has_no_invented_precision_or_end_to_end(tmp_path):
    ctx = make_ctx(tmp_path)
    ctx.write_stage_profile(real_profile())
    stored = json.loads((ctx.run_dir / "stage_profile.json").read_text())
    for key in ("config", "precision", "batch_size", "schema_version", "p50_ms", "mean_ms",
                "end_to_end_ms", "total_ms"):
        assert key not in stored and key not in stored["profile"]
    text = json.dumps(stored)
    assert "end_to_end_ms" not in text and "total_latency" not in text
    # it is not a benchmark record, and the benchmark validator says so
    with pytest.raises(art.BenchmarkValidationError):
        art.validate_benchmark_records([stored])


def test_stage_profile_builder_rejects_inconsistent_profiles():
    with pytest.raises(ValueError, match="profile_pipeline_stages"):
        art.build_stage_profile_artifact({"foo": 1}, run_id="r")
    bad = real_profile()
    bad["end_to_end_measured"] = True
    with pytest.raises(ValueError, match="end-to-end"):
        art.build_stage_profile_artifact(bad, run_id="r")
    fabricated = real_profile()
    next(s for s in fabricated["stages"] if s["status"] == "not_provided")["p50_ms"] = 0.0
    with pytest.raises(ValueError, match="not_provided but carries timing"):
        art.build_stage_profile_artifact(fabricated, run_id="r")


def test_stage_profile_is_covered_by_manifest_and_integrity(tmp_path):
    ctx = finished_run(tmp_path)
    profile_entry = next(e for e in json.loads((ctx.run_dir / "run_manifest.json").read_text())
                         ["artifacts"] if e["path"] == "stage_profile.json")
    assert profile_entry["artifact_type"] == "avfuse_stage_profile"
    (ctx.run_dir / "stage_profile.json").write_text("{}")
    assert art.verify_run_integrity(ctx.run_dir)["modified"][0]["path"] == "stage_profile.json"
