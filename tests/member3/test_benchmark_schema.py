"""Schema and configuration-template tests (pure JSON Schema; no GPU, no torch needed)."""
import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMAS = REPO_ROOT / "contracts" / "schemas"
SHA = "a" * 64


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def validator() -> Draft202012Validator:
    return Draft202012Validator(_load(SCHEMAS / "benchmark.schema.json"))


@pytest.fixture(scope="module")
def config_validator() -> Draft202012Validator:
    return Draft202012Validator(_load(SCHEMAS / "benchmark_config.schema.json"))


def make_record(**overrides) -> dict:
    """A valid synthetic CPU operator record. Numbers are placeholders for validation only."""
    record = {
        "schema_version": "0.2",
        "run_id": "run-test-0001",
        "timestamp_utc": "2026-01-01T00:00:00Z",
        "experiment_id": "schema-test",
        "config": "reference",
        "scope": "operator",
        "device": "cpu",
        "backend": "sdpa",
        "precision": {"mode": "native_float", "execution_dtype": "float32", "simulated": False,
                      "real_int8_kernel": False, "disclosure": "Executed natively in float32."},
        "batch_size": 1,
        "warmup": 2,
        "repeats": 3,
        "workload": {"input_boundary": "synthetic_tensors",
                     "input_shapes": {"q": [1, 2, 3, 4], "k": [1, 2, 5, 4]},
                     "input_duration_seconds": None, "checkpoint_sha256": None,
                     "input_identity_sha256": None},
        "mean_ms": 2.0, "median_ms": 2.0, "p50_ms": 2.0, "p95_ms": 3.0,
        "measurement": {"percentile_method": "linear", "synchronization": "not_required_cpu"},
        "timing_samples_ms": [1.0, 2.0, 3.0],
        "peak_memory_bytes": None,
        "memory_method": "not_measured",
        "conversion_overhead_ms": None,
        "environment": {"python_version": "3.12.0", "torch_version": "2.5.0",
                        "os_platform": "test-os",
                        "device_metadata": {"type": "cpu", "name": None, "properties": None},
                        "git_commit": None},
    }
    record.update(overrides)
    return record


def errors(validator, record) -> list:
    return list(validator.iter_errors(record))


# ------------------------------------------------------------------ schema well-formedness
def test_schemas_are_valid_draft_2020_12():
    for name in ("benchmark.schema.json", "benchmark_config.schema.json"):
        schema = _load(SCHEMAS / name)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        Draft202012Validator.check_schema(schema)


def test_agreed_identifiers_and_scopes_preserved():
    props = _load(SCHEMAS / "benchmark.schema.json")["properties"]
    assert props["config"]["enum"] == ["reference", "low_precision", "restore_value_output",
                                       "restore_query"]
    assert props["scope"]["enum"] == ["operator", "cached_feature_model", "preprocessing_once"]


# ------------------------------------------------------------------------ valid examples
def test_valid_cpu_operator_record_accepted(validator):
    assert errors(validator, make_record()) == []


def test_valid_cuda_simulated_int8_model_record_accepted(validator):
    record = make_record(
        config="low_precision", scope="cached_feature_model", device="cuda:0",
        precision={"mode": "simulated_int8", "execution_dtype": "float32", "simulated": True,
                   "real_int8_kernel": False,
                   "disclosure": "Symmetric INT8 fake quantization in float32; no INT8 kernel."},
        workload={"input_boundary": "cached_encoder_tokens", "input_shapes": None,
                  "input_duration_seconds": 4.0, "checkpoint_sha256": SHA,
                  "input_identity_sha256": SHA},
        measurement={"percentile_method": "lower", "synchronization": "cuda_synchronize"},
        peak_memory_bytes=123456, memory_method="cuda_max_memory_allocated",
        conversion_overhead_ms=0.25, conversion_overhead_included_in_latency=False,
        timing_samples_ms=None,
    )
    del record["timing_samples_ms"]
    record["timing_samples_ref"] = "artifacts/benchmarks/samples/run-test-0001.json"
    record["environment"] = {
        "python_version": "3.12.0", "torch_version": "2.5.0", "os_platform": "test-os",
        "device_metadata": {"type": "cuda", "name": "Test GPU", "properties": {"x": 1}},
        "git_commit": "0123456789abcdef", "git_dirty": False,
        "environment_report_ref": "artifacts/runs/environment_report.json"}
    assert errors(validator, record) == []


def test_preprocessing_once_without_attention_backend_accepted(validator):
    record = make_record(
        scope="preprocessing_once", backend="none",
        workload={"input_boundary": "raw_media", "input_shapes": None,
                  "input_duration_seconds": 2.5, "checkpoint_sha256": None,
                  "input_identity_sha256": SHA})
    assert errors(validator, record) == []


def test_extensions_and_notes_allowed(validator):
    assert errors(validator, make_record(extensions={"anything": [1, 2]}, notes="hi")) == []


# --------------------------------------------------------------------------- rejections
@pytest.mark.parametrize("bad", ["fp32", "Reference", "", "restore_values", None, 3])
def test_invalid_config_identifier_rejected(validator, bad):
    assert errors(validator, make_record(config=bad))


@pytest.mark.parametrize("bad", ["operator_only", "model", "preprocessing", ""])
def test_invalid_scope_rejected(validator, bad):
    assert errors(validator, make_record(scope=bad))


@pytest.mark.parametrize("field", ["mean_ms", "median_ms", "p50_ms", "p95_ms"])
def test_negative_summary_timing_rejected(validator, field):
    assert errors(validator, make_record(**{field: -0.001}))


def test_negative_timing_sample_rejected(validator):
    assert errors(validator, make_record(timing_samples_ms=[1.0, -2.0, 3.0]))


def test_negative_conversion_overhead_and_memory_rejected(validator):
    assert errors(validator, make_record(conversion_overhead_ms=-1))
    assert errors(validator, make_record(peak_memory_bytes=-1, memory_method="cuda_max_memory_allocated",
                                         device="cuda", measurement={
                                             "percentile_method": "linear",
                                             "synchronization": "cuda_synchronize"}))


def test_unknown_fields_rejected_at_every_level(validator):
    assert errors(validator, make_record(surprise=1))
    record = make_record()
    record["precision"]["extra"] = 1
    assert errors(validator, record)
    record = make_record()
    record["workload"]["extra"] = 1
    assert errors(validator, record)
    record = make_record()
    record["environment"]["extra"] = 1
    assert errors(validator, record)


@pytest.mark.parametrize("field", ["schema_version", "run_id", "timestamp_utc", "experiment_id",
                                   "warmup", "repeats", "workload", "environment", "precision",
                                   "memory_method", "peak_memory_bytes", "mean_ms", "median_ms"])
def test_missing_required_field_rejected(validator, field):
    record = make_record()
    del record[field]
    assert errors(validator, record)


def test_samples_or_reference_required(validator):
    record = make_record()
    del record["timing_samples_ms"]
    assert errors(validator, record)


def test_bad_hash_timestamp_device_and_counts_rejected(validator):
    record = make_record()
    record["workload"]["checkpoint_sha256"] = "XYZ"
    assert errors(validator, record)
    assert errors(validator, make_record(timestamp_utc="2026-01-01 00:00:00"))
    assert errors(validator, make_record(device="gpu0"))
    assert errors(validator, make_record(batch_size=0))
    assert errors(validator, make_record(repeats=0))
    assert errors(validator, make_record(warmup=-1))
    assert errors(validator, make_record(schema_version="0.1"))


def test_real_int8_kernel_claim_rejected(validator):
    record = make_record()
    record["precision"]["real_int8_kernel"] = True
    assert errors(validator, record)


def test_simulation_flag_must_match_mode(validator):
    record = make_record()
    record["precision"]["mode"] = "simulated_int8"  # but simulated stays False
    assert errors(validator, record)
    record["precision"]["mode"] = "native_float"
    record["precision"]["simulated"] = True
    assert errors(validator, record)


def test_scope_boundary_consistency(validator):
    raw = {"input_boundary": "raw_media", "input_shapes": None, "input_duration_seconds": None,
           "checkpoint_sha256": None, "input_identity_sha256": None}
    assert errors(validator, make_record(workload=raw))  # operator on raw media
    assert errors(validator, make_record(scope="preprocessing_once", backend="none"))  # not raw
    assert errors(validator, make_record(scope="preprocessing_once", workload=raw))  # has backend
    assert errors(validator, make_record(backend="none"))  # operator needs an attention backend


def test_device_consistency_rules(validator):
    assert errors(validator, make_record(device="cuda:0"))  # metadata/sync still say cpu
    assert errors(validator, make_record(
        measurement={"percentile_method": "linear", "synchronization": "cuda_synchronize"}))
    assert errors(validator, make_record(peak_memory_bytes=10,
                                         memory_method="cuda_max_memory_allocated"))  # CUDA mem on CPU


# ----------------------------------------------------------------------- nullable memory
def test_null_peak_memory_accepted_on_cpu(validator):
    record = make_record(peak_memory_bytes=None, memory_method="not_measured")
    assert errors(validator, record) == []


def test_null_peak_memory_requires_not_measured_method(validator):
    assert errors(validator, make_record(peak_memory_bytes=None,
                                         memory_method="process_peak_rss"))


def test_measured_peak_memory_cannot_claim_not_measured(validator):
    assert errors(validator, make_record(peak_memory_bytes=1024, memory_method="not_measured"))
    assert errors(validator, make_record(peak_memory_bytes=1024,
                                         memory_method="process_peak_rss")) == []


def test_zero_peak_memory_is_a_valid_integer(validator):
    assert errors(validator, make_record(peak_memory_bytes=0, memory_method="process_peak_rss")) == []


# ---------------------------------------------------------------------- config template
def test_example_config_validates(config_validator):
    config = _load(REPO_ROOT / "configs" / "benchmark.example.json")
    assert list(config_validator.iter_errors(config)) == []


def test_example_config_makes_no_hardware_or_result_claims():
    config = _load(REPO_ROOT / "configs" / "benchmark.example.json")
    assert config["device"] in {"cpu", "auto"}  # never assumes CUDA/MPS
    assert config["experiment_id"].startswith("UNSET_")
    assert config["checkpoint_path"] is None and config["cache_manifest_path"] is None
    assert config["operator_shapes_file"] is None
    text = json.dumps(config).lower()
    assert "p50" not in text and "result" not in text
    assert set(config["configs"]) == {"reference", "low_precision", "restore_value_output",
                                      "restore_query"}


@pytest.mark.parametrize("mutate", [
    lambda c: c.update(warmup=-1),
    lambda c: c.update(repeats=0),
    lambda c: c.update(batch_size=0),
    lambda c: c.update(device="gpu"),
    lambda c: c.update(unknown_option=True),
    lambda c: c.update(configs=["reference", "not_a_config"]),
    lambda c: c.update(measure_scopes=["preprocessing_once"]),
    lambda c: c.update(percentile_method="median"),
    lambda c: c["synchronization"].update(extra=1),
    lambda c: c.pop("output"),
])
def test_invalid_config_rejected(config_validator, mutate):
    config = copy.deepcopy(_load(REPO_ROOT / "configs" / "benchmark.example.json"))
    mutate(config)
    assert list(config_validator.iter_errors(config))
