import numpy as np
import pytest
import torch

from avfuse.member3_runtime import benchmark as bm
from avfuse.member3_runtime.benchmark import (
    DeviceUnavailableError,
    MemoryMeasurementUnavailableError,
    benchmark_callable,
    compute_percentile,
    resolve_device,
    summarize_latencies,
)

CPU = torch.device("cpu")


def test_cpu_benchmark_has_unavailable_peak_memory():
    x=torch.tensor([2.0]); record=benchmark_callable(lambda: x*x, torch.device('cpu'), warmup=1, repeats=3)
    assert record['repeats']==3 and record['p95_ms']>=0
    assert record['peak_memory_bytes'] is None


# ------------------------------------------------------------------ helpers
def fake_clock(monkeypatch, durations_ms):
    """Make the engine's clock return start/stop pairs yielding ``durations_ms``."""
    ticks = []
    t = 1_000
    for d in durations_ms:
        ticks.extend([t, t + round(d * 1e6)])
        t += 10_000_000_000
    it = iter(ticks)
    monkeypatch.setattr(bm, "_now_ns", lambda: next(it))
    return it


@pytest.fixture
def log(monkeypatch):
    """Record CUDA/MPS calls in order on a CPU-only machine (all mocked)."""
    events = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda d=None: events.append(("cuda_sync", str(d))))
    monkeypatch.setattr(
        torch.cuda, "reset_peak_memory_stats", lambda d=None: events.append(("reset", str(d)))
    )
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda d=None: events.append(("alloc", str(d))) or 1024)
    monkeypatch.setattr(
        torch.cuda, "max_memory_allocated", lambda d=None: events.append(("peak_alloc", str(d))) or 4096
    )
    monkeypatch.setattr(
        torch.cuda, "max_memory_reserved", lambda d=None: events.append(("peak_res", str(d))) or 8192
    )
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "synchronize", lambda: events.append(("mps_sync",)))
    return events


# ------------------------------------------------------------ basic behavior
def test_cpu_smoke_returns_full_record():
    x = torch.tensor([2.0])
    r = benchmark_callable(lambda: x * x, "cpu", warmup=2, repeats=4)
    assert r["device"] == "cpu" and r["warmup"] == 2 and r["repeats"] == 4
    assert len(r["timing_samples_ms"]) == 4
    assert all(s >= 0 for s in r["timing_samples_ms"])
    assert r["p50_ms"] <= r["p95_ms"] <= r["max_ms"]
    assert r["min_ms"] <= r["p50_ms"]
    assert r["percentile_method"] == "linear"
    assert r["synchronization"] == "not_required_cpu"


def test_default_arguments_preserved():
    calls = []
    r = benchmark_callable(lambda: calls.append(1), CPU)
    assert len(calls) == 10 + 50 and r["warmup"] == 10 and r["repeats"] == 50


def test_warmup_excluded_from_samples(monkeypatch):
    calls = []
    clock = fake_clock(monkeypatch, [1.0, 2.0, 3.0])
    r = benchmark_callable(lambda: calls.append(1), CPU, warmup=5, repeats=3)
    assert len(calls) == 8
    assert r["warmup"] == 5
    assert r["timing_samples_ms"] == pytest.approx([1.0, 2.0, 3.0])
    assert next(clock, None) is None  # clock read exactly twice per timed repeat


def test_zero_warmup_allowed():
    calls = []
    r = benchmark_callable(lambda: calls.append(1), CPU, warmup=0, repeats=2)
    assert len(calls) == 2 and r["warmup"] == 0


@pytest.mark.parametrize("repeats", [1, 2, 7])
def test_repetition_count(repeats):
    r = benchmark_callable(lambda: None, CPU, warmup=0, repeats=repeats)
    assert r["repeats"] == repeats and len(r["timing_samples_ms"]) == repeats


def test_stats_use_recorded_samples(monkeypatch):
    fake_clock(monkeypatch, [5.0, 1.0, 3.0, 2.0, 4.0])
    r = benchmark_callable(lambda: None, CPU, warmup=0, repeats=5)
    assert r["timing_samples_ms"] == pytest.approx([5.0, 1.0, 3.0, 2.0, 4.0])  # unsorted, unmodified
    assert r["p50_ms"] == pytest.approx(3.0)
    assert r["p95_ms"] == pytest.approx(4.8)
    assert r["median_ms"] == pytest.approx(3.0)
    assert r["mean_ms"] == pytest.approx(3.0)
    assert (r["min_ms"], r["max_ms"]) == (pytest.approx(1.0), pytest.approx(5.0))


# --------------------------------------------------------------- percentiles
def test_p50_and_p95_linear_known_values():
    s = [1, 2, 3, 4, 5]
    assert compute_percentile(s, 50) == pytest.approx(3.0)
    assert compute_percentile(s, 95) == pytest.approx(4.8)
    assert compute_percentile(s, 0) == 1 and compute_percentile(s, 100) == 5


def test_percentile_methods_differ_as_documented():
    s = [5, 1, 4, 2, 3]  # order must not matter
    assert compute_percentile(s, 95, "linear") == pytest.approx(4.8)
    assert compute_percentile(s, 95, "lower") == 4  # sorted[floor(.95*4)=3]
    assert compute_percentile(s, 95, "nearest_rank") == 5  # sorted[ceil(.95*5)-1=4]
    assert compute_percentile(s, 50, "nearest_rank") == 3


def test_lower_method_has_no_float_rounding_error():
    s = list(range(21))  # 0.95 * 20 must be treated as exactly 19
    assert compute_percentile(s, 95, "lower") == 19


@pytest.mark.parametrize("n", [1, 2, 3, 10])
def test_small_sample_counts_do_not_crash(n):
    s = [float(i + 1) for i in range(n)]
    for method in bm.PERCENTILE_METHODS:
        p50, p95 = compute_percentile(s, 50, method), compute_percentile(s, 95, method)
        assert s[0] <= p50 <= p95 <= s[-1]


def test_single_sample_percentiles_equal_sample():
    assert compute_percentile([2.5], 50) == 2.5 and compute_percentile([2.5], 95) == 2.5


def test_two_sample_linear_interpolation():
    assert compute_percentile([10, 20], 50) == pytest.approx(15.0)
    assert compute_percentile([10, 20], 95) == pytest.approx(19.5)


def test_linear_matches_numpy():
    rng = np.random.default_rng(0)
    s = rng.random(37).tolist()
    for q in (5, 50, 95, 99):
        assert compute_percentile(s, q) == pytest.approx(float(np.percentile(s, q)))


def test_percentile_input_validation():
    with pytest.raises(ValueError):
        compute_percentile([], 50)
    with pytest.raises(ValueError):
        compute_percentile([1.0], 101)
    with pytest.raises(ValueError):
        compute_percentile([1.0], 50, "mystery")
    with pytest.raises(ValueError):
        compute_percentile([1.0, float("nan")], 50)
    with pytest.raises(ValueError):
        summarize_latencies([])


def test_benchmark_percentile_method_recorded(monkeypatch):
    fake_clock(monkeypatch, [1.0, 2.0, 3.0, 4.0, 5.0])
    r = benchmark_callable(lambda: None, CPU, warmup=0, repeats=5, percentile_method="lower")
    assert r["percentile_method"] == "lower" and r["p95_ms"] == pytest.approx(4.0)
    with pytest.raises(ValueError, match="percentile"):
        benchmark_callable(lambda: None, CPU, warmup=0, repeats=1, percentile_method="bogus")


# ---------------------------------------------------------------- validation
@pytest.mark.parametrize(
    ("bad", "exc"),
    [(-1, ValueError), (1.5, TypeError), (True, TypeError), ("3", TypeError), (None, TypeError)],
)
def test_invalid_warmup(bad, exc):
    called = []
    with pytest.raises(exc, match="warmup"):
        benchmark_callable(lambda: called.append(1), CPU, warmup=bad, repeats=1)
    assert not called


@pytest.mark.parametrize(
    ("bad", "exc"),
    [(0, ValueError), (-3, ValueError), (2.5, TypeError), (True, TypeError), (None, TypeError)],
)
def test_invalid_repeats(bad, exc):
    called = []
    with pytest.raises(exc, match="repeats"):
        benchmark_callable(lambda: called.append(1), CPU, warmup=0, repeats=bad)
    assert not called


def test_non_callable_rejected():
    with pytest.raises(TypeError, match="callable"):
        benchmark_callable(42, CPU, warmup=0, repeats=1)


@pytest.mark.parametrize("bad", ["tpu0", "cpu:3", "cuda:x", "meta"])
def test_invalid_device(bad):
    with pytest.raises(ValueError):
        benchmark_callable(lambda: None, bad, warmup=0, repeats=1)


@pytest.mark.parametrize("bad", [5, None])
def test_wrong_device_type(bad):
    with pytest.raises(TypeError, match="device"):
        benchmark_callable(lambda: None, bad, warmup=0, repeats=1)


def test_require_memory_conflicts_with_disabled_memory():
    with pytest.raises(ValueError, match="require_memory"):
        benchmark_callable(
            lambda: None, CPU, warmup=0, repeats=1, measure_memory=False, require_memory=True
        )


# ------------------------------------------------------- callable exceptions
def test_exception_in_timed_call_propagates_unchanged():
    err = RuntimeError("boom")
    n = []

    def fn():
        n.append(1)
        if len(n) == 3:
            raise err

    with pytest.raises(RuntimeError) as info:
        benchmark_callable(fn, CPU, warmup=1, repeats=5)
    assert info.value is err and len(n) == 3


def test_exception_in_warmup_propagates():
    def fn():
        raise KeyError("warm")

    with pytest.raises(KeyError, match="warm"):
        benchmark_callable(fn, CPU, warmup=2, repeats=1)


# ----------------------------------------------------------- CPU sync / memory
def test_cpu_never_touches_accelerator_sync(monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("accelerator API called on CPU")

    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    monkeypatch.setattr(torch.mps, "synchronize", forbidden)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", forbidden)
    r = benchmark_callable(lambda: None, CPU, warmup=1, repeats=2)
    assert r["synchronization"] == "not_required_cpu"


def test_cpu_memory_unavailable_not_zero():
    r = benchmark_callable(lambda: None, CPU, warmup=0, repeats=1)
    assert r["peak_memory_bytes"] is None and r["peak_memory_reserved_bytes"] is None
    assert r["memory_baseline_bytes"] is None
    assert r["memory_method"] == "not_measured"
    assert "CPU" in r["memory_note"]


def test_require_memory_on_cpu_raises():
    called = []
    with pytest.raises(MemoryMeasurementUnavailableError, match="cpu"):
        benchmark_callable(lambda: called.append(1), CPU, warmup=0, repeats=1, require_memory=True)
    assert not called


# --------------------------------------------------------------------- CUDA
def test_cuda_sync_order_and_device(log):
    r = benchmark_callable(lambda: log.append(("call",)), "cuda:1", warmup=2, repeats=2)
    s = ("cuda_sync", "cuda:1")
    assert log == [
        ("call",), ("call",),
        s,  # after warm-up
        ("alloc", "cuda:1"), ("reset", "cuda:1"),
        s, ("call",), s,  # repeat 1
        s, ("call",), s,  # repeat 2
        s,  # before reading peak
        ("peak_alloc", "cuda:1"), ("peak_res", "cuda:1"),
    ]
    assert r["synchronization"] == "cuda_synchronize" and r["device"] == "cuda:1"


def test_cuda_without_index_uses_current_device(log):
    r = benchmark_callable(lambda: None, torch.device("cuda"), warmup=0, repeats=1)
    assert r["device"] == "cuda:1"
    syncs = [e for e in log if e[0] == "cuda_sync"]
    assert syncs and all(e[1] == "cuda:1" for e in syncs)


def test_cuda_uses_requested_non_zero_device_everywhere(log):
    benchmark_callable(lambda: None, "cuda:0", warmup=1, repeats=2)
    devices = {e[1] for e in log if len(e) > 1}
    assert devices == {"cuda:0"}


def test_cuda_peak_memory_measured(log):
    r = benchmark_callable(lambda: None, "cuda:0", warmup=0, repeats=1)
    assert r["peak_memory_bytes"] == 4096
    assert r["peak_memory_reserved_bytes"] == 8192
    assert r["memory_baseline_bytes"] == 1024
    assert r["memory_method"] == "cuda_max_memory_allocated"
    assert r["memory_note"] is None
    assert log.index(("reset", "cuda:0")) < log.index(("peak_alloc", "cuda:0"))


def test_cuda_memory_disabled_by_caller(log):
    r = benchmark_callable(lambda: None, "cuda:0", warmup=0, repeats=1, measure_memory=False)
    assert r["peak_memory_bytes"] is None and r["memory_method"] == "not_measured"
    assert not [e for e in log if e[0] in ("reset", "peak_alloc", "peak_res")]


def test_cuda_synchronize_disabled_labelled_and_skips_per_call_sync(log):
    r = benchmark_callable(
        lambda: log.append(("call",)), "cuda:0", warmup=0, repeats=3, synchronize=False
    )
    assert r["synchronization"] == "disabled"
    between = log[log.index(("reset", "cuda:0")) + 1 : log.index(("peak_alloc", "cuda:0"))]
    # only the single pre-read synchronization remains besides the three calls
    assert between == [("call",)] * 3 + [("cuda_sync", "cuda:0")]


def test_cuda_callable_exception_propagates(log):
    def fn():
        raise ValueError("kernel failed")

    with pytest.raises(ValueError, match="kernel failed"):
        benchmark_callable(fn, "cuda:0", warmup=0, repeats=1)


# ---------------------------------------------------------------------- MPS
def test_mps_sync_around_every_call(log):
    r = benchmark_callable(lambda: log.append(("call",)), "mps", warmup=1, repeats=2)
    s = ("mps_sync",)
    assert log == [("call",), s, s, ("call",), s, s, ("call",), s, s]
    assert r["synchronization"] == "mps_synchronize" and r["device"] == "mps"
    assert not [e for e in log if e[0].startswith("cuda")]


def test_mps_memory_marked_unavailable(log):
    r = benchmark_callable(lambda: None, "mps", warmup=0, repeats=1)
    assert r["peak_memory_bytes"] is None and r["peak_memory_reserved_bytes"] is None
    assert r["memory_method"] == "not_measured"
    assert "MPS" in r["memory_note"] and "peak" in r["memory_note"]


def test_mps_require_memory_raises(log):
    with pytest.raises(MemoryMeasurementUnavailableError, match="mps"):
        benchmark_callable(lambda: None, "mps", warmup=0, repeats=1, require_memory=True)


# --------------------------------------------------- unavailable accelerators
def test_cuda_unavailable_raises_without_calling_fn(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    called = []
    with pytest.raises(DeviceUnavailableError, match="never falls back to CPU"):
        benchmark_callable(lambda: called.append(1), "cuda", warmup=1, repeats=1)
    assert not called


def test_cuda_index_out_of_range(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(DeviceUnavailableError, match=r"index 3.*device_count\(\) == 1"):
        resolve_device("cuda:3")


def test_mps_unavailable_raises_without_calling_fn(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    called = []
    with pytest.raises(DeviceUnavailableError, match="mps"):
        benchmark_callable(lambda: called.append(1), torch.device("mps"), warmup=1, repeats=1)
    assert not called


def test_unavailable_error_is_runtime_error():
    assert issubclass(DeviceUnavailableError, RuntimeError)
    assert issubclass(MemoryMeasurementUnavailableError, RuntimeError)


def test_resolve_device_normalizes_cpu():
    assert resolve_device("cpu") == CPU and resolve_device(torch.device("cpu:0")) == CPU


def test_sync_device_cpu_noop_and_unsupported():
    bm.sync_device(CPU)
    with pytest.raises(ValueError):
        bm.sync_device(torch.device("meta"))


# ------------------------------------------------- conversion-only callables
def test_conversion_only_callable_can_be_benchmarked():
    x = torch.linspace(-1, 1, 16)
    scale = x.abs().max() / 127

    def fake_quant_roundtrip():  # conversion work only; no attention
        return torch.clamp(torch.round(x / scale), -127, 127) * scale

    r = benchmark_callable(fake_quant_roundtrip, CPU, warmup=1, repeats=3)
    assert len(r["timing_samples_ms"]) == 3 and r["p50_ms"] >= 0


def test_return_value_of_callable_is_discarded_and_not_required():
    r = benchmark_callable(lambda: {"anything": object()}, CPU, warmup=0, repeats=1)
    assert r["repeats"] == 1


# ------------------------------------------------------- legacy compatibility
def test_legacy_fields_and_positional_call_compatible():
    x = torch.tensor([2.0])
    r = benchmark_callable(lambda: x * x, torch.device("cpu"), 1, 3)
    for key in ("repeats", "p50_ms", "p95_ms", "peak_memory_bytes"):
        assert key in r
    assert isinstance(r["p50_ms"], float) and isinstance(r["p95_ms"], float)
    assert isinstance(r["repeats"], int)


def test_result_is_consistent_with_schema_contract_rules():
    r = benchmark_callable(lambda: None, CPU, warmup=1, repeats=5)
    assert len(r["timing_samples_ms"]) == r["repeats"]
    assert r["p50_ms"] <= r["p95_ms"]
    assert r["percentile_method"] in bm.PERCENTILE_METHODS
    assert r["synchronization"] in {
        "cuda_synchronize", "mps_synchronize", "not_required_cpu", "disabled"
    }
    assert (r["peak_memory_bytes"] is None) == (r["memory_method"] == "not_measured")


def test_inference_mode_toggle():
    seen = []
    benchmark_callable(lambda: seen.append(torch.is_inference_mode_enabled()), CPU, 0, 1)
    benchmark_callable(
        lambda: seen.append(torch.is_inference_mode_enabled()), CPU, 0, 1, inference_mode=False
    )
    assert seen == [True, False]
