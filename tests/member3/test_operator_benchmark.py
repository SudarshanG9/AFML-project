"""Tests for the Segment C operator microbenchmark. CPU only; CUDA/MPS behaviour is mocked.

All tensors here are tiny, deterministic and SYNTHETIC: no value produced by these tests is
a research finding.
"""
import pytest
import torch

from avfuse.member1_fusion.attention import fusion_attention
from avfuse.member3_runtime import operator_benchmark as ob
from avfuse.member3_runtime.benchmark import (
    DeviceUnavailableError,
    MemoryMeasurementUnavailableError,
)
from avfuse.member3_runtime.operator_benchmark import (
    DenseOracleLimitError,
    OperatorInputError,
    OperatorInputs,
    OperatorWorkload,
    benchmark_attention_operator,
    build_batch_head_sweep,
    make_synthetic_inputs,
    run_operator_sweep,
    score_matrix_estimate,
    synthetic_cases,
    validate_operator_inputs,
)
from avfuse.shared.contracts import PRECISION_MODES

CPU = torch.device("cpu")


def qkv(b=2, h=3, nq=5, nk=7, d=4, dv=None, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    dv = d if dv is None else dv
    return (
        torch.randn(b, h, nq, d, generator=g).to(dtype),
        torch.randn(b, h, nk, d, generator=g).to(dtype),
        torch.randn(b, h, nk, dv, generator=g).to(dtype),
    )


def bench(q, k, v, **kw):
    kw.setdefault("mode", "reference")
    kw.setdefault("backend", "sdpa")
    kw.setdefault("device", "cpu")
    kw.setdefault("input_kind", "synthetic")
    kw.setdefault("warmup", 0)
    kw.setdefault("repeats", 2)
    return benchmark_attention_operator(q, k, v, **kw)


@pytest.fixture
def fake_engine(monkeypatch):
    """Replace Segment B's engine inside the operator module; record how it is called."""
    calls = []

    def fake(fn, device, warmup=10, repeats=50, **kwargs):
        calls.append({"fn": fn, "device": device, "warmup": warmup, "repeats": repeats, **kwargs})
        return {
            "repeats": repeats,
            "warmup": warmup,
            "device": str(device),
            "p50_ms": 1.5,
            "p95_ms": 2.5,
            "timing_samples_ms": [1.5] * repeats,
            "percentile_method": kwargs.get("percentile_method"),
            "synchronization": "fake_sync",
            "peak_memory_bytes": 111,
            "peak_memory_reserved_bytes": 222,
            "memory_baseline_bytes": 33,
            "memory_method": "fake_memory",
            "memory_note": None,
        }

    monkeypatch.setattr(ob, "benchmark_callable", fake)
    return calls


# ------------------------------------------------------------ smoke / backends
def test_cpu_sdpa_smoke():
    q, k, v = qkv()
    r = bench(q, k, v)
    assert r["scope"] == "operator"
    assert r["backend"] == "sdpa" and r["api_backend"] == "sdpa"
    assert r["device"] == "cpu"
    assert len(r["timing_samples_ms"]) == 2
    assert 0 <= r["p50_ms"] <= r["p95_ms"]
    assert r["synchronization"] == "not_required_cpu"


def test_cpu_dense_oracle_smoke_tiny():
    q, k, v = qkv(b=1, h=2, nq=3, nk=4, d=4)
    r = bench(q, k, v, backend="dense_oracle")
    assert r["backend"] == "dense_oracle" and r["api_backend"] == "dense"
    assert len(r["timing_samples_ms"]) == 2
    assert "not a substitute for SDPA" in " ".join(r["notes"])


@pytest.mark.parametrize("mode", PRECISION_MODES)
def test_all_four_precision_modes(mode):
    q, k, v = qkv(b=1, h=2, nq=3, nk=4)
    r = bench(q, k, v, mode=mode)
    assert r["precision_mode"] == mode
    simulated = mode != "reference"
    assert r["precision_simulated_int8"] is simulated
    assert r["fake_quant_included_in_latency"] is simulated
    notes = " ".join(r["notes"])
    assert ("not INT8 kernel execution" in notes) is simulated


def test_backend_label_maps_to_member1_api(monkeypatch):
    seen = []

    def spy(q, k, v, mode="reference", backend="sdpa", attn_mask=None):
        seen.append(backend)
        return q

    monkeypatch.setattr(ob, "fusion_attention", spy)
    q, k, v = qkv(b=1, h=1, nq=2, nk=2)
    bench(q, k, v, backend="sdpa", repeats=1)
    bench(q, k, v, backend="dense_oracle", repeats=1)
    assert seen == ["sdpa", "dense"]
    assert ob.BACKEND_LABEL_TO_API == {"sdpa": "sdpa", "dense_oracle": "dense"}


@pytest.mark.parametrize("backend", ["custom_tiled", "dense", "flash", ""])
def test_unsupported_backend_labels_rejected(backend):
    q, k, v = qkv(b=1, h=1, nq=2, nk=2)
    with pytest.raises(ValueError, match="Unsupported operator backend"):
        bench(q, k, v, backend=backend)


def test_unknown_precision_mode_rejected():
    q, k, v = qkv(b=1, h=1, nq=2, nk=2)
    with pytest.raises(ValueError, match="Unknown precision mode"):
        bench(q, k, v, mode="int4")


# --------------------------------------------------------------------- sweep
def test_batch_head_sweep_order_and_fixed_dims():
    ws = build_batch_head_sweep(
        batch_sizes=[1, 2], head_counts=[2, 4], q_tokens=3, k_tokens=5, head_dim=4
    )
    assert [(w.batch_size, w.num_heads) for w in ws] == [(1, 2), (1, 4), (2, 2), (2, 4)]
    assert {(w.q_tokens, w.k_tokens, w.head_dim) for w in ws} == {(3, 5, 4)}


def test_sweep_requires_explicit_dimensions():
    with pytest.raises(TypeError):
        build_batch_head_sweep(batch_sizes=[1], head_counts=[1])  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        build_batch_head_sweep(batch_sizes=[], head_counts=[1], q_tokens=1, k_tokens=1, head_dim=1)
    with pytest.raises(ValueError):
        OperatorWorkload(batch_size=0, num_heads=1, q_tokens=1, k_tokens=1, head_dim=1)


def test_run_sweep_covers_cases_backends_modes_in_order():
    ws = build_batch_head_sweep(
        batch_sizes=[1, 2], head_counts=[1, 2], q_tokens=3, k_tokens=4, head_dim=4
    )
    cases = synthetic_cases(ws, device="cpu")
    modes = ("reference", "low_precision")
    results = run_operator_sweep(
        cases,
        input_kind="synthetic",
        device="cpu",
        modes=modes,
        backends=("sdpa", "dense_oracle"),
        warmup=0,
        repeats=1,
    )
    assert len(results) == 4 * 2 * 2
    expected = [
        (w.batch_size, w.num_heads, b, m)
        for w in ws
        for b in ("sdpa", "dense_oracle")
        for m in modes
    ]
    got = [(r["batch_size"], r["num_heads"], r["backend"], r["precision_mode"]) for r in results]
    assert got == expected
    assert all(r["input_kind"] == "synthetic" for r in results)
    assert results[0]["workload_name"] == ws[0].name


def test_sweep_validates_everything_before_timing(fake_engine):
    good = OperatorInputs(*qkv(b=1, h=1, nq=2, nk=2))
    bad = OperatorInputs(*qkv(b=1, h=1, nq=2, nk=2, d=4)[:2], torch.zeros(1, 1, 3, 4))
    with pytest.raises(OperatorInputError, match="K and V sequence lengths differ"):
        run_operator_sweep([good, bad], input_kind="synthetic", device="cpu")
    assert fake_engine == []  # nothing was timed


def test_synthetic_inputs_are_deterministic_and_on_device():
    w = OperatorWorkload(batch_size=2, num_heads=2, q_tokens=3, k_tokens=4, head_dim=5, v_dim=6)
    a = make_synthetic_inputs(w, device="cpu", seed=3)
    b = make_synthetic_inputs(w, device="cpu", seed=3)
    c = make_synthetic_inputs(w, device="cpu", seed=4)
    assert tuple(a.q.shape) == (2, 2, 3, 5) and tuple(a.v.shape) == (2, 2, 4, 6)
    assert torch.equal(a.q, b.q) and torch.equal(a.v, b.v)
    assert not torch.equal(a.q, c.q)
    assert a.q.device.type == "cpu"


# ---------------------------------------------------------------- validation
@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda q, k, v: (q[0], k, v), "q must be rank-4"),
        (lambda q, k, v: (q, k[..., 0], v), "k must be rank-4"),
        (lambda q, k, v: (q, k, v.unsqueeze(0)), "v must be rank-4"),
        (lambda q, k, v: (q, k[:1], v), "Batch sizes differ"),
        (lambda q, k, v: (q, k[:, :2], v[:, :2]), "Head counts differ"),
        (lambda q, k, v: (q[..., :3], k, v), "Q and K head dimensions differ"),
        (lambda q, k, v: (q, k, v[:, :, :3]), "K and V sequence lengths differ"),
        (lambda q, k, v: (q[:, :, :0], k, v), "empty dimension"),
    ],
)
def test_shape_validation(mutate, message):
    q, k, v = mutate(*qkv())
    with pytest.raises(OperatorInputError, match=message):
        validate_operator_inputs(q, k, v, None, CPU)


def test_value_dim_may_differ_from_head_dim():
    q, k, v = qkv(dv=6)
    validate_operator_inputs(q, k, v, None, CPU)
    r = bench(q, k, v)
    assert r["v_shape"][-1] == 6


def test_non_tensor_rejected():
    q, k, v = qkv()
    with pytest.raises(OperatorInputError, match="must be a torch.Tensor"):
        validate_operator_inputs(q.tolist(), k, v, None, CPU)  # type: ignore[arg-type]


@pytest.mark.parametrize("dtype", [torch.float64, torch.int32, torch.int64, torch.bool])
def test_unsupported_dtype_rejected(dtype):
    q, k, v = qkv()
    with pytest.raises(OperatorInputError, match="unsupported dtype"):
        validate_operator_inputs(q.to(dtype), k, v, None, CPU)


def test_mixed_dtypes_rejected():
    q, k, v = qkv()
    with pytest.raises(OperatorInputError, match="share one dtype"):
        validate_operator_inputs(q, k.to(torch.float16), v, None, CPU)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_supported_dtypes_accepted_and_recorded(dtype):
    q, k, v = qkv(dtype=dtype)
    validate_operator_inputs(q, k, v, None, CPU)
    assert score_matrix_estimate(q, k)["dtype"] == str(dtype)


def test_device_mismatch_with_meta_tensor():
    q, k, v = qkv()
    with pytest.raises(OperatorInputError, match="never moved implicitly"):
        validate_operator_inputs(q, k.to("meta"), v, None, CPU)


def test_device_mismatch_against_mocked_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    q, k, v = qkv()  # CPU tensors, CUDA requested: must NOT be silently moved
    with pytest.raises(OperatorInputError, match="benchmark device is cuda:0"):
        bench(q, k, v, device="cuda")
    assert q.device.type == "cpu"


def test_unavailable_accelerator_raises_without_cpu_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    q, k, v = qkv()
    with pytest.raises(DeviceUnavailableError):
        bench(q, k, v, device="cuda")


def test_mask_valid_shapes_accepted_and_used():
    q, k, v = qkv(b=2, h=3, nq=5, nk=7)
    for shape in [(5, 7), (1, 5, 7), (1, 1, 5, 7), (2, 1, 5, 7), (2, 3, 5, 7)]:
        mask = torch.ones(*shape, dtype=torch.bool)
        validate_operator_inputs(q, k, v, mask, CPU)
    mask = torch.ones(1, 1, 5, 7, dtype=torch.bool)
    mask[..., 3:] = False
    r = bench(q, k, v, attn_mask=mask)
    assert r["has_attn_mask"] is True and r["attn_mask_shape"] == [1, 1, 5, 7]
    assert r["attn_mask_dtype"] == "torch.bool"


def test_masked_callable_matches_member1_output():
    q, k, v = qkv()
    mask = torch.ones(1, 1, 5, 7, dtype=torch.bool)
    mask[..., 4:] = False
    run = ob.make_attention_callable(q, k, v, mode="low_precision", backend="dense_oracle",
                                     attn_mask=mask)
    expected = fusion_attention(q, k, v, mode="low_precision", backend="dense", attn_mask=mask)
    assert torch.equal(run(), expected)


@pytest.mark.parametrize(
    "shape",
    [(5, 8), (3, 7), (4, 1, 5, 7), (1, 1, 1, 5, 7), (7,), (2, 3, 5, 7, 1)],
)
def test_mask_incompatible_shapes_rejected(shape):
    q, k, v = qkv()
    with pytest.raises(OperatorInputError, match="attn_mask"):
        validate_operator_inputs(q, k, v, torch.ones(*shape, dtype=torch.bool), CPU)


def test_mask_dtype_and_device_rejected():
    q, k, v = qkv()
    with pytest.raises(OperatorInputError, match="dtype must be torch.bool"):
        validate_operator_inputs(q, k, v, torch.zeros(5, 7, dtype=torch.float16), CPU)
    with pytest.raises(OperatorInputError, match="attn_mask is on device"):
        validate_operator_inputs(q, k, v, torch.ones(5, 7, dtype=torch.bool, device="meta"), CPU)


def test_float_mask_matching_dtype_accepted():
    q, k, v = qkv()
    validate_operator_inputs(q, k, v, torch.zeros(5, 7, dtype=torch.float32), CPU)


# ---------------------------------------------------------- dense safety limit
def test_dense_limit_blocks_before_any_timing(fake_engine):
    q, k, v = qkv()
    estimate = score_matrix_estimate(q, k)["bytes"]
    with pytest.raises(DenseOracleLimitError, match="dense_oracle refused"):
        bench(q, k, v, backend="dense_oracle", dense_max_score_bytes=estimate - 1)
    assert fake_engine == []
    bench(q, k, v, backend="dense_oracle", dense_max_score_bytes=estimate)  # at limit: allowed
    assert len(fake_engine) == 1


def test_default_dense_limit_blocks_large_workload_without_allocating(fake_engine):
    # expand() creates zero-copy views, so this "12.8 GB" workload allocates nothing.
    q = torch.zeros(1, 1, 1, 4).expand(1, 8, 20000, 4)
    k = torch.zeros(1, 1, 1, 4).expand(1, 8, 20000, 4)
    v = torch.zeros(1, 1, 1, 4).expand(1, 8, 20000, 4)
    assert score_matrix_estimate(q, k)["bytes"] > ob.DEFAULT_DENSE_MAX_SCORE_BYTES
    with pytest.raises(DenseOracleLimitError):
        bench(q, k, v, backend="dense_oracle")
    assert fake_engine == []
    bench(q, k, v, backend="sdpa")  # the limit applies to the dense oracle only
    assert len(fake_engine) == 1


def test_dense_limit_value_validated():
    q, k, v = qkv()
    for bad in (0, -1, None, True, 1.5):
        with pytest.raises(ValueError, match="dense_max_score_bytes"):
            bench(q, k, v, dense_max_score_bytes=bad)  # type: ignore[arg-type]


def test_sweep_enforces_dense_limit_without_skipping(fake_engine):
    case = OperatorInputs(*qkv())
    with pytest.raises(DenseOracleLimitError):
        run_operator_sweep(
            [case], input_kind="synthetic", device="cpu",
            backends=("sdpa", "dense_oracle"), dense_max_score_bytes=8,
        )
    assert fake_engine == []


# ------------------------------------------------------------ score estimate
@pytest.mark.parametrize(
    "dtype, element_size", [(torch.float32, 4), (torch.float16, 2), (torch.bfloat16, 2)]
)
def test_score_matrix_estimate_values(dtype, element_size):
    q, k, v = qkv(b=2, h=3, nq=5, nk=7, dtype=dtype)
    e = score_matrix_estimate(q, k)
    assert e["bytes"] == 2 * 3 * 5 * 7 * element_size
    assert e["bytes_per_element"] == element_size
    assert e["mib"] == pytest.approx(e["bytes"] / 2**20)
    assert (e["batch_size"], e["num_heads"], e["q_tokens"], e["k_tokens"]) == (2, 3, 5, 7)
    assert e["is_estimate_only"] is True
    assert "NOT the memory allocated by fused SDPA" in e["note"]
    r = bench(q, k, v)
    assert r["score_matrix_bytes"] == e["bytes"]
    assert r["score_matrix_estimate"] == e
    assert "not" in r["score_matrix_note"].lower()


def test_score_estimate_uses_member1_function_and_detects_drift(monkeypatch):
    q, k, _v = qkv()
    monkeypatch.setattr(ob, "score_matrix_bytes", lambda q, k: 1)
    with pytest.raises(RuntimeError, match="disagrees"):
        score_matrix_estimate(q, k)


def test_estimate_independent_of_q_k_different_lengths():
    q, k, _v = qkv(b=1, h=1, nq=3, nk=11)
    assert score_matrix_estimate(q, k)["bytes"] == 1 * 1 * 3 * 11 * 4


# ------------------------------------------------- synthetic / real labelling
def test_synthetic_labelling():
    q, k, v = qkv()
    r = bench(q, k, v, input_kind="synthetic")
    assert r["input_kind"] == "synthetic"
    assert r["evidence_status"] == ob.EVIDENCE_SYNTHETIC
    assert any("not a research finding" in n for n in r["notes"])
    assert r["shape_source"] is None


def test_real_labelling_requires_shape_source():
    q, k, v = qkv()
    with pytest.raises(ValueError, match="shape_source"):
        bench(q, k, v, input_kind="real")
    with pytest.raises(ValueError, match="shape_source"):
        bench(q, k, v, input_kind="real", shape_source="   ")
    r = bench(q, k, v, input_kind="real", shape_source="member1 shape report (test placeholder)")
    assert r["input_kind"] == "real"
    assert r["evidence_status"] == ob.EVIDENCE_REAL_DECLARED
    assert r["shape_source"].startswith("member1")
    assert not any("not a research finding" in n for n in r["notes"])


def test_input_kind_is_mandatory_and_validated():
    q, k, v = qkv()
    with pytest.raises(TypeError):
        benchmark_attention_operator(  # type: ignore[call-arg]
            q, k, v, mode="reference", backend="sdpa", device="cpu"
        )
    with pytest.raises(ValueError, match="input_kind"):
        bench(q, k, v, input_kind="simulated")


# -------------------------------------------------- Segment B engine usage
def test_uses_segment_b_engine_and_merges_its_fields(fake_engine):
    q, k, v = qkv()
    r = bench(q, k, v, warmup=3, repeats=9, percentile_method="nearest_rank",
              measure_memory=False, require_memory=False)
    assert len(fake_engine) == 1
    call = fake_engine[0]
    assert call["device"] == CPU
    assert (call["warmup"], call["repeats"]) == (3, 9)
    assert call["percentile_method"] == "nearest_rank"
    assert call["measure_memory"] is False and call["require_memory"] is False
    # Segment B's fields appear verbatim in the result
    assert r["p50_ms"] == 1.5 and r["p95_ms"] == 2.5
    assert r["synchronization"] == "fake_sync"
    assert r["peak_memory_bytes"] == 111 and r["peak_memory_reserved_bytes"] == 222
    assert r["memory_baseline_bytes"] == 33 and r["memory_method"] == "fake_memory"
    assert r["percentile_method"] == "nearest_rank"
    assert (r["warmup"], r["repeats"]) == (3, 9)
    assert len(r["timing_samples_ms"]) == 9


def test_timed_callable_is_exactly_one_fusion_attention_call(monkeypatch, fake_engine):
    calls = []

    def spy(q, k, v, mode="reference", backend="sdpa", attn_mask=None):
        calls.append((q, k, v, mode, backend, attn_mask))
        return q

    monkeypatch.setattr(ob, "fusion_attention", spy)
    q, k, v = qkv()
    mask = torch.ones(5, 7, dtype=torch.bool)
    bench(q, k, v, mode="restore_query", backend="dense_oracle", attn_mask=mask)
    assert calls == []  # setup/validation never invoke the operator itself
    fake_engine[0]["fn"]()
    assert len(calls) == 1
    cq, ck, cv, cmode, cbackend, cmask = calls[0]
    assert cq is q and ck is k and cv is v and cmask is mask
    assert (cmode, cbackend) == ("restore_query", "dense")


def test_warmup_and_repeats_reach_the_real_engine(monkeypatch):
    q, k, v = qkv()
    count = {"n": 0}
    real = ob.fusion_attention

    def counting(*a, **kw):
        count["n"] += 1
        return real(*a, **kw)

    monkeypatch.setattr(ob, "fusion_attention", counting)
    r = bench(q, k, v, warmup=2, repeats=3)
    assert count["n"] == 5
    assert (r["warmup"], r["repeats"]) == (2, 3)
    assert len(r["timing_samples_ms"]) == 3


def test_cpu_memory_reported_unavailable_not_zero():
    q, k, v = qkv()
    r = bench(q, k, v)
    assert r["peak_memory_bytes"] is None
    assert r["memory_method"] == "not_measured"
    assert r["memory_note"]
    with pytest.raises(MemoryMeasurementUnavailableError):
        bench(q, k, v, require_memory=True)


# ------------------------------------------------- nothing created/moved in timing
def test_no_tensor_creation_or_transfer_inside_timed_callable(monkeypatch):
    q, k, v = qkv()
    mask = torch.ones(1, 1, 5, 7, dtype=torch.bool)
    runs = [
        ob.make_attention_callable(q, k, v, mode=m, backend=b, attn_mask=mask)
        for m in PRECISION_MODES
        for b in ("sdpa", "dense_oracle")
    ]
    forbidden = []

    def trap(name):
        def inner(*a, **kw):
            forbidden.append(name)
            raise AssertionError(f"{name} called inside the timed callable")
        return inner

    for fn_name in ("randn", "rand", "zeros", "ones", "empty", "tensor", "as_tensor", "full"):
        monkeypatch.setattr(torch, fn_name, trap(f"torch.{fn_name}"))
    for method in ("to", "cpu", "cuda", "clone", "contiguous"):
        monkeypatch.setattr(torch.Tensor, method, trap(f"Tensor.{method}"))
    for run in runs:
        run()
    assert forbidden == []


def test_benchmark_does_not_move_or_modify_inputs():
    q, k, v = qkv()
    q0, k0, v0 = q.clone(), k.clone(), v.clone()
    bench(q, k, v, mode="low_precision")
    assert torch.equal(q, q0) and torch.equal(k, k0) and torch.equal(v, v0)
    assert q.device.type == "cpu"


def test_result_discloses_scope_and_nothing_is_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    q, k, v = qkv()
    r = bench(q, k, v, mode="low_precision")
    notes = " ".join(r["notes"])
    assert "Operator-only latency" in notes and "Not cached-feature model latency" in notes
    assert "included in the measured latency" in notes
    assert list(tmp_path.iterdir()) == []  # Segment C writes no artifacts


# ------------------------------------------- Nq != Nk, INT8 warning, real declaration
@pytest.mark.parametrize("backend", ["sdpa", "dense_oracle"])
def test_different_query_and_key_token_counts(backend):
    q, k, v = qkv(b=1, h=2, nq=3, nk=11, d=4, dv=6)
    r = bench(q, k, v, backend=backend)
    assert r["q_shape"] == [1, 2, 3, 4]
    assert r["k_shape"] == [1, 2, 11, 4]
    assert r["v_shape"] == [1, 2, 11, 6]
    assert r["score_matrix_estimate"]["q_tokens"] == 3
    assert r["score_matrix_estimate"]["k_tokens"] == 11
    assert r["score_matrix_bytes"] == 1 * 2 * 3 * 11 * 4


def test_nq_nk_mismatch_is_not_treated_as_an_error():
    q, k, v = qkv(nq=2, nk=9)
    validate_operator_inputs(q, k, v, None, CPU)  # must not raise


@pytest.mark.parametrize("mode", ["low_precision", "restore_value_output", "restore_query"])
def test_simulated_int8_warning_in_result_metadata(mode):
    q, k, v = qkv(b=1, h=1, nq=2, nk=3)
    r = bench(q, k, v, mode=mode)
    warning = r["precision_note"]
    assert "NOT an isolated conversion-overhead" in warning
    assert "floating-point" in warning
    assert "not real INT8 storage" in warning
    assert "not INT8 kernel execution" in warning
    assert "not evidence of INT8 hardware speedup" in warning
    assert warning in r["notes"]
    assert r["precision_simulated_int8"] is True and r["fake_quant_included_in_latency"] is True


def test_reference_mode_makes_no_fake_quant_claim():
    q, k, v = qkv(b=1, h=1, nq=2, nk=3)
    r = bench(q, k, v, mode="reference")
    assert r["precision_simulated_int8"] is False
    assert r["fake_quant_included_in_latency"] is False
    assert "no fake quantization" in r["precision_note"]
    assert not any("conversion-overhead" in n for n in r["notes"])


def test_real_declaration_is_documented_as_caller_declared():
    q, k, v = qkv(b=1, h=1, nq=2, nk=3)
    r = bench(q, k, v, input_kind="real", shape_source="placeholder source for test")
    assert any("declared by the caller" in n and "not verified" in n for n in r["notes"])
    assert "caller_declared" in r["evidence_status"]


def test_module_documents_member1_api_and_estimate_nature():
    doc = " ".join(ob.__doc__.split())
    assert "fusion_attention(q, k, v, mode=mode, backend=api_backend, attn_mask=attn_mask)" in doc
    assert "NOT a measurement of what fused SDPA allocates" in doc
    assert "INT8 kernel timings" in doc and "not** INT8" in doc


def test_readme_documents_backend_mapping_and_estimate():
    from pathlib import Path

    readme = (Path(ob.__file__).parent / "README.md").read_text()
    for label, api in ob.BACKEND_LABEL_TO_API.items():
        assert f"`{label}`" in readme and f'`backend="{api}"`' in readme
    for mode in PRECISION_MODES:
        assert f"`{mode}`" in readme
    assert "not** the memory allocated by fused SDPA" in readme
