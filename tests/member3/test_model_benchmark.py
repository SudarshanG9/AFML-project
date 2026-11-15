"""Segment E model benchmark + run integration. Mock models, synthetic data, CPU only.

Nothing here is a measurement of a real detector.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import torch

from avfuse.member3_runtime import artifacts as art
from avfuse.member3_runtime import experiment as ex
from avfuse.member3_runtime import model_benchmark as mb
from avfuse.member3_runtime.benchmark import DeviceUnavailableError
from avfuse.member3_runtime.feature_loader import ClipRef, make_feature_provider
from avfuse.member3_runtime.model_benchmark import ConversionBenchmark, benchmark_precision_modes
from avfuse.shared.contracts import PRECISION_MODES

REPO = Path(__file__).resolve().parents[2]
SHA = "cd" * 32
NOW = datetime(2026, 5, 1, 12, 30, 45, 123000, tzinfo=timezone.utc)


def fake_report():
    return {
        "report_type": "avfuse_environment_report", "report_version": "1.0",
        "generated_at_utc": "2026-05-01T12:30:45.000000Z", "python": {"version": "3.12.0"},
        "os": {"platform": "Test-OS-1.0"},
        "cpu": {"description": "Test CPU", "logical_cores": 4, "physical_cores": 2},
        "git": {"commit": None, "dirty": None},
        "torch": {"available": True, "version": "2.5.0", "num_threads": 4,
                  "cuda": {"available": False, "runtime_version": None, "device_count": 0,
                           "devices": []},
                  "mps": {"available": False, "host_architecture": "arm64"}},
        "errors": [],
    }


def make_ctx(tmp_path, checkpoint_path=None):
    cfg = json.loads((REPO / "configs/benchmark.example.json").read_text("utf-8"))
    cfg["experiment_id"] = "seg-e-test"
    kw = {"checkpoint_path": checkpoint_path} if checkpoint_path else {}
    return art.create_run_context(cfg, runs_root=tmp_path / "runs", repo_root=tmp_path,
                                  allow_unset_inputs=True, environment_report=fake_report(),
                                  now=NOW, **kw)


class Mock:
    checkpoint_sha256 = SHA
    device = torch.device("cpu")
    training = False

    def __init__(self, supported=PRECISION_MODES):
        self.supported = set(supported)
        self.inputs = []

    def predict_from_cached_features(self, features, mode):
        self.inputs.append((mode, features))
        if mode not in self.supported:
            raise NotImplementedError(f"{mode} pending")
        return torch.tensor([0.5])


FEATS = {"video_tokens": torch.randn(6, 768), "audio_tokens": torch.randn(4, 768)}
KW = {"warmup": 1, "repeats": 3, "measure_memory": False}


def test_four_modes_records_are_schema_valid_and_honest(tmp_path):
    ctx, m = make_ctx(tmp_path), Mock()
    res = benchmark_precision_modes(m, FEATS, ctx=ctx, **KW)
    assert res.status == "complete" and res.four_mode_complete
    assert [r["config"] for r in res.records] == list(PRECISION_MODES)
    art.validate_benchmark_records(res.records)
    for r in res.records:
        assert r["scope"] == "cached_feature_model"
        assert r["workload"]["input_boundary"] == "cached_encoder_tokens"
        assert r["workload"]["input_shapes"] == {"audio_tokens": [4, 768],
                                                 "video_tokens": [6, 768]}
        assert r["workload"]["checkpoint_sha256"] == SHA
        assert r["workload"]["input_identity_sha256"] == res.input_identity_sha256
        assert r["backend"] == "none" and r["batch_size"] == 1 and r["device"] == "cpu"
        assert r["precision"]["execution_dtype"] == "float32"
        assert r["warmup"] == 1 and r["repeats"] == 3 and len(r["timing_samples_ms"]) == 3
        assert r["peak_memory_bytes"] is None and r["memory_method"] == "not_measured"
        assert r["measurement"] == {"percentile_method": "linear",
                                    "synchronization": "not_required_cpu"}
        assert "raw-video" in r["notes"]
    assert (tmp_path / "runs" / ctx.run_id / "benchmark_records.jsonl").is_file()


def test_same_cached_sample_object_used_for_every_mode(tmp_path):
    m = Mock()
    benchmark_precision_modes(m, FEATS, ctx=make_ctx(tmp_path), **KW)
    assert all(f is FEATS for _, f in m.inputs)
    assert {mode for mode, _ in m.inputs} == set(PRECISION_MODES)


def test_segment_b_engine_is_used(tmp_path, monkeypatch):
    calls = []
    real = mb.benchmark_callable
    monkeypatch.setattr(mb, "benchmark_callable",
                        lambda fn, device, **k: calls.append((device, k)) or real(fn, device, **k))
    benchmark_precision_modes(Mock(), FEATS, ctx=make_ctx(tmp_path), modes=("reference",),
                              percentile_method="nearest_rank", warmup=2, repeats=4,
                              measure_memory=False)
    assert len(calls) == 1 and calls[0][0] == "cpu"
    assert calls[0][1]["warmup"] == 2 and calls[0][1]["repeats"] == 4
    assert calls[0][1]["percentile_method"] == "nearest_rank"


def test_unsupported_modes_produce_no_records_strict(tmp_path):
    ctx = make_ctx(tmp_path)
    res = benchmark_precision_modes(Mock(supported=("reference",)), FEATS, ctx=ctx, **KW)
    assert res.status == "blocked" and res.records == [] and res.written is None
    assert not (tmp_path / "runs" / ctx.run_id / "benchmark_records.jsonl").exists()
    assert set(res.sweep.unsupported_modes) == {"low_precision", "restore_value_output",
                                                "restore_query"}


def test_partial_benchmark_only_records_supported(tmp_path):
    res = benchmark_precision_modes(Mock(supported=("reference",)), FEATS,
                                    ctx=make_ctx(tmp_path), allow_partial=True, **KW)
    assert res.status == "partial" and not res.four_mode_complete
    assert [r["config"] for r in res.records] == ["reference"]


def test_reference_only_request_is_complete_for_requested_mode(tmp_path):
    res = benchmark_precision_modes(Mock(supported=("reference",)), FEATS,
                                    ctx=make_ctx(tmp_path), modes=("reference",), **KW)
    r = res.records[0]
    assert res.status == "complete" and not res.four_mode_complete
    assert r["conversion_overhead_ms"] is None and r["precision"]["mode"] == "native_float"


def test_conversion_overhead_null_without_callable_and_disclosed(tmp_path):
    res = benchmark_precision_modes(Mock(), FEATS, ctx=make_ctx(tmp_path), **KW)
    for r in res.records:
        assert r["conversion_overhead_ms"] is None
        assert r["conversion_overhead_included_in_latency"] is None
    lp = next(r for r in res.records if r["config"] == "low_precision")
    assert "included in the total model latency" in lp["notes"]
    ref = next(r for r in res.records if r["config"] == "reference")
    assert "Conversion overhead" not in ref["notes"]


def test_dedicated_conversion_callable_is_benchmarked_separately(tmp_path):
    n = []
    conv = ConversionBenchmark(lambda: n.append(1), "fake-quantize Q/K/V only",
                               included_in_latency=True)
    res = benchmark_precision_modes(Mock(), FEATS, ctx=make_ctx(tmp_path),
                                    conversion_benchmarks={"low_precision": conv}, **KW)
    lp = next(r for r in res.records if r["config"] == "low_precision")
    assert lp["conversion_overhead_ms"] >= 0 and lp["conversion_overhead_included_in_latency"]
    assert "fake-quantize Q/K/V only" in lp["notes"]
    assert len(n) == 1 + 3  # warmup + repeats
    others = [r for r in res.records if r["config"] != "low_precision"]
    assert all(r["conversion_overhead_ms"] is None for r in others)
    with pytest.raises(ValueError, match="reference"):
        benchmark_precision_modes(Mock(), FEATS, ctx=make_ctx(tmp_path / "b"),
                                  conversion_benchmarks={"reference": conv}, **KW)


def test_unavailable_device_is_not_silently_cpu(tmp_path):
    if torch.cuda.is_available():
        pytest.skip("CUDA present")
    m = Mock()
    m.device = torch.device("cuda")
    with pytest.raises(DeviceUnavailableError):
        benchmark_precision_modes(m, FEATS, ctx=make_ctx(tmp_path), device="cuda",
                                  modes=("reference",), **KW)


def test_model_device_mismatch_and_train_mode_rejected(tmp_path):
    m = Mock()
    m.device = torch.device("cuda")
    with pytest.raises(mb.ModelDeviceMismatchError):
        benchmark_precision_modes(m, FEATS, ctx=make_ctx(tmp_path), device="cpu", **KW)
    t = Mock()
    t.training = True
    with pytest.raises(RuntimeError, match="training mode"):
        benchmark_precision_modes(t, FEATS, ctx=make_ctx(tmp_path / "b"), **KW)


def test_checkpoint_hash_must_match_run_context(tmp_path):
    ckpt = tmp_path / "w.pth"
    ckpt.write_bytes(b"weights")
    ctx = make_ctx(tmp_path, checkpoint_path=ckpt)
    with pytest.raises(mb.CheckpointIdentityError):
        benchmark_precision_modes(Mock(), FEATS, ctx=ctx, modes=("reference",), **KW)
    good = Mock()
    good.checkpoint_sha256 = ctx.checkpoint["sha256"]
    res = benchmark_precision_modes(good, FEATS, ctx=ctx, modes=("reference",), **KW)
    assert res.records[0]["workload"]["checkpoint_sha256"] == ctx.checkpoint["sha256"]


def test_sample_identity_is_content_hash():
    a, b = mb.sample_identity(FEATS), mb.sample_identity(FEATS)
    assert a["sha256"] == b["sha256"] and len(a["sha256"]) == 64
    other = {"video_tokens": FEATS["video_tokens"] + 1, "audio_tokens": FEATS["audio_tokens"]}
    assert mb.sample_identity(other)["sha256"] != a["sha256"]


def test_run_context_receives_valid_records_and_verifies(tmp_path):
    ctx = make_ctx(tmp_path)
    benchmark_precision_modes(Mock(), FEATS, ctx=ctx, **KW)
    ctx.finalize()
    assert ctx.verify()


# ---------------------------------------------------------------- run integration
def seed_features(root, split, ids):
    for i, cid in enumerate(ids):
        d = root / split / cid
        d.mkdir(parents=True)
        np.savez(d / "features.npz", video_features=np.ones((4 + i, 768), np.float32),
                 audio_features=np.ones((3 + i, 768), np.float32))


@pytest.fixture
def exp_world(tmp_path):
    root = tmp_path / "feats"
    seed_features(root, "val", ["v1", "v2"])
    clips = [ClipRef("v1", "val"), ClipRef("v2", "val")]
    return root, clips, make_feature_provider("dimodif_npz", root)


def _execute(tmp_path, exp_world, model, **kw):
    _, clips, provider = exp_world
    ctx = make_ctx(tmp_path)
    kw.setdefault("data_origin", "synthetic_test")
    summary = ex.execute_experiment(model, ctx, clips, provider, checkpoint_sha256=SHA,
                                    warmup=1, repeats=2, measure_memory=False, **kw)
    return ctx, summary


def test_full_four_mode_run_finalizes_and_verifies(tmp_path, exp_world):
    ctx, s = _execute(tmp_path, exp_world, Mock())
    assert s["status"] == "complete" and s["four_mode_complete"] and s["finalized"]
    assert s["labels_accessed"] is False and s["data_origin"] == "synthetic_test"
    assert s["benchmark"]["scope"] == "cached_feature_model" and s["benchmark"]["backend"] == "none"
    manifest = json.loads((ctx.run_dir / "run_manifest.json").read_text())
    paths = {a["path"] for a in manifest["artifacts"]}
    assert {"predictions.jsonl", "execution_summary.json", "benchmark_records.jsonl"} <= paths
    assert ctx.verify()
    # no feature caches or checkpoints copied into the run
    assert not list(ctx.run_dir.rglob("*.npz")) and not list(ctx.run_dir.rglob("*.pth"))


def test_reference_only_run_completes_but_is_not_four_mode(tmp_path, exp_world):
    _, s = _execute(tmp_path, exp_world, Mock(supported=("reference",)), modes=("reference",))
    assert s["status"] == "complete" and s["finalized"] and not s["four_mode_complete"]
    assert s["benchmark"]["benchmarked_modes"] == ["reference"]


def test_blocked_four_mode_run_publishes_nothing_and_is_not_finalized(tmp_path, exp_world):
    ctx, s = _execute(tmp_path, exp_world, Mock(supported=("reference",)))
    assert s["status"] == "blocked" and not s["finalized"] and not s["four_mode_complete"]
    assert set(s["predictions"]["unsupported_modes"]) == {
        "low_precision", "restore_value_output", "restore_query"}
    assert not (ctx.run_dir / "predictions.jsonl").exists()
    assert not (ctx.run_dir / "benchmark_records.jsonl").exists()
    assert not (ctx.run_dir / "run_manifest.json").exists()
    assert (ctx.run_dir / "execution_summary.json").is_file()


def test_partial_run_is_not_finalized(tmp_path, exp_world):
    ctx, s = _execute(tmp_path, exp_world, Mock(supported=("reference",)), allow_partial=True)
    assert s["status"] == "partial" and not s["finalized"] and not s["four_mode_complete"]
    assert not (ctx.run_dir / "run_manifest.json").exists()


def test_runtime_failure_leaves_failure_summary_and_no_manifest(tmp_path, exp_world):
    class Boom(Mock):
        def predict_from_cached_features(self, features, mode):
            raise RuntimeError("device exploded")

    _, clips, provider = exp_world
    ctx = make_ctx(tmp_path)
    with pytest.raises(Exception, match="device exploded"):
        ex.execute_experiment(Boom(), ctx, clips, provider, checkpoint_sha256=SHA)
    fail = json.loads((ctx.run_dir / "failure_summary.json").read_text())
    assert fail["status"] == "failed" and fail["error"]["type"] == "ModeFailedError"
    assert not (ctx.run_dir / "predictions.jsonl").exists()
    assert not (ctx.run_dir / "run_manifest.json").exists()


def test_dimodif_reference_path_reports_missing_inputs(tmp_path):
    ctx = make_ctx(tmp_path)
    with pytest.raises(ex.MissingInputError, match="manifest"):
        ex.run_dimodif_reference(ctx, upstream_root=tmp_path, checkpoint_name="lavdf",
                                 manifest_path=tmp_path / "nope.jsonl", feature_root=tmp_path)
    (tmp_path / "m.jsonl").write_text('{"clip_id":"a","split":"val"}\n')
    with pytest.raises(ex.MissingInputError, match="Feature root"):
        ex.run_dimodif_reference(ctx, upstream_root=tmp_path, checkpoint_name="lavdf",
                                 manifest_path=tmp_path / "m.jsonl",
                                 feature_root=tmp_path / "nofeatures")


def test_dimodif_adapter_needs_explicit_checkpoint_and_existing_root(tmp_path):
    with pytest.raises(ValueError, match="explicitly"):
        ex.build_dimodif_adapter(tmp_path, None, "cpu")
    with pytest.raises(ex.MissingInputError):
        ex.build_dimodif_adapter(tmp_path / "missing", "lavdf", "cpu")


def test_dimodif_reference_path_runs_with_patched_adapter(tmp_path, monkeypatch):
    """Exercises the real-reference wiring with a stand-in adapter (not the real model)."""
    root = tmp_path / "feats"
    seed_features(root, "val", ["v1", "v2"])
    manifest = tmp_path / "manifest_whole.jsonl"
    manifest.write_text("".join(
        json.dumps({"clip_id": c, "split": s, "label": 1, "feature_path": f"x/{c}.pt"}) + "\n"
        for c, s in (("v1", "val"), ("v2", "val"), ("t1", "test"))))
    from avfuse.member2_data.readiness import DIMODIF_LAVDF_SHA256

    adapter = Mock(supported=("reference",))
    adapter.checkpoint_sha256 = DIMODIF_LAVDF_SHA256
    seen = {}

    def fake_build(upstream_root, checkpoint_name, device):
        seen.update(root=upstream_root, name=checkpoint_name, device=device)
        return adapter

    monkeypatch.setattr(ex, "build_dimodif_adapter", fake_build)
    ctx = make_ctx(tmp_path)
    s = ex.run_dimodif_reference(ctx, upstream_root="UP", checkpoint_name="lavdf",
                                 manifest_path=manifest, feature_root=root, warmup=1, repeats=2,
                                 measure_memory=False)
    assert seen == {"root": "UP", "name": "lavdf", "device": "cpu"}
    assert s["status"] == "complete" and s["finalized"] and s["requested_modes"] == ["reference"]
    assert s["predictions"]["n_rows"] == 2 and s["predictions"]["split"] == "val"
    assert s["features"]["format"] == "dimodif_npz"
    rows = [json.loads(line) for line in (ctx.run_dir / "predictions.jsonl").read_text().splitlines()]
    assert {r["clip_id"] for r in rows} == {"v1", "v2"} and all(r["config"] == "reference" for r in rows)
    assert all(r["checkpoint_sha256"] == DIMODIF_LAVDF_SHA256 for r in rows)


def test_cuda_device_index_mismatch_is_rejected(monkeypatch):
    # Avoid requiring actual CUDA hardware while checking that same-type devices
    # are not treated as interchangeable.
    monkeypatch.setattr(mb, "resolve_device", lambda d: torch.device(d))
    model = Mock()
    model.device = torch.device("cuda:1")
    with pytest.raises(mb.ModelDeviceMismatchError, match="cuda:1.*cuda:0"):
        mb._assert_device(model, "cuda:0")


def test_subset_run_is_not_finalized(tmp_path, exp_world):
    ctx, s = _execute(tmp_path, exp_world, Mock(), max_clips=1)
    assert s["status"] == "partial"
    assert s["dataset_complete"] is False
    assert s["predictions"]["full_split"] is False
    assert not s["finalized"]
    assert not (ctx.run_dir / "run_manifest.json").exists()


def test_test_split_experiment_does_not_benchmark_test_features(tmp_path, exp_world):
    root, _, _ = exp_world
    seed_features(root, "test", ["t1"])
    clips = [ClipRef("t1", "test")]
    provider = make_feature_provider("dimodif_npz", root)
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}")
    ctx = make_ctx(tmp_path)
    with pytest.raises(ValueError, match="validation sample"):
        ex.execute_experiment(
            Mock(), ctx, clips, provider, checkpoint_sha256=SHA, split="test",
            protocol_path=protocol, unseal=True, purpose="held-out predictions only",
            warmup=1, repeats=2, measure_memory=False)
    assert not (tmp_path / "test_access.jsonl").exists()
    assert not (ctx.run_dir / "predictions.jsonl").exists()


def test_blocked_benchmark_prevents_run_finalization(tmp_path, exp_world):
    class TurnsUnsupportedAfterPredictionPass(Mock):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def predict_from_cached_features(self, features, mode):
            self.calls += 1
            # 2 clips x 4 modes are the prediction pass; report low_precision
            # unsupported on the separate benchmark capability check.
            if self.calls > 8 and mode == "low_precision":
                raise NotImplementedError("mode unavailable during benchmark")
            return torch.tensor([0.5])

    ctx, s = _execute(tmp_path, exp_world, TurnsUnsupportedAfterPredictionPass())
    assert s["status"] == "blocked"
    assert s["benchmark"]["status"] == "blocked"
    assert not s["finalized"]
    assert not (ctx.run_dir / "run_manifest.json").exists()


def test_lavdf_reference_path_rejects_a_non_agreed_checkpoint_hash(tmp_path, monkeypatch):
    from avfuse.member3_runtime.predictions import CheckpointMismatchError

    root = tmp_path / "feats"
    root.mkdir()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"clip_id": "v1", "split": "val"}) + "\n")
    adapter = Mock(supported=("reference",))  # SHA deliberately differs from agreed LAV-DF hash
    monkeypatch.setattr(ex, "build_dimodif_adapter", lambda *a, **k: adapter)
    ctx = make_ctx(tmp_path)
    with pytest.raises(CheckpointMismatchError, match="mismatch"):
        ex.run_dimodif_reference(ctx, upstream_root="UP", checkpoint_name="lavdf",
                                 manifest_path=manifest, feature_root=root,
                                 warmup=1, repeats=2, measure_memory=False)
    assert adapter.inputs == []
    assert not (ctx.run_dir / "predictions.jsonl").exists()
