"""Tests for the generic stage profiler.

Every callable here is a trivial SYNTHETIC counter or a mock. No Member 2 code, LAV-DF data,
media, checkpoint or cached feature is used, and no timing here is evidence of real
pipeline performance.
"""
import re
from pathlib import Path

import pytest
import torch

from avfuse.member3_runtime import stage_profiler as sp
from avfuse.member3_runtime.stage_profiler import (
    COMPOSITE_STAGES,
    INDIVIDUAL_STAGES,
    LATENCY_KEYS,
    OPTIONAL_STAGES,
    REQUIRED_STAGES,
    STAGE_ORDER,
    ExternalMeasurement,
    StageExecutionError,
    StageInput,
    StageSettings,
    profile_pipeline_stages,
)

SPEC_STAGES = [
    "metadata_indexing", "split_sampling", "selection_list_generation", "dataset_transfer",
    "archive_extraction", "video_decode", "audio_decode", "raw_array_serialization",
    "clips_csv_serialization", "clip_validation", "manifest_row_construction", "split_audit",
    "manifest_serialization", "feature_encoding", "fusion", "head",
]
PRE_REQUIRED = [s for s in REQUIRED_STAGES if INDIVIDUAL_STAGES[s] == "preprocessing_once"]


class Recorder:
    """Counting callables that also append to a shared ordered log."""

    def __init__(self):
        self.log = []
        self.counts = {}

    def make(self, name):
        self.counts[name] = 0

        def fn():
            self.counts[name] += 1
            self.log.append(name)

        return fn

    def stages(self, names=REQUIRED_STAGES):
        return {n: self.make(n) for n in names}


def run(stages, **kw):
    kw.setdefault("device", "cpu")
    kw.setdefault("repeatable_settings", StageSettings(warmup=0, repeats=2))
    return profile_pipeline_stages(stages, **kw)


def by_stage(profile):
    return {r["stage"]: r for r in profile["stages"]}


@pytest.fixture
def fake_engine(monkeypatch):
    calls = []

    def fake(fn, device, warmup=10, repeats=50, **kwargs):
        calls.append({"fn": fn, "device": device, "warmup": warmup, "repeats": repeats, **kwargs})
        return {
            "device": str(device), "warmup": warmup, "repeats": repeats,
            "p50_ms": 1.0, "p95_ms": 2.0, "timing_samples_ms": [1.0] * repeats,
        }

    monkeypatch.setattr(sp, "benchmark_callable", fake)
    return calls


# ------------------------------------------------------------- stage definitions
def test_all_sixteen_spec_stages_are_defined_with_exact_names():
    assert list(INDIVIDUAL_STAGES) == SPEC_STAGES
    assert set(OPTIONAL_STAGES) == {"dataset_transfer", "archive_extraction"}
    assert len(REQUIRED_STAGES) == 14
    assert set(STAGE_ORDER) == set(INDIVIDUAL_STAGES) | set(COMPOSITE_STAGES)
    assert len(set(STAGE_ORDER)) == len(STAGE_ORDER)  # no duplicates
    assert set(sp.STAGE_DESCRIPTIONS) == set(STAGE_ORDER)


def test_scope_classification():
    for stage in SPEC_STAGES:
        expected = "cached_feature_model" if stage in ("fusion", "head") else "preprocessing_once"
        assert sp.STAGE_SCOPE_GROUP[stage] == expected
    assert sp.STAGE_SCOPE_GROUP["feature_encoding"] == "preprocessing_once"
    for name in COMPOSITE_STAGES:
        assert sp.STAGE_SCOPE_GROUP[name] == "preprocessing_once"


def test_old_first_draft_stage_names_are_rejected_not_aliased():
    for old in ("decode", "encode", "face_extraction"):
        with pytest.raises(ValueError, match=f"Unknown stage '{old}'"):
            run({old: lambda: None})


def test_stage_order_is_deterministic_and_pipeline_ordered():
    rec = Recorder()
    stages = rec.stages()
    names = list(stages)
    reversed_stages = {n: stages[n] for n in reversed(names)}
    shuffled = {n: stages[n] for n in sorted(names)}
    p1, p2 = run(reversed_stages), run(shuffled)
    assert p1["stage_order"] == p2["stage_order"]
    expected = [s for s in STAGE_ORDER if s not in COMPOSITE_STAGES]
    assert p1["stage_order"] == expected
    assert expected == SPEC_STAGES


def test_composites_are_ordered_after_what_they_cover():
    for comp, covers in COMPOSITE_STAGES.items():
        assert all(STAGE_ORDER.index(c) < STAGE_ORDER.index(comp) for c in covers)


# ---------------------------------------------------------------- missing stages
def test_missing_stages_are_not_provided_with_no_latency_fields():
    rec = Recorder()
    p = run(rec.stages(("fusion",)))
    recs = by_stage(p)
    for name in SPEC_STAGES:
        if name == "fusion":
            continue
        r = recs[name]
        assert r["status"] == "not_provided"
        assert r["callable_supplied"] is False
        assert r["measurement_completed"] is False
        assert r["measurement_origin"] is None
        assert r["input_kind"] is None
        for key in LATENCY_KEYS + ("repeats", "warmup", "device", "synchronization"):
            assert key not in r
        for value in r.values():  # nothing in the record is a 0 ms stand-in
            assert value != 0.0 or value is False
        assert "not 0 ms" in r["note"]


def test_missing_feature_encoding_fusion_and_head_are_explicit():
    rec = Recorder()
    p = run(rec.stages([s for s in PRE_REQUIRED if s != "feature_encoding"]))
    assert {"feature_encoding", "fusion", "head"} <= set(p["missing_stages"])
    assert p["scope_summary"]["preprocessing_once"]["missing_stages"] == ["feature_encoding"]
    for name in ("feature_encoding", "fusion", "head"):
        assert by_stage(p)[name]["status"] == "not_provided"
        assert not any(k in by_stage(p)[name] for k in LATENCY_KEYS)
    assert p["complete"] is False


def test_no_stages_supplied():
    p = run({})
    assert p["missing_stages"] == list(REQUIRED_STAGES)
    assert p["optional_not_provided"] == list(OPTIONAL_STAGES)
    assert p["measured_stages"] == [] and p["complete"] is False
    assert p["evidence_status"] == sp.EVIDENCE_NONE_MEASURED
    assert all(r["status"] == "not_provided" for r in p["stages"])


def test_defined_stage_names_are_not_evidence_of_measurement():
    p = run(Recorder().stages(("head",)))
    assert p["defined_stages"] == SPEC_STAGES
    assert p["measured_stages"] == ["head"]
    assert set(p["defined_stages"]) - set(p["measured_stages"])
    r = by_stage(p)["head"]
    assert (r["callable_supplied"], r["measurement_completed"]) == (True, True)
    assert r["measurement_origin"] == "benchmark_callable"


def test_optional_stages_do_not_block_completeness():
    p = run(Recorder().stages(REQUIRED_STAGES))
    assert p["complete"] is True and p["missing_stages"] == []
    assert p["optional_not_provided"] == ["dataset_transfer", "archive_extraction"]
    assert all(by_stage(p)[s]["status"] == "not_provided" for s in OPTIONAL_STAGES)
    assert all(by_stage(p)[s]["required"] is False for s in OPTIONAL_STAGES)


def test_scope_summary_separates_preprocessing_from_cached_model():
    p = run(Recorder().stages(("fusion", "head")))
    cached = p["scope_summary"]["cached_feature_model"]
    pre = p["scope_summary"]["preprocessing_once"]
    assert cached["complete"] is True and cached["missing_stages"] == []
    assert pre["complete"] is False and pre["measured_stages"] == []
    assert p["complete"] is False  # preprocessing is still missing


# ------------------------------------------------------------ composite stages
def test_combined_decode_leaves_separate_stages_not_provided():
    rec = Recorder()
    p = run({"combined_audio_video_decode": rec.make("combined_audio_video_decode")})
    recs = by_stage(p)
    comp = recs["combined_audio_video_decode"]
    assert comp["status"] == "measured" and comp["stage_kind"] == "composite"
    assert comp["covers"] == ["video_decode", "audio_decode"]
    assert comp["measurement_completed"] is True
    for name in ("video_decode", "audio_decode"):
        r = recs[name]
        assert r["status"] == "not_provided"
        assert not any(k in r for k in LATENCY_KEYS)
        assert r["covered_only_by_composite"] == ["combined_audio_video_decode"]
        assert name in p["missing_stages"]
    assert "combined_audio_video_decode" not in p["measured_stages"]
    assert p["composite_measurements"] == ["combined_audio_video_decode"]
    assert any("aggregates" in c for c in p["caveats"])


def test_no_latency_is_inferred_from_a_composite():
    p = run({"combined_decode_and_array_serialization": lambda: None})
    recs = by_stage(p)
    for name in ("video_decode", "audio_decode", "raw_array_serialization"):
        assert recs[name]["status"] == "not_provided"
        assert not any(k in recs[name] for k in LATENCY_KEYS)
    assert sorted(
        n for n, r in recs.items() if r["measurement_completed"]
    ) == ["combined_decode_and_array_serialization"]


def test_separate_decode_timings_only_when_separate_callables_exist():
    rec = Recorder()
    p = run(rec.stages(("video_decode",)))
    recs = by_stage(p)
    assert recs["video_decode"]["status"] == "measured"
    assert len(recs["video_decode"]["timing_samples_ms"]) == 1
    assert recs["audio_decode"]["status"] == "not_provided"
    assert "audio_decode" in p["missing_stages"] and "video_decode" not in p["missing_stages"]


def test_separate_and_combined_decode_are_independent_measurements():
    rec = Recorder()
    p = run(rec.stages(("video_decode", "audio_decode", "combined_audio_video_decode")))
    recs = by_stage(p)
    assert all(recs[n]["status"] == "measured"
               for n in ("video_decode", "audio_decode", "combined_audio_video_decode"))
    assert rec.counts == {"video_decode": 1, "audio_decode": 1, "combined_audio_video_decode": 1}
    assert "covered_only_by_composite" not in recs["video_decode"]
    assert p["measured_stages"] == ["video_decode", "audio_decode"]  # composite not counted


def test_composite_never_makes_a_profile_complete():
    rec = Recorder()
    names = [s for s in REQUIRED_STAGES if s not in
             ("video_decode", "audio_decode", "raw_array_serialization")]
    stages = rec.stages(names)
    stages["combined_decode_and_array_serialization"] = rec.make(
        "combined_decode_and_array_serialization")
    p = run(stages)
    assert p["complete"] is False
    assert p["missing_stages"] == ["video_decode", "audio_decode", "raw_array_serialization"]


def test_combined_manifest_build_and_selection_command_cover_expected_stages():
    assert COMPOSITE_STAGES["combined_manifest_build"] == (
        "clip_validation", "manifest_row_construction", "split_audit", "manifest_serialization")
    assert COMPOSITE_STAGES["combined_selection_command"] == (
        "metadata_indexing", "split_sampling", "selection_list_generation")
    p = run({"combined_manifest_build": lambda: None})
    assert by_stage(p)["split_audit"]["covered_only_by_composite"] == ["combined_manifest_build"]
    assert by_stage(p)["metadata_indexing"].get("covered_only_by_composite") is None


def test_selection_list_generation_is_described_as_not_a_download():
    text = sp.STAGE_DESCRIPTIONS["selection_list_generation"]
    assert "does NOT download" in text
    assert "external" in sp.STAGE_DESCRIPTIONS["dataset_transfer"].lower()
    assert "not network" in sp.STAGE_DESCRIPTIONS["archive_extraction"].lower()


# ------------------------------------------------------------ settings / one-time
def test_preprocessing_stages_default_to_one_time(fake_engine):
    profile_pipeline_stages(Recorder().stages(REQUIRED_STAGES), device="cpu")
    assert len(fake_engine) == len(REQUIRED_STAGES)
    seen = [(c["warmup"], c["repeats"]) for c in fake_engine]
    # pipeline order: 12 preprocessing_once stages (incl. feature_encoding), then fusion, head
    assert seen[:12] == [(0, 1)] * 12
    assert seen[12:] == [(10, 50)] * 2


def test_one_time_stages_really_run_once_with_the_real_engine():
    rec = Recorder()
    p = run(rec.stages(), repeatable_settings=StageSettings(warmup=1, repeats=3))
    for name in PRE_REQUIRED:
        assert rec.counts[name] == 1
        assert by_stage(p)[name]["settings_source"] == "default_one_time"
        assert len(by_stage(p)[name]["timing_samples_ms"]) == 1
    for name in ("fusion", "head"):
        assert rec.counts[name] == 4  # 1 warm-up + 3 timed
        assert by_stage(p)[name]["settings_source"] == "default_repeatable"


def test_independent_per_stage_settings(fake_engine):
    stages = Recorder().stages(("split_audit", "feature_encoding", "fusion", "head"))
    profile_pipeline_stages(
        stages, device="cpu",
        stage_settings={
            "split_audit": StageSettings(warmup=2, repeats=7),
            "fusion": StageSettings(warmup=3, repeats=9),
        },
    )
    got = [(c["warmup"], c["repeats"]) for c in fake_engine]
    # order: split_audit, feature_encoding, fusion, head
    assert got == [(2, 7), (0, 1), (3, 9), (10, 50)]


def test_repeatable_default_is_configurable(fake_engine):
    profile_pipeline_stages(
        Recorder().stages(("fusion", "head")), device="cpu",
        repeatable_settings=StageSettings(warmup=4, repeats=8),
    )
    assert [(c["warmup"], c["repeats"]) for c in fake_engine] == [(4, 8), (4, 8)]


def test_stage_settings_source_is_recorded():
    p = run(Recorder().stages(("clip_validation", "fusion")),
            stage_settings={"clip_validation": StageSettings(0, 2)})
    assert by_stage(p)["clip_validation"]["settings_source"] == "stage_settings"
    assert by_stage(p)["clip_validation"]["repeats"] == 2


def test_settings_forwarded_with_percentile_and_memory_flags(fake_engine):
    profile_pipeline_stages(
        Recorder().stages(("fusion",)), device="cpu",
        percentile_method="lower", measure_memory=False,
    )
    assert fake_engine[0]["percentile_method"] == "lower"
    assert fake_engine[0]["measure_memory"] is False


def test_invalid_stage_settings_rejected():
    for bad in ((0, 0), (-1, 1)):
        with pytest.raises(ValueError):
            StageSettings(*bad)
    with pytest.raises(ValueError):
        StageSettings(warmup=True, repeats=1)  # bool is not a count
    rec = Recorder()
    with pytest.raises(TypeError, match="StageSettings"):
        run(rec.stages(), stage_settings={"head": (0, 1)})  # type: ignore[dict-item]
    assert rec.log == []


# ------------------------------------------------------------------ independence
def test_each_stage_benchmarked_independently(fake_engine):
    rec = Recorder()
    stages = rec.stages()
    profile_pipeline_stages(stages, device="cpu")
    assert len(fake_engine) == len(REQUIRED_STAGES)
    expected = [stages[n] for n in STAGE_ORDER if n in stages]
    assert [c["fn"] for c in fake_engine] == expected
    assert len({id(c["fn"]) for c in fake_engine}) == len(REQUIRED_STAGES)
    assert rec.log == []  # the profiler never runs callables outside the engine


def test_real_engine_runs_each_stage_in_its_own_contiguous_block():
    rec = Recorder()
    run(rec.stages(), repeatable_settings=StageSettings(warmup=1, repeats=2))
    expected = [n for n in STAGE_ORDER if n in rec.counts for _ in range(rec.counts[n])]
    assert rec.log == expected


def test_stage_devices_resolved_per_stage(monkeypatch, fake_engine):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    p = run(Recorder().stages(("video_decode", "feature_encoding", "fusion")),
            stage_devices={"feature_encoding": "cuda"})
    assert [c["device"] for c in fake_engine] == [
        torch.device("cpu"), torch.device("cuda:0"), torch.device("cpu")]
    assert by_stage(p)["feature_encoding"]["device"] == "cuda:0"
    assert p["device"] == "cpu"


# ------------------------------------------------------------------ failures
def test_callable_exception_is_propagated_not_swallowed():
    rec = Recorder()
    stages = rec.stages(("metadata_indexing", "video_decode", "fusion"))

    def boom():
        raise ValueError("ffmpeg exploded")

    stages["video_decode"] = boom
    with pytest.raises(StageExecutionError, match="video_decode") as info:
        run(stages)
    err = info.value
    assert err.stage == "video_decode"
    assert isinstance(err.__cause__, ValueError)
    # records finished before the failure, in pipeline order (earlier stages only)
    assert [r["stage"] for r in err.completed_stages] == [
        "metadata_indexing", "split_sampling", "selection_list_generation",
        "dataset_transfer", "archive_extraction",
    ]
    assert err.completed_stages[0]["status"] == "measured"
    assert rec.counts["fusion"] == 0  # later stages never ran


def test_failure_during_warmup_is_visible():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("warm-up failure")

    with pytest.raises(StageExecutionError, match="fusion"):
        run({"fusion": flaky}, repeatable_settings=StageSettings(warmup=1, repeats=1))


def test_failure_never_produces_a_zero_latency_record():
    def boom():
        raise RuntimeError("nope")

    with pytest.raises(StageExecutionError) as info:
        run({"head": boom})
    for r in info.value.completed_stages:
        assert not any(k in r for k in LATENCY_KEYS)


# ---------------------------------------------------------- external measurements
def test_external_measurement_recorded_without_running_or_percentiles(fake_engine):
    ext = ExternalMeasurement(durations_ms=[1234.5], source="Colab shell log, 2 Nov (test)")
    p = run({}, external_measurements={"dataset_transfer": ext})
    r = by_stage(p)["dataset_transfer"]
    assert fake_engine == []  # never passed through the engine
    assert r["status"] == "externally_measured"
    assert r["measurement_origin"] == "external"
    assert r["callable_supplied"] is False and r["measurement_completed"] is True
    assert r["timing_samples_ms"] == [1234.5] and r["repeats"] == 1
    assert r["external_source"].startswith("Colab")
    assert r["synchronization"] == "not_applicable_external"
    assert "p50_ms" not in r and "p95_ms" not in r  # no percentiles derived
    assert "dataset_transfer" in p["measured_stages"]
    assert "dataset_transfer" not in p["optional_not_provided"]


def test_external_measurement_validation():
    with pytest.raises(ValueError, match="source"):
        ExternalMeasurement(durations_ms=[1.0], source="  ")
    with pytest.raises(ValueError, match="at least one"):
        ExternalMeasurement(durations_ms=[], source="x")
    for bad in (0, 0.0, -1.0, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            ExternalMeasurement(durations_ms=[bad], source="x")
    with pytest.raises(TypeError):
        run({}, external_measurements={"split_audit": 12.0})  # type: ignore[dict-item]


def test_stage_cannot_have_callable_and_external_measurement():
    with pytest.raises(ValueError, match="both a callable and an external"):
        run({"split_audit": lambda: None},
            external_measurements={"split_audit": ExternalMeasurement([1.0], "x")})


def test_settings_and_devices_apply_only_to_callables():
    ext = {"archive_extraction": ExternalMeasurement([5.0], "shell timing (test)")}
    with pytest.raises(ValueError, match="no callable"):
        run({}, external_measurements=ext,
            stage_settings={"archive_extraction": StageSettings(0, 1)})
    with pytest.raises(ValueError, match="no callable"):
        run({}, external_measurements=ext, stage_devices={"archive_extraction": "cpu"})
    # input metadata is allowed for external measurements
    p = run({}, external_measurements=ext,
            stage_inputs={"archive_extraction": StageInput("real", "LAV-DF subset (test)")})
    assert by_stage(p)["archive_extraction"]["input_kind"] == "real"


# ------------------------------------------------------------------- input kinds
def test_synthetic_profile_is_not_research_evidence():
    p = run(Recorder().stages(), input_kind="synthetic")
    assert p["input_kind"] == "synthetic"
    assert p["evidence_status"] == sp.EVIDENCE_SYNTHETIC
    assert p["complete"] is True and p["eligible_as_real_pipeline_profile"] is False


def test_unspecified_is_the_default_and_non_research():
    p = run(Recorder().stages())
    assert p["input_kind"] == "unspecified"
    assert p["evidence_status"] == sp.EVIDENCE_UNSPECIFIED
    assert p["eligible_as_real_pipeline_profile"] is False
    assert all(r["input_kind"] == "unspecified" for r in p["stages"] if r["measurement_completed"])


def test_real_profile_requires_source_metadata():
    with pytest.raises(ValueError, match="non-empty source"):
        run(Recorder().stages(), input_kind="real")
    with pytest.raises(ValueError, match="non-empty source"):
        run(Recorder().stages(), input_kind="real", input_source="   ")
    with pytest.raises(ValueError, match="non-empty source"):
        StageInput(kind="real")
    with pytest.raises(ValueError, match="non-empty source"):
        StageInput(kind="real", source="")


def test_real_profile_records_source_and_is_caller_declared():
    p = run(Recorder().stages(), input_kind="real",
            input_source="LAV-DF test clips (placeholder description)")
    assert p["input_kind"] == "real"
    assert p["evidence_status"] == sp.EVIDENCE_REAL_DECLARED
    assert p["complete"] is True and p["eligible_as_real_pipeline_profile"] is True
    for r in p["stages"]:
        if r["measurement_completed"]:
            assert r["input_source"] == "LAV-DF test clips (placeholder description)"
    assert any("not verified" in c for c in p["caveats"])


def test_incomplete_real_profile_is_not_eligible():
    p = run(Recorder().stages(PRE_REQUIRED), input_kind="real", input_source="placeholder")
    assert p["evidence_status"] == sp.EVIDENCE_REAL_DECLARED
    assert p["complete"] is False and p["eligible_as_real_pipeline_profile"] is False


def test_per_stage_input_override_gives_mixed_non_research():
    p = run(Recorder().stages(), input_kind="real", input_source="placeholder",
            stage_inputs={"fusion": StageInput(kind="synthetic")})
    assert by_stage(p)["fusion"]["input_kind"] == "synthetic"
    assert by_stage(p)["split_audit"]["input_kind"] == "real"
    assert p["input_kind"] == "mixed" and p["evidence_status"] == sp.EVIDENCE_MIXED
    assert p["eligible_as_real_pipeline_profile"] is False


def test_invalid_input_kinds_rejected():
    for bad in ("real_dataset", "simulated", ""):
        with pytest.raises(ValueError, match="input kind"):
            run(Recorder().stages(), input_kind=bad)
    with pytest.raises(ValueError, match="input kind"):
        StageInput(kind="bogus")


# --------------------------------------------------------- validation up front
def test_validation_happens_before_any_stage_runs():
    rec = Recorder()
    stages = rec.stages()
    with pytest.raises(ValueError):
        run(stages, input_kind="real")
    with pytest.raises(ValueError, match="percentile"):
        run(stages, percentile_method="median")
    with pytest.raises(ValueError, match="no callable"):
        run(stages, stage_settings={"dataset_transfer": StageSettings(0, 1)})
    with pytest.raises(TypeError):
        run({**stages, "head": 42})  # type: ignore[arg-type]
    assert rec.log == []


def test_unknown_names_in_every_mapping_rejected():
    with pytest.raises(ValueError, match="Unknown stage 'bogus'"):
        run({"bogus": lambda: None})
    with pytest.raises(ValueError, match="Unknown stage 'bogus'"):
        run({}, external_measurements={"bogus": ExternalMeasurement([1.0], "x")})
    with pytest.raises(ValueError, match="Unknown stage 'bogus'"):
        run(Recorder().stages(("head",)), stage_settings={"bogus": StageSettings(0, 1)})


# ------------------------------------------------------ no end-to-end aggregation
def test_no_end_to_end_latency_is_invented():
    p = run(Recorder().stages())
    assert p["end_to_end_measured"] is False
    forbidden = re.compile(r"(^|_)(total|sum|e2e)(_|$)|end_to_end_latency")
    for container in [p, *p["stages"], *p["scope_summary"].values()]:
        assert not [key for key in container if forbidden.search(key)]
    caveats = " ".join(p["caveats"])
    assert "must not be summed" in caveats
    assert "never report cached-feature latency as raw-video end-to-end latency" in caveats.lower()
    assert "never mix the groups into one latency number" in caveats.lower()


def test_profiler_writes_no_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run(Recorder().stages(), input_kind="synthetic")
    assert list(tmp_path.iterdir()) == []


def test_profiler_is_decoupled_from_member_packages():
    source = Path(sp.__file__).read_text()
    imports = re.findall(r"^\s*(?:from|import)\s+(\S+)", source, flags=re.MULTILINE)
    assert not [i for i in imports if i.startswith(("avfuse.member1", "avfuse.member2"))]


def test_readme_names_every_stage_and_composite():
    readme = (Path(sp.__file__).parent / "README.md").read_text()
    for name in STAGE_ORDER:
        assert f"`{name}`" in readme, f"README does not mention {name}"
    assert "not yet implemented" not in readme.lower()
    assert "`not_provided`" in readme
