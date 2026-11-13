"""Segment F CLI. Mock adapter + synthetic temp data; nothing here measures a real detector."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from avfuse.member1_fusion.dimodif_adapter import CHECKPOINTS
from avfuse.member2_data.seal import accesses
from avfuse.member3_runtime import artifacts as art
from avfuse.member3_runtime import cli
from avfuse.shared.contracts import PRECISION_MODES

CKPT_BYTES = b"synthetic checkpoint bytes (test only)"
CKPT_SHA = hashlib.sha256(CKPT_BYTES).hexdigest()


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


class MockModel:
    device = torch.device("cpu")
    training = False

    def __init__(self, supported=PRECISION_MODES, sha=CKPT_SHA, boom=False):
        self.supported, self.checkpoint_sha256, self.boom = set(supported), sha, boom
        self.calls = []

    def predict_from_cached_features(self, features, mode):
        self.calls.append(mode)
        if self.boom:
            raise RuntimeError("kernel exploded")
        if mode not in self.supported:
            raise NotImplementedError(f"{mode} not integrated (Member 1)")
        return torch.tensor([0.3])


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(art, "collect_environment_report", lambda root: fake_report())
    up = tmp_path / "upstream"
    stem = CHECKPOINTS["lavdf"]
    (up / "ckpt" / "dfd").mkdir(parents=True)
    (up / "ckpt" / "dfd" / f"{stem}.pth").write_bytes(CKPT_BYTES)
    (up / "ckpt" / "dfd" / f"{stem}.json").write_text("{}")
    feats = tmp_path / "feats"
    rows = []
    for split, ids in (("val", ["v1", "v2"]), ("test", ["t1", "t2"])):
        for cid in ids:
            d = feats / split / cid
            d.mkdir(parents=True)
            np.savez(d / "features.npz", video_features=np.ones((4, 768), np.float32),
                     audio_features=np.ones((3, 768), np.float32))
            rows.append({"clip_id": cid, "split": split, "label": 1, "feature_path": f"x/{cid}.pt"})
    manifest = tmp_path / "manifest_whole.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}")
    state = {"model": MockModel()}
    monkeypatch.setattr(cli, "_build_model", lambda args: state["model"])
    base = ["--upstream-root", str(up), "--checkpoint-name", "lavdf", "--manifest", str(manifest),
            "--feature-root", str(feats)]
    runopts = ["--runs-root", str(tmp_path / "runs"), "--warmup", "1", "--repeats", "2",
               "--no-memory"]
    return {"tmp": tmp_path, "base": base, "run": runopts, "protocol": protocol,
            "state": state, "up": up, "feats": feats}


def go(*argv):
    return cli.main([str(a) for a in argv])


def run_dirs(w):
    root = w["tmp"] / "runs"
    return sorted(root.iterdir()) if root.exists() else []


# --------------------------------------------------------------------------- help / usage
def test_help_and_subcommand_help_exit_zero(capsys):
    for argv in (["--help"], ["run", "--help"], ["predict", "--help"], ["benchmark", "--help"],
                 ["validate-config", "--help"], ["inspect", "--help"]):
        with pytest.raises(SystemExit) as ei:
            cli.main(argv)
        assert ei.value.code == 0
    assert "Exit codes" in capsys.readouterr().out


def test_no_command_is_usage_error(capsys):
    assert cli.main([]) == cli.EXIT_USAGE


def test_unknown_command_and_bad_choice_are_argparse_usage_errors():
    for argv in (["frobnicate"], ["run", "--checkpoint-name", "nope"], ["run", "--split", "train"]):
        with pytest.raises(SystemExit) as ei:
            cli.main(argv)
        assert ei.value.code == 2


@pytest.mark.parametrize("extra,match", [
    (["--modes", "int4"], "unknown mode"),
    (["--modes", "reference", "reference"], "duplicates"),
    (["--device", "auto"], "--device"),
    (["--repeats", "0"], "repeats"),
    (["--max-clips", "0"], "max-clips"),
    (["--expected-checkpoint-sha256", "XYZ"], "64 lowercase"),
    (["--protocol", "p.json"], "only to --split test"),
    (["--unseal-test"], "only to --split test"),
    (["--require-lavdf-hash", "--expected-checkpoint-sha256", "a" * 64], "either"),
])
def test_invalid_arguments_exit_2(world, capsys, extra, match):
    assert go("run", *world["base"], *world["run"], *extra) == cli.EXIT_USAGE
    assert match in capsys.readouterr().err
    assert run_dirs(world) == []


def test_required_args_missing(capsys):
    assert go("run") == cli.EXIT_USAGE
    assert "--upstream-root is required" in capsys.readouterr().err


def test_missing_inputs_exit_6(world, capsys):
    base = list(world["base"])
    base[base.index("--manifest") + 1] = str(world["tmp"] / "nope.jsonl")
    assert go("run", *base, *world["run"]) == cli.EXIT_MISSING
    assert "manifest not found" in capsys.readouterr().err
    base = list(world["base"])
    base[base.index("--feature-root") + 1] = str(world["tmp"] / "nofeats")
    assert go("run", *base) == cli.EXIT_MISSING


# ------------------------------------------------------------------- validate-config
def test_validate_config_ok_loads_no_model(world, capsys, monkeypatch):
    monkeypatch.setattr(cli, "_build_model", lambda a: pytest.fail("model must not be built"))
    assert go("validate-config", *world["base"], "--json") == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] and out["clips_in_split"] == 2 and out["test_access"] == "not requested"


def test_validate_config_missing_checkpoint_exit_6(world):
    (world["up"] / "ckpt" / "dfd" / f"{CHECKPOINTS['lavdf']}.pth").unlink()
    assert go("validate-config", *world["base"]) == cli.EXIT_MISSING


def test_validate_config_bad_config(world, tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert go("validate-config", *world["base"], "--config", bad) == cli.EXIT_USAGE
    bad.write_text('{"device": "tpu"}')
    assert go("validate-config", *world["base"], "--config", bad) == cli.EXIT_USAGE
    assert go("validate-config", *world["base"], "--config", tmp_path / "gone.json") \
        == cli.EXIT_MISSING


# ---------------------------------------------------------------------------- run: success
def test_run_val_four_modes_complete_exit_0(world, capsys):
    assert go("run", *world["base"], *world["run"], "--json") == cli.EXIT_OK
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "complete" and out["finalized"] and out["four_mode_complete"]
    assert out["labels_accessed"] is False and out["data_origin"] == "real_cached_features"
    (rd,) = run_dirs(world)
    assert art.verify_run_integrity(rd)["ok"]
    rows = [json.loads(x) for x in (rd / "predictions.jsonl").read_text().splitlines()]
    assert len(rows) == 8 and {r["split"] for r in rows} == {"val"}
    assert {r["checkpoint_sha256"] for r in rows} == {CKPT_SHA}
    assert all("label" not in r for r in rows)
    recs = [json.loads(x) for x in (rd / "benchmark_records.jsonl").read_text().splitlines()]
    assert {r["scope"] for r in recs} == {"cached_feature_model"}
    assert {r["workload"]["checkpoint_sha256"] for r in recs} == {CKPT_SHA}


def test_inspect_complete_run(world, capsys):
    go("run", *world["base"], *world["run"])
    capsys.readouterr()
    (rd,) = run_dirs(world)
    assert go("inspect", rd, "--json") == cli.EXIT_OK
    out = json.loads(capsys.readouterr().out)
    assert out["integrity_ok"] and out["complete_four_mode_validation_run"]
    assert "Not a release approval" in out["note"]


def test_inspect_detects_tampering(world, capsys):
    go("run", *world["base"], *world["run"])
    capsys.readouterr()
    (rd,) = run_dirs(world)
    (rd / "predictions.jsonl").write_text("tampered\n")
    assert go("inspect", rd, "--json") == cli.EXIT_FAILED
    assert json.loads(capsys.readouterr().out)["integrity_ok"] is False


def test_inspect_missing_dir_and_empty_dir(world, tmp_path):
    assert go("inspect", tmp_path / "nope") == cli.EXIT_MISSING
    (tmp_path / "empty").mkdir()
    assert go("inspect", tmp_path / "empty") == cli.EXIT_INCOMPLETE


# ------------------------------------------------------- unsupported / incomplete / failed
def test_reference_only_adapter_four_modes_is_blocked_exit_3(world, capsys):
    world["state"]["model"] = MockModel(supported=("reference",))
    assert go("run", *world["base"], *world["run"]) == cli.EXIT_BLOCKED
    out = capsys.readouterr().out
    assert "status=blocked" in out and "UNSUPPORTED low_precision" in out
    (rd,) = run_dirs(world)
    assert not (rd / "predictions.jsonl").exists() and not (rd / "run_manifest.json").exists()
    assert go("inspect", rd) == cli.EXIT_BLOCKED


def test_reference_only_request_succeeds_but_is_not_four_mode(world, capsys):
    world["state"]["model"] = MockModel(supported=("reference",))
    assert go("run", *world["base"], *world["run"], "--modes", "reference", "--json") == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "complete" and out["four_mode_complete"] is False
    (rd,) = run_dirs(world)
    assert go("inspect", rd, "--json") == cli.EXIT_OK
    assert json.loads(capsys.readouterr().out)["complete_four_mode_validation_run"] is False


def test_allow_partial_is_incomplete_exit_4_and_not_finalized(world, capsys):
    world["state"]["model"] = MockModel(supported=("reference",))
    assert go("run", *world["base"], *world["run"], "--allow-partial") == cli.EXIT_INCOMPLETE
    (rd,) = run_dirs(world)
    assert not (rd / "run_manifest.json").exists()
    assert go("inspect", rd) == cli.EXIT_INCOMPLETE


def test_max_clips_subset_is_incomplete_exit_4(world, capsys):
    assert go("run", *world["base"], *world["run"], "--max-clips", "1", "--json") \
        == cli.EXIT_INCOMPLETE
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "partial" and out["finalized"] is False
    assert out["dataset_complete"] is False
    (rd,) = run_dirs(world)
    assert not (rd / "run_manifest.json").exists()


def test_runtime_failure_exit_1_with_failure_summary(world, capsys):
    world["state"]["model"] = MockModel(boom=True)
    assert go("run", *world["base"], *world["run"]) == cli.EXIT_FAILED
    assert "ERROR[FAILED]" in capsys.readouterr().err
    (rd,) = run_dirs(world)
    assert (rd / "failure_summary.json").is_file() and not (rd / "run_manifest.json").exists()
    assert go("inspect", rd) == cli.EXIT_FAILED


def test_checkpoint_mismatch_exit_7_before_any_inference(world, capsys):
    model = world["state"]["model"]
    assert go("run", *world["base"], *world["run"], "--expected-checkpoint-sha256", "ab" * 32) \
        == cli.EXIT_CHECKPOINT
    assert go("run", *world["base"], *world["run"], "--require-lavdf-hash") == cli.EXIT_CHECKPOINT
    assert model.calls == [] and run_dirs(world) == []


def test_existing_run_id_is_usage_error_unless_overwrite(world, capsys):
    assert go("run", *world["base"], *world["run"], "--run-id", "fixed1") == 0
    assert go("run", *world["base"], *world["run"], "--run-id", "fixed1") == cli.EXIT_USAGE
    assert "already exists" in capsys.readouterr().err


# ------------------------------------------------------------------------ sealed test
def test_test_split_without_authorization_exit_5_and_no_access(world, capsys):
    model = world["state"]["model"]
    p = world["protocol"]
    for extra in ([], ["--protocol", p], ["--protocol", p, "--unseal-test"],
                  ["--protocol", p, "--unseal-test", "--purpose", "  "],
                  ["--unseal-test", "--purpose", "final"],
                  ["--protocol", world["tmp"] / "none.json", "--unseal-test", "--purpose", "f"]):
        assert go("run", *world["base"], *world["run"], "--split", "test", *extra) \
            == cli.EXIT_SEALED
    assert go("predict", *world["base"], "--split", "test", "--output",
              world["tmp"] / "t.jsonl") == cli.EXIT_SEALED
    assert model.calls == [] and run_dirs(world) == [] and accesses(p) == []


def test_authorized_test_run_logs_once_and_skips_benchmark(world, capsys):
    p = world["protocol"]
    assert go("run", *world["base"], *world["run"], "--split", "test", "--protocol", p,
              "--unseal-test", "--purpose", "final held-out run", "--json") == 0
    out = json.loads(capsys.readouterr().out)
    assert out["split"] == "test" and "benchmark" not in out
    log = accesses(p)
    assert len(log) == 1 and log[0]["purpose"] == "final held-out run"
    assert log[0]["configs"] == sorted(PRECISION_MODES)
    (rd,) = run_dirs(world)
    assert not (rd / "benchmark_records.jsonl").exists()
    assert go("inspect", rd, "--json") == cli.EXIT_OK
    assert json.loads(capsys.readouterr().out)["complete_four_mode_validation_run"] is False


def test_explicit_benchmark_with_test_is_rejected(world):
    p = world["protocol"]
    assert go("run", *world["base"], *world["run"], "--split", "test", "--protocol", p,
              "--unseal-test", "--purpose", "x", "--benchmark") == cli.EXIT_USAGE
    assert accesses(p) == []


def test_blocked_test_run_never_opens_the_seal(world, capsys):
    """Unsupported modes are caught by a validation-clip preflight before the ledger is written."""
    world["state"]["model"] = MockModel(supported=("reference",))
    p = world["protocol"]
    assert go("run", *world["base"], *world["run"], "--split", "test", "--protocol", p,
              "--unseal-test", "--purpose", "final") == cli.EXIT_BLOCKED
    assert "NOT opened" in capsys.readouterr().err
    assert accesses(p) == []
    assert run_dirs(world) == []


def test_validation_run_never_reads_test_features(world, monkeypatch):
    seen = []
    real = np.load
    monkeypatch.setattr("avfuse.member3_runtime.feature_loader.np.load",
                        lambda p, **k: seen.append(str(p)) or real(p, **k))
    assert go("run", *world["base"], *world["run"], "--no-hash-cache") == 0
    assert seen and all("val" in Path(s).parts for s in seen)


def test_test_preflight_fails_closed_without_validation_clips(world, capsys):
    # Keep only test rows: preflight must refuse before opening the test ledger.
    rows = [
        {"clip_id": "t1", "split": "test", "label": 1, "feature_path": "x/t1.pt"},
        {"clip_id": "t2", "split": "test", "label": 0, "feature_path": "x/t2.pt"},
    ]
    manifest = world["tmp"] / "test_only_manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\\n" for row in rows))
    p = world["protocol"]
    assert go(
        "run", "--upstream-root", world["up"], "--checkpoint-name", "lavdf",
        "--manifest", manifest, "--feature-root", world["feats"], *world["run"],
        "--split", "test", "--protocol", p, "--unseal-test",
        "--purpose", "final held-out run",
    ) == cli.EXIT_BLOCKED
    assert "validation preflight" in capsys.readouterr().err.lower()
    assert accesses(p) == []
    assert run_dirs(world) == []


# ------------------------------------------------------------------------ predict / benchmark
def test_predict_writes_validated_jsonl(world, capsys):
    out = world["tmp"] / "pred.jsonl"
    assert go("predict", *world["base"], "--output", out, "--run-id", "r1", "--json") == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["published"] and rep["n_rows"] == 8 and rep["full_split"]
    from avfuse.member2_data.predictions import read_predictions
    assert len(read_predictions(out)) == 8


def test_predict_blocked_and_subset_exit_codes(world):
    out = world["tmp"] / "pred.jsonl"
    world["state"]["model"] = MockModel(supported=("reference",))
    assert go("predict", *world["base"], "--output", out) == cli.EXIT_BLOCKED
    assert not out.exists()
    world["state"]["model"] = MockModel()
    assert go("predict", *world["base"], "--output", out, "--max-clips", "1") \
        == cli.EXIT_INCOMPLETE
    assert go("predict", *world["base"], "--output", out) == cli.EXIT_USAGE  # exists, no overwrite


def test_benchmark_command_complete_and_blocked(world, capsys):
    assert go("benchmark", *world["base"], *world["run"], "--json") == 0
    out = json.loads(capsys.readouterr().out)
    assert out["scope"] == "cached_feature_model" and out["finalized"] and out["split"] == "val"
    (rd,) = run_dirs(world)
    assert art.verify_run_integrity(rd)["ok"]
    world["state"]["model"] = MockModel(supported=("reference",))
    assert go("benchmark", *world["base"], *world["run"]) == cli.EXIT_BLOCKED
    assert go("benchmark", *world["base"], *world["run"], "--benchmark-clip-id", "zzz") \
        == cli.EXIT_USAGE


def test_benchmark_has_no_test_options():
    with pytest.raises(SystemExit):
        cli.main(["benchmark", "--split", "test"])


# ----------------------------------------------------------------------------- smoke
def test_legacy_smoke_prints_all_four_modes(capsys):
    assert cli.main(["--smoke"]) == 0
    out = capsys.readouterr().out
    assert "DIAGNOSTIC ONLY" in out
    for m in PRECISION_MODES:
        assert m in out


def test_python_dash_m_entry_point(world):
    import subprocess
    import sys
    r = subprocess.run([sys.executable, "-m", "avfuse.member3_runtime", "--help"],
                       capture_output=True, text=True, check=False)
    assert r.returncode == 0 and "validate-config" in r.stdout
