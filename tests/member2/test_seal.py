import subprocess
import sys
from pathlib import Path

from avfuse.member2_data.seal import accesses
from avfuse.shared.jsonl import write_jsonl

from synthetic import manifest, predictions  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def _run(*args):
    return subprocess.run([sys.executable, str(ROOT / "scripts/m2_evaluate.py"), *args], capture_output=True,
                          text=True, env={"PYTHONPATH": str(ROOT / "src"), "PATH": ""})


def test_test_split_is_sealed_and_logged(tmp_path):
    m = manifest()
    man, pred, proto = tmp_path / "m.jsonl", tmp_path / "p.jsonl", tmp_path / "protocol.json"
    write_jsonl(man, m)
    write_jsonl(pred, predictions(m))
    assert _run("calibrate", "--manifest", str(man), "--predictions", str(pred), "--out", str(proto)).returncode == 0
    common = ["evaluate", "--manifest", str(man), "--predictions", str(pred), "--split", "test", "--n-boot", "30"]
    r = _run(*common, "--out", str(tmp_path / "t"))                                   # no protocol
    assert r.returncode != 0 and "SEALED" in r.stderr
    r = _run(*common, "--protocol", str(proto), "--out", str(tmp_path / "t"))         # no flag
    assert r.returncode != 0 and "sealed" in r.stderr and accesses(proto) == []
    r = _run(*common, "--protocol", str(proto), "--unseal-test", "--purpose", "unit test", "--out", str(tmp_path / "t"))
    assert r.returncode == 0, r.stderr
    assert [x["purpose"] for x in accesses(proto)] == ["unit test"]


def test_changed_manifest_refused(tmp_path):
    m = manifest()
    man, pred, proto = tmp_path / "m.jsonl", tmp_path / "p.jsonl", tmp_path / "protocol.json"
    write_jsonl(man, m)
    write_jsonl(pred, predictions(m))
    _run("calibrate", "--manifest", str(man), "--predictions", str(pred), "--out", str(proto))
    write_jsonl(man, m[:-1])
    r = _run("evaluate", "--manifest", str(man), "--predictions", str(pred), "--protocol", str(proto),
             "--split", "val", "--out", str(tmp_path / "v"))
    assert r.returncode != 0 and "changed" in r.stderr
