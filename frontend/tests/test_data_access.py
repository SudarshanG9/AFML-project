from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from avfuse_ui.data_access import (
    documented_shortcut_table,
    documented_validation_table,
    documented_test_results,
    discover_files,
    feature_file_summary_bytes,
    parse_jsonl_text,
    validate_manifest,
    validate_predictions,
)


def test_parse_jsonl_skips_blank_lines_and_reports_line_number() -> None:
    assert parse_jsonl_text('\n{"a": 1}\n\n{"b": 2}\n') == [{"a": 1}, {"b": 2}]
    with pytest.raises(ValueError, match="line 2"):
        parse_jsonl_text('\nnot-json\n')


def test_manifest_checks_schema_duplicates_and_group_leakage(tmp_path: Path) -> None:
    schema = tmp_path / "manifest.schema.json"
    schema.write_text(json.dumps({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["clip_id", "dataset", "source_id", "identity_id", "split", "label", "manipulation_type", "feature_path"],
        "properties": {
            "clip_id": {"type": "string"}, "dataset": {"type": "string"},
            "source_id": {"type": "string"}, "identity_id": {"type": "string"},
            "split": {"enum": ["train", "val", "test"]}, "label": {"enum": [0, 1]},
            "manipulation_type": {"enum": ["real", "audio_fake", "video_fake", "both_fake"]},
            "feature_path": {"type": "string"},
        },
        "additionalProperties": False,
    }))
    rows = [
        {"clip_id": "c1", "dataset": "d", "source_id": "s1", "identity_id": "i1", "split": "train", "label": 0, "manipulation_type": "real", "feature_path": "a.pt"},
        {"clip_id": "c2", "dataset": "d", "source_id": "s1", "identity_id": "i2", "split": "val", "label": 1, "manipulation_type": "audio_fake", "feature_path": "b.pt"},
    ]
    errors = validate_manifest(rows, schema)
    assert any("source-group" in error for error in errors)


def test_prediction_rows_reject_ground_truth_and_duplicates(tmp_path: Path) -> None:
    schema = tmp_path / "prediction.schema.json"
    schema.write_text(json.dumps({
        "$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object",
        "required": ["run_id", "clip_id", "split", "config", "score_fake", "checkpoint_sha256"],
        "properties": {
            "run_id": {"type": "string"}, "clip_id": {"type": "string"},
            "split": {"enum": ["val", "test"]},
            "config": {"enum": ["reference", "low_precision", "restore_value_output", "restore_query"]},
            "score_fake": {"type": "number", "minimum": 0, "maximum": 1},
            "checkpoint_sha256": {"type": "string"},
        }, "additionalProperties": False,
    }))
    row = {"run_id": "r", "clip_id": "c", "split": "val", "config": "reference", "score_fake": 0.7, "checkpoint_sha256": "a"}
    errors = validate_predictions([row, {**row, "label": 1}], schema)
    assert any("ground-truth" in error for error in errors)
    errors = validate_predictions([row, row], schema)
    assert any("duplicate prediction" in error for error in errors)


def test_feature_npz_upload_is_inspected_in_memory() -> None:
    buff = io.BytesIO()
    np.savez(buff, audio_features=np.zeros((12, 768), dtype=np.float32), video_features=np.zeros((10, 768), dtype=np.float32))
    info = feature_file_summary_bytes(buff.getvalue(), "cache.npz")
    assert info["contract_compatible"] is True
    assert info["loader_format"] == "dimodif_npz"
    assert info["shapes"]["audio_features"] == [12, 768]


def test_feature_pt_upload_checks_shared_token_contract() -> None:
    buff = io.BytesIO()
    torch.save({"audio_tokens": torch.zeros(1, 12, 768), "video_tokens": torch.zeros(1, 10, 768)}, buff)
    info = feature_file_summary_bytes(buff.getvalue(), "cache.pt")
    assert info["contract_compatible"] is True
    assert info["tensors"]["audio_tokens"]["shape"] == [1, 12, 768]


def test_documented_tables_parse_repository_formats() -> None:
    val = """| Mode | AUROC | AUPRC | Recall |\n|---|---:|---:|---:|\n| reference | 0.9984 | 0.9995 | 0.9933 |\n| low_precision | 0.9984 | 0.9995 | 0.9933 |\n"""
    assert len(documented_validation_table(val)) == 2
    audit = """| Feature | val AUROC | test AUROC | Verdict |\n|---|---:|---:|---|\n| **trailing silence** | **0.682** | **0.660** | **FLAG** |\n| duration | 0.628 | 0.621 | below the rule |\n"""
    rows = documented_shortcut_table(audit)
    assert len(rows) == 2
    assert rows[0]["feature"] == "trailing silence"
    assert rows[0]["test_shortcut_auroc"] == pytest.approx(0.660)


def test_discovery_finds_benchmark_records_inside_run_bundle(tmp_path: Path) -> None:
    p = tmp_path / "artifacts" / "runs" / "run-123"
    p.mkdir(parents=True)
    (p / "benchmark_records.jsonl").write_text("{}\n")
    reports = tmp_path / "artifacts" / "reports" / "checks"
    reports.mkdir(parents=True)
    (reports / "check_reference_val.json").write_text("{}")
    day2 = tmp_path / "artifacts" / "day2"
    day2.mkdir(parents=True)
    (day2 / "attention_audit.json").write_text("{}")
    found = discover_files(tmp_path)
    assert p / "benchmark_records.jsonl" in found["benchmarks"]
    assert reports / "check_reference_val.json" in found["prediction_checks"]
    assert day2 / "attention_audit.json" in found["attention_audits"]



def test_discovery_finds_nested_manifest_and_member3_run_layout(tmp_path: Path) -> None:
    manifest = tmp_path / "data" / "pilot" / "test_day2" / "manifest_rows.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}\\n")
    run = tmp_path / "artifacts" / "member3" / "runs" / "validation-run-1"
    run.mkdir(parents=True)
    (run / "run_manifest.json").write_text("{}")
    (run / "predictions.jsonl").write_text("{}\\n")
    (run / "benchmark_records.jsonl").write_text("{}\\n")
    (run / "stage_profile.json").write_text("{}")
    found = discover_files(tmp_path)
    assert manifest in found["manifests"]
    assert run in found["run_dirs"]
    assert run / "predictions.jsonl" in found["predictions"]
    assert run / "benchmark_records.jsonl" in found["benchmarks"]
    assert run / "stage_profile.json" in found["stage_profiles"]


def test_documented_test_results_parse_published_heldout_report() -> None:
    report = '''
- Predictions: Member 1's run `C10_RECOVERY_R2`, 4 modes x 2,000 test videos.
- Manifest: `manifest_whole.jsonl` (sha `8339b247...`), `split == test`: 500 real, 500 audio_fake, 500 video_fake, 500 both_fake, 1,738 source groups.
- Threshold: 0.6627789735794068, frozen on val (protocol sha `358c0372...`).
- Bootstrap: 1,000 resamples, seed 0, whole source groups.
- Test seal: opened once on 2026-11-14 (ledger has 2 lines from the same session: the evaluation, then the trailing-silence control).
Member 1's inference took 3 attempts.
| Mode | AUROC [95% CI] | AUPRC | BCE | Recall | Real FPR |
|---|---|---|---|---|---|
| reference | 0.9980 [0.9969, 0.9989] | 0.9994 | 0.0678 | 0.992 | 0.042 |
| low_precision | 0.9980 [0.9969, 0.9989] | 0.9993 | 0.0681 | 0.992 | 0.042 |
| restore_value_output | 0.9980 [0.9969, 0.9989] | 0.9993 | 0.0682 | 0.992 | 0.042 |
| restore_query | 0.9980 [0.9969, 0.9989] | 0.9994 | 0.0677 | 0.992 | 0.042 |

Reference, per fake type vs real:
| Fake type | AUROC | Recall |
|---|---|---|
| audio_fake | 0.9990 | 0.994 |
| video_fake | 0.9955 | 0.984 |
| both_fake | 0.9996 | 0.998 |

Paired BCE differences (same videos):
- low_precision vs reference: +0.0003 [-0.0002, +0.0010]
- restore_value_output vs low_precision: +0.0001 [-0.0004, +0.0005]
- restore_query vs low_precision: -0.0004 [-0.0010, +0.0001]
- restore_value_output vs restore_query: +0.0005 [-0.0002, +0.0012]
'''
    parsed = documented_test_results(report)
    assert len(parsed['rows']) == 4
    assert parsed['rows'][0]['auroc'] == pytest.approx(0.9980)
    assert parsed['rows'][0]['auroc_ci_low'] == pytest.approx(0.9969)
    assert parsed['metadata']['run_id'] == 'C10_RECOVERY_R2'
    assert parsed['metadata']['total_test_videos'] == 2000
    assert parsed['metadata']['source_groups'] == 1738
    assert parsed['metadata']['threshold'] == pytest.approx(0.6627789735794068)
    assert len(parsed['per_type']) == 3
    assert len(parsed['paired_bce']) == 4
