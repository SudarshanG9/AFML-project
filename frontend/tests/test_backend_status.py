from pathlib import Path

from avfuse_ui.backend_status import detect_repo_root, inspect_backend


def test_detect_repo_root_finds_checkout_ancestor(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    frontend = root / "frontend"
    frontend.mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='x'\n")
    (root / "src" / "avfuse").mkdir(parents=True)
    assert detect_repo_root(frontend) == root


def test_inspection_does_not_mark_reference_only_smoke_backend_as_ready(tmp_path: Path) -> None:
    root = tmp_path
    (root / "README.md").write_text("scaffold")
    member1 = root / "src" / "avfuse" / "member1_fusion"
    member3 = root / "src" / "avfuse" / "member3_runtime"
    member1.mkdir(parents=True)
    member3.mkdir(parents=True)
    (member1 / "dimodif_adapter.py").write_text('if mode != "reference":\n    raise NotImplementedError')
    (member1 / "model_adapter.py").write_text("@abstractmethod\nclass ModelAdapter: ...")
    (member3 / "cli.py").write_text('parser.add_argument("--smoke")\nparser.error("Real model not yet integrated")')
    status = inspect_backend(root)
    assert status["four_mode_model_ready"] is False
    assert status["real_inference_ready"] is False
    assert status["cli_status"] == "synthetic smoke only"
    assert status["adapter_mode_status"] == "reference-only in checked-in DiMoDifAdapter"



def test_inspection_detects_integrated_four_mode_cached_feature_cli(tmp_path: Path) -> None:
    member1 = tmp_path / "src" / "avfuse" / "member1_fusion"
    member3 = tmp_path / "src" / "avfuse" / "member3_runtime"
    member1.mkdir(parents=True)
    member3.mkdir(parents=True)
    (member1 / "dimodif_adapter.py").write_text(
        "def forward_with_mode(self, features, mode):\\n"
        "    with intervene_dimodif_attention(self.model, mode=mode): pass\\n"
        "def predict_from_cached_features(self, features, mode):\\n"
        "    if mode == 'reference': return self.forward_reference(features)\\n"
        "    return self.forward_with_mode(features, mode=mode)\\n"
    )
    (member1 / "dimodif_intervention.py").write_text("def fusion_attention(q, k, v, mode): ...")
    (member3 / "cli.py").write_text('"validate-config" "run" "predict" "benchmark" "inspect"\\n')
    for name in ("__main__.py", "experiment.py", "feature_loader.py", "model_benchmark.py", "predictions.py", "runner.py"):
        (member3 / name).write_text("# present\\n")
    status = inspect_backend(tmp_path)
    assert status["four_mode_model_ready"] is True
    assert status["real_cli_ready"] is True
    assert status["real_inference_ready"] is True
