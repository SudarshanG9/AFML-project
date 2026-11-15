from pathlib import Path

from avfuse_ui.backend_status import detect_repo_root, inspect_backend


def test_detect_repo_root_finds_checkout_ancestor(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    frontend = root / "frontend"
    frontend.mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='x'\n")
    (root / "src" / "avfuse").mkdir(parents=True)
    assert detect_repo_root(frontend) == root


def test_missing_source_does_not_imply_readiness(tmp_path):
    status = inspect_backend(tmp_path)
    assert not status["source_available"]
    assert not status["real_inference_ready"]


def test_actual_backend_interfaces_without_model_loading():
    root = Path(__file__).resolve().parents[2]
    status = inspect_backend(root)
    assert status["source_available"]
    assert status["interfaces_imported"]
    assert status["assets_available"].startswith("Not checked")
    assert status["run_completed"] == "Not checked"
