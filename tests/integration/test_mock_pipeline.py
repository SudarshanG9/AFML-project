import torch
from avfuse.member1_fusion.diagnostic_model import DiagnosticModel
from avfuse.member3_runtime.runner import run_four_modes
from avfuse.shared.contracts import PRECISION_MODES

def test_synthetic_smoke_runs_all_four_modes():
    torch.manual_seed(5)
    model=DiagnosticModel(dimension=16).eval()
    features={'audio_tokens':torch.randn(2,5,16),'video_tokens':torch.randn(2,7,16)}
    predictions=run_four_modes(model, features)
    assert tuple(predictions)==PRECISION_MODES
    for scores in predictions.values():
        assert scores.shape==(2,) and bool(((scores>=0)&(scores<=1)).all())
