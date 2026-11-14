"""Reference-only, five-layer CPU parity; stops before full cohort inference."""
import argparse
from contextlib import ExitStack
from pathlib import Path
from common import ROOT, context, save, review_bundle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    args = p.parse_args()
    run, c = context(args.run_dir)
    assert not (run / 'runtime_parity.json').exists()
    import torch
    from avfuse.member1_fusion.dimodif_adapter import DiMoDifAdapter
    from scripts.dayB_inference import load_features
    torch.manual_seed(42)
    item = next(x for x in c['approved_workloads'] if x['role'] == 'typical_anchor')
    adapter = DiMoDifAdapter(c['upstream'], checkpoint_name='lavdf', device='cpu')
    assert not adapter.model.training and all(not x.requires_grad and x.dtype == torch.float32 for x in adapter.model.parameters())
    frozen = {k: v.detach().clone() for k, v in adapter.model.state_dict().items()}
    features = load_features(Path(item['feature_path']))
    captured = []
    def capture(_module, _inputs, output):
        captured.append(output[0].detach().clone())
    layers = adapter.model.transformer_encoder.layers
    assert len(layers) == 5
    trace = []
    with ExitStack() as stack:
        for layer in layers:
            stack.callback(layer.self_attn.register_forward_hook(capture).remove)
        with torch.inference_mode():
            native = adapter.forward_reference(features)
            native_attention = captured[:]
            captured.clear()
            wrapped = adapter.forward_with_mode(features, 'reference', trace=trace)
    assert len(native_attention) == len(captured) == 5 and len(trace) == 20
    per_layer = []
    for index, (a, b) in enumerate(zip(native_attention, captured)):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)
        per_layer.append({'layer': index, 'max_abs_error': float((a-b).abs().max()), 'bit_exact': torch.equal(a,b)})
    torch.testing.assert_close(native['logits'], wrapped['logits'], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(native['score_fake'], wrapped['score_fake'], atol=1e-6, rtol=1e-5)
    assert all(torch.equal(v, adapter.model.state_dict()[k]) for k, v in frozen.items())
    assert all(x['fraction_changed'] == 0 for x in trace)
    save(run / 'runtime_parity.json', {'status': 'PASS', 'device': 'cpu', 'torch': torch.__version__,
         'checkpoint_sha256': adapter.checkpoint_sha256, 'clip_id': item['clip_id'],
         'feature_sha256': item['feature_sha256'], 'valid_length': native['valid_length'],
         'per_layer': per_layer, 'trace': trace, 'model_weights_unchanged': True,
         'logit_max_abs_error': float((native['logits']-wrapped['logits']).abs().max()),
         'score_max_abs_error': float((native['score_fake']-wrapped['score_fake']).abs().max()),
         'native_score': float(native['score_fake'].item()), 'wrapped_score': float(wrapped['score_fake'].item()),
         'logit_tolerances': {'atol':1e-5, 'rtol':1e-4}, 'score_tolerances': {'atol':1e-6,'rtol':1e-5},
         'full_reference_cohort_run': False, 'test_feature_access': False})
    review_bundle(run, 'runtime_parity')

if __name__ == '__main__':
    main()
