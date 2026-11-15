"""Audit matched/upstream attention inputs for the five-layer DiMoDif intervention."""
import argparse
import json
from pathlib import Path
import sys

import torch

# Direct `python scripts/<name>.py` execution must also find the sibling scripts package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from avfuse.member1_fusion.dimodif_adapter import DiMoDifAdapter
from scripts.dayB_inference import load_features


def capture_inputs(adapter, features, mode):
    layers = adapter.model.transformer_encoder.layers
    inputs = []
    hooks = []

    def make_hook(layer_index):
        def hook(_module, args):
            inputs.append((layer_index, args[0].detach().cpu().clone()))
        return hook

    try:
        for i, layer in enumerate(layers):
            hooks.append(layer.self_attn.register_forward_pre_hook(make_hook(i)))
        result = adapter.forward_with_mode(features, mode=mode)
    finally:
        for handle in hooks:
            handle.remove()

    inputs.sort(key=lambda item: item[0])
    if len(inputs) != len(layers):
        raise RuntimeError(f'Expected {len(layers)} attention inputs, got {len(inputs)}')
    return [tensor for _, tensor in inputs], float(result['score_fake'][0].item())


def audit(adapter, features):
    low, low_score = capture_inputs(adapter, features, 'low_precision')
    restored, restored_score = capture_inputs(adapter, features, 'restore_value_output')
    assert len(low) == len(restored)
    rows = []
    for i, (a, b) in enumerate(zip(low, restored)):
        assert a.shape == b.shape and a.dtype == b.dtype
        diff = (a - b).abs()
        rows.append({
            'layer': i,
            'input_shape': list(a.shape),
            'upstream_bit_exact': bool(torch.equal(a, b)),
            'upstream_max_abs_difference': float(diff.max().item()),
            'upstream_mean_abs_difference': float(diff.mean().item()),
        })
    if not rows[0]['upstream_bit_exact']:
        raise RuntimeError('Layer-0 upstream is unexpectedly different')
    return {
        'description': 'Descriptive whole-model upstream audit, not per-layer causal isolation',
        'low_precision_score': low_score,
        'restore_value_output_score': restored_score,
        'layers': rows,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--upstream', default='third_party/dimodif')
    p.add_argument('--device', default='cpu')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    adapter = DiMoDifAdapter(a.upstream, checkpoint_name='lavdf', device=a.device)
    expected_sha = 'e1e59006d1bb6ad36a3859af5486a6649ed41bba23258b7c41e24f873be7948e'
    if adapter.checkpoint_sha256 != expected_sha:
        raise RuntimeError('Unapproved detector checkpoint')
    features = load_features(a.features)
    with torch.inference_mode():
        result = audit(adapter, features)
    result['checkpoint_sha256'] = adapter.checkpoint_sha256
    result['feature_file'] = str(a.features.resolve())
    result['device'] = str(adapter.device)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for row in result['layers']:
        print(row)
    print('Wrote', a.out)


if __name__ == '__main__':
    main()
