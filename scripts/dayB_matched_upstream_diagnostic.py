"""Separate G7 diagnostic: one restored layer, all other layers low precision.

No labels, accuracy evaluation, training or replacement of the main four modes.
Local target output is measured AFTER the unchanged FP32 output projection.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from types import MethodType

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from avfuse.member1_fusion.dimodif_adapter import DiMoDifAdapter
from avfuse.member1_fusion.dimodif_intervention import intervene_dimodif_attention, _make_forward
from avfuse.member1_fusion.precision import prepare_qkv
from scripts.dayB_inference import load_features


@contextmanager
def diagnostic_intervention(model, target, trace):
    layers = list(model.transformer_encoder.layers)
    if target is not None and (not isinstance(target, int) or not 0 <= target < len(layers)):
        raise ValueError('Target layer outside model')
    with intervene_dimodif_attention(model, 'low_precision', trace=trace):
        if target is not None:
            module = layers[target].self_attn
            module.forward = MethodType(_make_forward(target, 'restore_value_output', trace), module)
        yield


def capture_run(model, forward, target=None):
    layers = list(model.transformer_encoder.layers)
    inputs, outputs, trace, hooks = {}, {}, [], []

    def pre(i):
        def record(_module, args):
            if i in inputs:
                raise RuntimeError('Attention layer called more than once')
            inputs[i] = args[0].detach().cpu().clone()
        return record

    def post(i):
        def record(_module, _args, output):
            outputs[i] = output[0].detach().cpu().clone()
        return record

    try:
        for i, layer in enumerate(layers):
            hooks.append(layer.self_attn.register_forward_pre_hook(pre(i)))
            hooks.append(layer.self_attn.register_forward_hook(post(i)))
        with torch.inference_mode(), diagnostic_intervention(model, target, trace):
            result = forward()
    finally:
        for hook in hooks:
            hook.remove()
    if set(inputs) != set(range(len(layers))) or set(outputs) != set(inputs):
        raise RuntimeError('Missing layer hooks; do not claim upstream parity')
    if len(trace) != 4 * len(layers):
        raise RuntimeError('Missing precision boundary trace')
    return {'inputs': inputs, 'outputs': outputs, 'trace': trace, 'result': result}


def tensor_difference(a, b):
    if a.shape != b.shape or a.dtype != b.dtype or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise RuntimeError('Incompatible or nonfinite diagnostic tensors')
    error = (a - b).abs()
    return {'bit_exact': bool(torch.equal(a, b)), 'max_abs': float(error.max()),
            'mean_abs': float(error.mean()), 'shape': list(a.shape), 'dtype': str(a.dtype)}


def projected_qkv(module, inputs):
    x = inputs.to(module.in_proj_weight.device)
    b, t, _ = x.shape
    return F.linear(x, module.in_proj_weight, module.in_proj_bias).reshape(
        b, t, 3, module.num_heads, module.head_dim).permute(2, 0, 3, 1, 4).unbind(0)


@torch.inference_mode()
def audit_matched(model, forward, score_of):
    layers = list(model.transformer_encoder.layers)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    original_forwards = [layer.self_attn.forward for layer in layers]
    hook_counts = [(len(layer.self_attn._forward_pre_hooks), len(layer.self_attn._forward_hooks)) for layer in layers]
    baseline = capture_run(model, forward)
    baseline_score = score_of(baseline['result'])
    rows = []
    for target, layer in enumerate(layers):
        restored = capture_run(model, forward, target)
        comparisons = [tensor_difference(baseline['inputs'][i], restored['inputs'][i]) for i in range(len(layers))]
        if not all(r['bit_exact'] for r in comparisons[:target + 1]):
            raise RuntimeError('Matched low-precision prefix/target inputs failed')
        q, k, v = projected_qkv(layer.self_attn, baseline['inputs'][target])
        rq, rk, rv = projected_qkv(layer.self_attn, restored['inputs'][target])
        lq, lk, _ = prepare_qkv(q, k, v, 'low_precision')
        tq, tk, tv = prepare_qkv(rq, rk, rv, 'restore_value_output')
        if not (torch.equal(lq, tq) and torch.equal(lk, tk) and torch.equal(tv, rv)):
            raise RuntimeError('Target Q/K matching or V restoration failed')
        for r in restored['trace']:
            expected = 'restore_value_output' if r['layer'] == target else 'low_precision'
            if r['mode'] != expected:
                raise RuntimeError('More than the target layer changed precision mode')
            if r['layer'] == target and r['tensor'] in ('V', 'attention_output') and r['fraction_changed'] != 0:
                raise RuntimeError('Target value/output boundary was quantized')
        restored_score = score_of(restored['result'])
        if not all(torch.isfinite(torch.tensor(s)) and 0 <= s <= 1 for s in (baseline_score, restored_score)):
            raise RuntimeError('Invalid diagnostic score')
        rows.append({'target_layer': target,
             'modes': ['restore_value_output' if i == target else 'low_precision' for i in range(len(layers))],
             'upstream_by_layer': comparisons, 'prefix_and_target_upstream_bit_exact': True,
             'projected_Q_bit_exact': bool(torch.equal(q, rq)), 'projected_K_bit_exact': bool(torch.equal(k, rk)),
             'quantized_Q_bit_exact': bool(torch.equal(lq, tq)), 'quantized_K_bit_exact': bool(torch.equal(lk, tk)),
             'target_V_unquantized': bool(torch.equal(tv, rv)),
             'target_output_after_FP32_projection': tensor_difference(baseline['outputs'][target], restored['outputs'][target]),
             'low_precision_score': baseline_score, 'target_restored_score': restored_score,
             'score_delta_target_minus_lowp': restored_score - baseline_score,
             'trace': restored['trace']})
    unchanged = all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    forwards_restored = all(layer.self_attn.forward == original for layer, original in zip(layers, original_forwards))
    hooks_restored = hook_counts == [(len(layer.self_attn._forward_pre_hooks), len(layer.self_attn._forward_hooks)) for layer in layers]
    if not (unchanged and forwards_restored and hooks_restored):
        raise RuntimeError('Diagnostic failed weight/method/hook preservation')
    return {'description': 'Separate one-target-layer restoration with low-precision prefix AND suffix',
         'local_effect_boundary': 'target attention output after unchanged FP32 output projection',
         'downstream_effect': 'single-clip final-score response with suffix modes fixed lowp; suffix activations may change',
         'not_a_whole_model_restoration_replacement': True, 'not_forensic_accuracy_evidence': True,
         'weights_unchanged': unchanged, 'original_forwards_restored': forwards_restored,
         'hooks_restored': hooks_restored, 'low_precision_trace': baseline['trace'], 'layers': rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--upstream', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    adapter = DiMoDifAdapter(a.upstream, checkpoint_name='lavdf', device=a.device)
    if adapter.checkpoint_sha256 != 'e1e59006d1bb6ad36a3859af5486a6649ed41bba23258b7c41e24f873be7948e':
        raise RuntimeError('Unapproved checkpoint')
    if len(adapter.model.transformer_encoder.layers) != 5:
        raise RuntimeError('Expected the approved five-layer detector')
    features = load_features(a.features)
    official = adapter.forward_with_mode(features, mode='low_precision')
    # forward_reference preserves the original preparation/aggregation. Its model
    # call runs inside diagnostic_intervention, so the attention modes are lowp.
    def forward():
        return adapter.forward_reference(features)
    parity = capture_run(adapter.model, forward)
    if not torch.equal(parity['result']['logits'], official['logits']):
        raise RuntimeError('Diagnostic lowp logits differ from the original adapter')
    result = audit_matched(adapter.model, forward, lambda r: float(r['score_fake'][0]))
    if result['layers'][0]['low_precision_score'] != float(official['score_fake'][0]):
        raise RuntimeError('Diagnostic lowp baseline differs from original adapter')
    result.update({'checkpoint_sha256': adapter.checkpoint_sha256, 'device': str(adapter.device),
                   'feature_file': str(a.features.resolve()), 'official_low_precision_score': float(official['score_fake'][0]),
                   'baseline_matches_original_adapter': True, 'baseline_logits_bit_exact': True})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('Wrote', a.out)


if __name__ == '__main__':
    main()
