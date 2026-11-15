"""G6-only reference/lowp CPU operator timing of the original Day A capture."""
import argparse
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--g5-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    sys.path[:0] = [str(a.g5_root / 'src'), str(a.g5_root / 'g5')]
    import torch
    from common import load, save, digest, CHECKPOINT
    from avfuse.member3_runtime.operator_benchmark import benchmark_attention_operator
    data = Path(__file__).resolve().parents[2] / 'artifacts/dayB/local_deliverables/g6_colab/AFML_G6_CPU_Addon/dayA'
    handoff = load(data / 'operator_handoff.json')
    if handoff['passed'] is not True or handoff['checkpoint_sha256'] != CHECKPOINT:
        raise RuntimeError('Invalid original Day A provenance')
    saved = torch.load(data / 'layer0_qkv.pt', map_location='cpu', weights_only=True)
    q, k, v = [saved[name] for name in ('q', 'k', 'v')]
    for tensor in (q, k, v):
        if list(tensor.shape) != handoff['q_shape'] or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
            raise RuntimeError('Day A Q/K/V shape/dtype/finiteness mismatch')
    results = [benchmark_attention_operator(q, k, v, mode=mode, backend=backend, device='cpu',
        input_kind='real', shape_source='Original Day A real pilot capture; layer0_qkv.pt SHA256=' + digest(data / 'layer0_qkv.pt'),
        attn_mask=saved.get('attn_mask'), warmup=10, repeats=50,
        measure_memory=True, require_memory=False, workload_name='DayA_layer0_B1_H4_N1202_D16')
        for backend in ('sdpa', 'dense_oracle') for mode in ('reference', 'low_precision')]
    save(a.out, {'checkpoint_sha256': CHECKPOINT, 'qkv_sha256': digest(data / 'layer0_qkv.pt'),
        'provenance': handoff, 'scope': 'isolated_attention_operator',
        'precision_claim': 'simulated INT8, no integer kernels', 'results': results})


if __name__ == '__main__':
    main()
