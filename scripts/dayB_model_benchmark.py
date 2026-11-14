"""Day B real cached-feature model benchmark; no media decoding or test access.

The benchmark_callable includes precision conversions performed in the model.
One source NPZ is benchmarked per invocation; keep workloads matched across modes.
"""
import argparse
from pathlib import Path
import sys

import torch

# Make the scripts package importable when run directly by file path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from avfuse.member1_fusion.dimodif_adapter import DiMoDifAdapter
from avfuse.member3_runtime.artifacts import (
    build_benchmark_record,
    environment_block,
    validate_benchmark_records,
    write_benchmark_records,
)
from avfuse.member3_runtime.benchmark import benchmark_callable
from avfuse.member3_runtime.environment import collect_environment_report
from avfuse.member3_runtime.provenance import hash_checkpoint, hash_feature_cache
from avfuse.shared.contracts import PRECISION_MODES
from scripts.dayB_inference import load_features

EXPECTED_SHA = 'e1e59006d1bb6ad36a3859af5486a6649ed41bba23258b7c41e24f873be7948e'


def build_model_benchmark_records(
    adapter, features, source_path: Path, *, device: str, modes,
    warmup: int, repeats: int, run_id: str, experiment_id: str,
):
    source_path = Path(source_path)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if adapter.checkpoint_sha256 != EXPECTED_SHA:
        raise RuntimeError('Benchmark checkpoint SHA mismatch')
    if adapter.model.training or any(p.requires_grad for p in adapter.model.parameters()):
        raise RuntimeError('Model must be eval and fully frozen')

    # Do not benchmark host-to-device copies. Inputs are resident on the device.
    dev = torch.device(device)
    inputs = {key: tensor.detach().to(dev) for key, tensor in features.items()}
    shapes = {key: list(tensor.shape) for key, tensor in inputs.items()}
    report = collect_environment_report()
    environment = environment_block(report, str(dev), git=report.get('git'))
    checkpoint = hash_checkpoint(adapter.checkpoint_path)
    # Hash this ONE actual NPZ cache directory, not the full 4-GB dataset.
    identity = hash_feature_cache(source_path.parent)
    records = []
    for mode in modes:
        if mode not in PRECISION_MODES:
            raise ValueError(f'Unknown precision configuration: {mode}')
        measurement = benchmark_callable(
            lambda mode=mode: adapter.predict_from_cached_features(inputs, mode),
            device=dev, warmup=warmup, repeats=repeats,
            synchronize=True, measure_memory=True,
        )
        record = build_benchmark_record(
            measurement,
            run_id=run_id,
            experiment_id=experiment_id,
            config=mode,
            scope='cached_feature_model',
            backend='sdpa',
            batch_size=1,
            execution_dtype=torch.float32,
            input_boundary='cached_encoder_tokens',
            input_shapes=shapes,
            environment=environment,
            checkpoint=checkpoint,
            input_identity=identity,
            conversion_overhead_ms=None,
            # Total mode latency includes conversions, but isolated conversion-only
            # milliseconds have not been measured: leave that separate field null.
            conversion_overhead_included_in_latency=(mode != 'reference'),
            notes=(
                'Cached encoder tokens already placed on the benchmark device; '
                'includes model preparation/padding, attention and detection head. '
                'Any fake INT8 quantize/dequantize executes inside the timed model. '
                'No raw-media decoding or true INT8 kernel is timed.'
            ),
        )
        records.append(record)
    validate_benchmark_records(records, run_id=run_id)
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--upstream', default='third_party/dimodif')
    p.add_argument('--device', default='cpu')
    p.add_argument('--modes', nargs='+', choices=PRECISION_MODES,
                   default=list(PRECISION_MODES))
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--repeats', type=int, default=50)
    p.add_argument('--run-id', required=True)
    p.add_argument('--experiment-id', default='dayB-cached-feature-model')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    adapter = DiMoDifAdapter(args.upstream, checkpoint_name='lavdf', device=args.device)
    features = load_features(args.features)
    records = build_model_benchmark_records(
        adapter, features, args.features,
        device=args.device, modes=args.modes,
        warmup=args.warmup, repeats=args.repeats,
        run_id=args.run_id, experiment_id=args.experiment_id,
    )
    result = write_benchmark_records(
        records, args.out.parent, run_id=args.run_id,
        filename=args.out.name, overwrite=False,
    )
    for r in records:
        print(r['config'], r['p50_ms'], r['p95_ms'], r['peak_memory_bytes'])
    print('Wrote', result['record_count'], 'schema-validated records:', args.out)


if __name__ == '__main__':
    main()
