import torch

from avfuse.member3_runtime.benchmark import benchmark_callable


def test_cpu_benchmark_has_unavailable_peak_memory():
    x=torch.tensor([2.0]); record=benchmark_callable(lambda: x*x, torch.device('cpu'), warmup=1, repeats=3)
    assert record['repeats']==3 and record['p95_ms']>=0
    assert record['peak_memory_bytes'] is None
