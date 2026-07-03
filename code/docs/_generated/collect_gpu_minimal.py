from __future__ import annotations
import json, math, os, time
from pathlib import Path
import torch

DEVICE = torch.device('cuda')
torch.manual_seed(42)

SCENARIOS = {
    'torch_gemm': {'elements': 2048, 'iterations': 20},
    'pipeline_fusion': {'elements': 1 << 20, 'iterations': 30},
    'copy_vectorized': {'elements': 1 << 22, 'iterations': 30},
    'kv_block': {'tokens': 64, 'width': 2048, 'iterations': 30},
}
CHAPTERS = {
    'ch01': 'torch_gemm', 'ch02': 'torch_gemm', 'ch03': 'torch_gemm',
    'ch04': 'pipeline_fusion', 'ch05': 'copy_vectorized', 'ch06': 'pipeline_fusion',
    'ch07': 'copy_vectorized', 'ch08': 'pipeline_fusion', 'ch09': 'pipeline_fusion',
    'ch10': 'torch_gemm', 'ch11': 'kv_block', 'ch12': 'kv_block', 'ch13': 'kv_block',
    'ch14': 'torch_gemm', 'ch15': 'kv_block', 'ch16': 'torch_gemm', 'ch17': 'kv_block',
    'ch18': 'kv_block', 'ch19': 'copy_vectorized', 'ch20': 'pipeline_fusion',
}
SUMMARY = {
    'torch_gemm': 'looped small GEMMs -> single batched matmul',
    'pipeline_fusion': 'three separate tensor ops -> one vectorized expression',
    'copy_vectorized': 'chunked copy loop -> single bulk device copy',
    'kv_block': 'per-token KV row update -> block vectorized update',
}

def bench(fn, iterations=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iterations):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return {'mean_ms': sum(times)/len(times), 'min_ms': min(times), 'max_ms': max(times), 'times_ms': times}

def scenario_gemm():
    batch, m, k, n = 64, 32, 64, 32
    a = torch.randn(batch, m, k, device=DEVICE, dtype=torch.float16)
    b = torch.randn(batch, k, n, device=DEVICE, dtype=torch.float16)
    out_loop = torch.empty(batch, m, n, device=DEVICE, dtype=torch.float16)
    out_batch = torch.empty_like(out_loop)
    def baseline():
        for i in range(batch):
            out_loop[i].copy_(a[i] @ b[i])
        return out_loop
    def optimized():
        torch.bmm(a, b, out=out_batch)
        return out_batch
    return baseline, optimized, lambda: float((out_loop - out_batch).abs().max().item())

def scenario_pipeline():
    x = torch.randn(SCENARIOS['pipeline_fusion']['elements'], device=DEVICE)
    t1 = torch.empty_like(x); t2 = torch.empty_like(x); out = torch.empty_like(x); out2 = torch.empty_like(x)
    def baseline():
        torch.mul(x, 1.25, out=t1)
        torch.add(t1, 0.5, out=t2)
        torch.clamp(t2, min=0, out=out)
        return out
    def optimized():
        torch.clamp(x * 1.25 + 0.5, min=0, out=out2)
        return out2
    return baseline, optimized, lambda: float((out - out2).abs().max().item())

def scenario_copy():
    n = SCENARIOS['copy_vectorized']['elements']
    src = torch.randn(n, device=DEVICE)
    dst = torch.empty_like(src); dst2 = torch.empty_like(src)
    chunk = 4096
    def baseline():
        for off in range(0, n, chunk):
            dst[off:off+chunk].copy_(src[off:off+chunk])
        return dst
    def optimized():
        dst2.copy_(src)
        return dst2
    return baseline, optimized, lambda: float((dst - dst2).abs().max().item())

def scenario_kv():
    tokens = SCENARIOS['kv_block']['tokens']; width = SCENARIOS['kv_block']['width']
    kv = torch.randn(tokens, width, device=DEVICE)
    kv2 = kv.clone()
    update = torch.randn(tokens, width, device=DEVICE)
    scale = torch.linspace(0.9, 1.1, tokens, device=DEVICE).view(tokens, 1)
    def baseline():
        for i in range(tokens):
            kv[i].add_(update[i] * scale[i])
        return kv
    def optimized():
        kv2.add_(update * scale)
        return kv2
    return baseline, optimized, lambda: float((kv - kv2).abs().max().item())

BUILDERS = {
    'torch_gemm': scenario_gemm,
    'pipeline_fusion': scenario_pipeline,
    'copy_vectorized': scenario_copy,
    'kv_block': scenario_kv,
}

def run_scenario(name):
    baseline, optimized, err = BUILDERS[name]()
    # establish correctness after one pair before timing mutating scenarios
    baseline(); optimized(); torch.cuda.synchronize()
    max_err = err()
    # rebuild to reset mutating state for timing
    baseline, optimized, err = BUILDERS[name]()
    iterations = SCENARIOS[name].get('iterations', 20)
    b = bench(baseline, iterations=iterations)
    o = bench(optimized, iterations=iterations)
    baseline(); optimized(); torch.cuda.synchronize()
    max_err = err()
    return {'summary': SUMMARY[name], 'baseline': b, 'optimized': o, 'speedup': b['mean_ms']/o['mean_ms'] if o['mean_ms'] else 0.0, 'max_abs_err': max_err}

def main():
    results = {
        'host': {
            'torch': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'gpu': torch.cuda.get_device_name(0),
            'device_count': torch.cuda.device_count(),
            'note': 'ssh mi direct PyTorch CUDA minimal microbench; RTX 2060 non-canonical, clocks not locked',
        },
        'scenarios': {},
        'chapters': {},
    }
    for name in BUILDERS:
        torch.cuda.empty_cache()
        result = run_scenario(name)
        results['scenarios'][name] = result
        print(name, '{:.6f}'.format(result['baseline']['mean_ms']), '{:.6f}'.format(result['optimized']['mean_ms']), '{:.3f}x'.format(result['speedup']), 'err', result['max_abs_err'], flush=True)
    for ch, scenario in CHAPTERS.items():
        r = results['scenarios'][scenario]
        results['chapters'][ch] = {'scenario': scenario, 'summary': r['summary'], 'baseline_ms': r['baseline']['mean_ms'], 'optimized_ms': r['optimized']['mean_ms'], 'speedup': r['speedup'], 'max_abs_err': r['max_abs_err']}
    out = Path('docs/_generated/gpu_minimal_rtx2060_metrics.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print('WROTE', out)

if __name__ == '__main__':
    main()
