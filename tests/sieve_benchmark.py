import pyoptgemm
import torch
import argparse

BATCH_SIZE = 2048

def runtime(f, runs=100, warmup=10):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for _ in range(warmup):
        f()
    torch.cuda.synchronize()
    
    start_event.record()
    for _ in range(runs):
        f()
    end_event.record()
    torch.cuda.synchronize()

    return start_event.elapsed_time(end_event) / runs  # milliseconds per op

def bench_single_op(m, n, k, backend="torch"):
    if m == 0 or n == 0 or k == 0:
        return 0.0

    a = torch.randn((m, k), dtype=torch.float16, device='cuda')
    b = torch.randn((k, n), dtype=torch.float16, device='cuda')

    if backend == "torch":
        return runtime(lambda: a @ b)
    elif backend == "pyoptgemm":
        return runtime(lambda: pyoptgemm.mm(a, b))
    else:
        raise ValueError("Invalid backend selected")

matrix_configs = {
    "Phi-3-mini": {
        "w1": (16384, 3072), "w1-A": (0, 0), "w2": (3072, 8192), "w2-A": (0, 0),
        "wqkv": (9216, 3072), "wqkv-A": (0, 0), "wo": (3072, 3072), "wo-A": (0, 0)
    },
    "Phi-3-mini-C-23%": {
        "w1": (16384, 1792), "w1-A": (3072, 1792), "w2": (1792, 8192), "w2-A": (1792, 3072),
        "wqkv": (9216, 1792), "wqkv-A": (3072, 1792), "wo": (3072, 3072), "wo-A": (0, 0)
    },
    "Phi-3-mini-C-35%": {
        "w1": (16384, 1536), "w1-A": (3072, 1536), "w2": (1536, 8192), "w2-A": (1536, 3072),
        "wqkv": (9216, 1536), "wqkv-A": (3072, 1536), "wo": (3072, 3072), "wo-A": (0, 0)
    },
    "Phi-3-mini-C-60%": {
        "w1": (16384, 896), "w1-A": (3072, 896), "w2": (896, 8192), "w2-A": (896, 3072),
        "wqkv": (9216, 896), "wqkv-A": (3072, 896), "wo": (3072, 896), "wo-A": (3072, 896)
    },
    "LLaMA-3.1-8B": {
        "w1": (28672, 4096), "w1-A": (0, 0), "w2": (4096, 14336), "w2-A": (0, 0),
        "wqkv": (6144, 4096), "wqkv-A": (0, 0), "wo": (4096, 4096), "wo-A": (0, 0)
    },
    "LLaMA-3.1-8B-C-26%": {
        "w1": (28672, 2304), "w1-A": (4096, 2304), "w2": (2304, 14336), "w2-A": (2304, 4096),
        "wqkv": (6144, 2304), "wqkv-A": (4096, 2304), "wo": (4096, 4096), "wo-A": (0, 0)
    },
    "LLaMA-3.1-8B-C-39%": {
        "w1": (28672, 1920), "w1-A": (4096, 1920), "w2": (1920, 14336), "w2-A": (1920, 4096),
        "wqkv": (6144, 1920), "wqkv-A": (4096, 1920), "wo": (4096, 4096), "wo-A": (0, 0)
    },
    "LLaMA-3.1-8B-C-60%": {
        "w1": (28672, 1280), "w1-A": (4096, 1280), "w2": (1280, 14336), "w2-A": (1280, 4096),
        "wqkv": (6144, 1280), "wqkv-A": (4096, 1280), "wo": (4096, 1280), "wo-A": (4096, 1280)
    },
    "LLaMA-3.1-70B": {
        "w1": (57344, 8192), "w1-A": (0, 0), "w2": (8192, 28672), "w2-A": (0, 0),
        "wqkv": (10240, 8192), "wqkv-A": (0, 0), "wo": (8192, 8192), "wo-A": (0, 0)
    },
    "LLaMA-3.1-70B-C-41%": {
        "w1": (57344, 3712), "w1-A": (8192, 3712), "w2": (3712, 28672), "w2-A": (3712, 8192),
        "wqkv": (10240, 3712), "wqkv-A": (8192, 3712), "wo": (8192, 8192), "wo-A": (0, 0)
    },
    "LLaMA-3.1-70B-C-56%": {
        "w1": (57344, 2688), "w1-A": (8192, 2688), "w2": (2688, 28672), "w2-A": (2688, 8192),
        "wqkv": (10240, 2688), "wqkv-A": (8192, 2688), "wo": (8192, 2688), "wo-A": (8192, 2688)
    },
    "LLaMA-3.1-70B-C-75%": {
        "w1": (57344, 1536), "w1-A": (8192, 1536), "w2": (1536, 28672), "w2-A": (1536, 8192),
        "wqkv": (10240, 1536), "wqkv-A": (8192, 1536), "wo": (8192, 1536), "wo-A": (8192, 1536)
    }
}

def benchmark(backend='pyoptgemm'):
    results = {}

    for model_name, matrices in matrix_configs.items():
        print(f"\n=== {model_name} ===")
        times = {}

        for mat_name, (rows, cols) in matrices.items():
            if rows == 0 or cols == 0:
                continue

            m, n, k = (rows, BATCH_SIZE, cols) if "-A" not in mat_name else (BATCH_SIZE, cols, rows)
            op_time = bench_single_op(m, n, k, backend=backend)
            times[mat_name] = op_time

            print(f"{mat_name:<8}: {op_time:.3f} ms")

        results[model_name] = times

    # Compute and output speed-ups for compressed models
    for model_name, matrices in matrix_configs.items():
        if "-C-" not in model_name:
            continue  # skip original models

        print(f"\n--- Speed-up analysis for {model_name} ---")
        
        # Extract original model name
        orig_model_prefix = model_name.split('-C-')[0]
        orig_times = results[orig_model_prefix]
        compressed_times = results[model_name]

        total_orig_time = 0
        total_compressed_time = 0
        for mat in ["w1", "w2", "wqkv", "wo"]:
            orig_time = orig_times[mat.split("-")[0]]
            total_orig_time += orig_time
            adapter_mat = mat + "-A"
            mat_time = compressed_times.get(mat, 0.0)
            adapter_time = compressed_times.get(adapter_mat, 0.0)

            compressed_time = mat_time + adapter_time
            total_compressed_time += compressed_time
            speedup = orig_time / compressed_time if compressed_time > 0 else float('inf')
            print(f"{mat}: original={orig_time:.3f} ms, compressed+adapter={compressed_time:.3f} ms, speed-up={speedup:.2f}x")
       
        total_speedup = total_orig_time / total_compressed_time if total_compressed_time > 0 else float('inf')
        print(f"Total: original={total_orig_time:.3f} ms, compressed+adapter={total_compressed_time:.3f} ms, speed-up={total_speedup:.2f}x")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=['torch', 'pyoptgemm', 'both'], default='pyoptgemm')
    args = parser.parse_args()

    if args.backend in ['torch', 'pyoptgemm']:
        benchmark(backend=args.backend)
    else:
        print("Benchmarking with torch backend:")
        benchmark(backend='torch')
        print("\n==============================\n")
        benchmark(backend='pyoptgemm')
