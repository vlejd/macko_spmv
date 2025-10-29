import macko_spmv
import torch
import numpy as np
import time
import fire


def print_results(times_us, tag):
    times_us = np.array(times_us)
    print()
    print(tag)
    print(f"Mean +- std [us]: {np.mean(times_us):.1f} +- {np.std(times_us):.1f}")
    print(f"Min - max [us]: {np.min(times_us):.1f} {np.max(times_us):.1f}")
    print(
        f"5-95 percentile [us]: {np.percentile(times_us, 5):.1f} {np.percentile(times_us, 95):.1f}"
    )


def benchmark_launch_time_f(f, args, warmups, kernel_launch_runs, tag):
    for _ in range(warmups):
        f(*args)
        torch.cuda.synchronize()

    kernel_laun_times_us = []
    for _ in range(kernel_launch_runs):
        start_ns = time.time_ns()
        f(*args)
        end_ns = time.time_ns()
        kernel_laun_times_us.append((end_ns - start_ns) / 1000)
        torch.cuda.synchronize()

    print_results(kernel_laun_times_us, tag)


def benchmark_f(f, args, warmups, computation_runs, cache_size_mb, tag):

    flush_cache = cache_size_mb is not None
    if flush_cache:
        cache = torch.zeros(cache_size_mb * 1000000, device="cuda", dtype=torch.uint8)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(warmups):
        f(*args)

    torch.cuda.synchronize()

    times_us = []
    for _ in range(computation_runs):
        if flush_cache:
            cache += 1
        start.record()
        f(*args)
        end.record()
        torch.cuda.synchronize()
        times_us.append(start.elapsed_time(end) * 1000)

    print_results(times_us, tag)


def benchmark(
    M_rows,
    M_cols,
    density,
    seed,
    warmup_rounds=100,
    timing_rounds=1000,
    cache_size=4,
    bench_csc=False,
):
    # Make data
    torch.manual_seed(seed)
    M_cpu = torch.rand((M_rows, M_cols), dtype=torch.float16, device="cpu") *2 -1
    M_cpu[torch.rand_like(M_cpu) > density] = 0
    M_cuda = M_cpu.cuda()
    V_cuda = torch.rand((M_cols,), device="cuda", dtype=torch.float16) *2 -1
    compressed_M = macko_spmv.compress(M_cuda)

    # Sparse
    M_csr = M_cuda.to_sparse_csr()
    V_cuda_2d = V_cuda.reshape(-1, 1)

    # Sheck for correctness
    expected_result = M_cuda @ V_cuda
    result = macko_spmv.multiply(compressed_M, V_cuda)    
    assert torch.allclose(
        expected_result, result, rtol=0.01, atol=0.01
    ), f"Fail at {seed} {density}"

    result = torch.sparse.mm(M_csr, V_cuda_2d).flatten()

    assert torch.allclose(
        expected_result, result, rtol=0.1, atol=0.1
    ), f"Fail at {seed} {density}"

    real_density = (M_cuda != 0).sum() / (M_rows * M_cols)
    print(f"Real density: {real_density:.3f}")

    # benchmark
    benchmark_f(
        torch.matmul,
        (M_cuda, V_cuda),
        warmup_rounds,
        timing_rounds,
        cache_size,
        "torch",
    )
    if bench_csc:

        benchmark_f(
            torch.sparse.mm,
            (M_csr, V_cuda_2d),
            warmup_rounds,
            timing_rounds,
            cache_size,
            "CSR",
        )

    benchmark_f(
        macko_spmv.multiply,
        (compressed_M, V_cuda),
        warmup_rounds,
        timing_rounds,
        cache_size,
        "macko_spmv",
    )

    benchmark_launch_time_f(
        torch.matmul,
        (M_cuda, V_cuda),
        warmup_rounds,
        timing_rounds,
        "torch launch time",
    )
    benchmark_launch_time_f(
        macko_spmv.multiply,
        (compressed_M, V_cuda),
        warmup_rounds,
        timing_rounds,
        "macko_spmv launch time",
    )


if __name__ == "__main__":
    fire.Fire(benchmark)
