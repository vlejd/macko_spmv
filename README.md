# Overview

MACKO-SpMV: **M**utually **A**ligned **C**ompressed coordinates **K**ernel **O**ptimised **Sp**arse **M**atrix **V**ector multiplication is a new format for representing sparse matrices and a cuda kernel for efficient matrix vector multiplication using this format.
It is targeted at sparsities between 20-90\%, commonly seen in Neural Network pruning.

These sparsities were historically very hard to make use of in practice, because existing formats like CSR are not optimized for this range.
We hope this library will help to spark more interest in the field of neural network pruning and find uses outside it as well.
Debate whether quantization is better than pruning is beyond the scope of this work, but we hope we will help to bridge the gap.

You can find more information in [out writeup](media/README.md).
For even more technical information see our paper (TODO coming soon).


# Usage

Call `macko_spmv.compress` to compress your sparse matrix and `macko_spmv.multiply` to perform SpMV.

```
import macko_spmv
import torch

# Make some random matrix M and random vector V
M = torch.rand((4096, 8192), dtype=torch.float16, device="cuda")
V = torch.rand((8192,), dtype=torch.float16, device="cuda")

# Make The matrix sparse
M[M<0.5] = 0.

# Compress it to MACKO format
compressed_M = macko_spmv.compress(M)

Y = macko_spmv.multiply(compressed_M, V)

print(Y)
print(M@Y)
```

Note: this library compiles cuda kernels on first import. The first run may take a while.

Compatible with torch tensors and torch compilation.
See `tests/test_macko_spmv.py` for more advanced usage.


# Performance

Performance depends on what GPU you are using, size of the matrix and the density of the matrix.
In general, you should expect following results for consumer GPUs (tested on NVIDIA 2080, 3090, 4090).

fp16 values
- 50% matrix density: 1.5x memory reduction, 1.3-1.5x speedup over cuBLAS
- 10% matrix density: 6.25x memory reduction, 3.5-4.5x speedup over cuBLAS
- Faster with smaller memory footprint compared to CSR format for all densities above 10\%.

This translates directly to End2End LLM inference improvements.

For Llama 2-7b model in fp16 pruned with wanda in unstructured mode
- 50% density: Memory goes from 13.59GB to 8.87GB, tokens/sec from 66.53 to 98.60
- 10% density: Memory goes from 13.59GB to 2.67GB, tokens/sec from 66.53 to 255.01

You can find detailed benchmarks in [our writeup](media/README.md).
MACKO format works across all GPUs and the memory reduction is the same.
However, the SpMV algorithm is not tuned yet, and the performance may vary across GPUS.
A special type of optimization is needed for server GPUs (H100, V100).


# Setup

You can directly install this repository using `pip` thanks to `pyproject.toml`.

This library compiles it's cuda kernels on the fly using `torch.load_inline`.
Including this library for the first time can take a minute.
To prevent recompilation, set `MACKO_SPMV_BUILD_DIRECTORY` environment variable.


# Useful commands

- Compile the benchmarking kernel (you can also add  `-arch=sm_<your gpu>`): `nvcc -O3   benchmark.cu -o benchmark -lcublas -lcusparse --resource-usage`
- Profile code: `sudo /usr/local/cuda/bin/ncu -f --set full -o profiles/benchmark ./benchmark 4096 4096 0 0 1 0 7 47 0.5`
- Set GPU compute frequency: `sudo nvidia-smi -i 0 -lgc 1800,1800`
- Reset GPU compute frequency: `sudo nvidia-smi -i 0 -rgc`
- Query real time GPU stats: `nvidia-smi --query-gpu=index,timestamp,power.draw,clocks.sm,clocks.mem,clocks.gr,memory.used --format=csv -l 1`
- Get gpu name: `nvidia-smi --query-gpu=name --format=csv,noheader | tr ' ' '_'`


# Instructions for End2End model

This process is more involved, please refer to [technical readme](TECHNICAL_README.md).


# Citing

If you use this library, please cite the following paper:
__ TODO Paper coming soon__.


# Authors

- Vladimir Macko: GrizzlyTech.dev; Comenius University, Bratislava, Slovakia, 
- Vladimír Boža: Comenius University, Bratislava, Slovakia


# Contributions

Contributions are welcome!

Current challenges

- Support more data types (bfloat16, fp8)
- Torch batch routing
- Installation / build system optimization
- Profiling on server GPUs
- Benchmarking other models & other pruning algorithms
- Further kernel optimization

If you want to collaborate and can write GEMV that matches/beats cuBLAS on H100 GPU,
or if you have access to H100 with enabled profiling, please reach out.
Discord: @vlejd
