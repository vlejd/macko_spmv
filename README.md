# Overview

MACKO-SpMV: **M**utually **A**ligned **C**ompressed coordinates **K**ernel **O**ptimised **Sp**arse **M**atrix **V**ector multiplication is a new format for representing sparse matrices and a cuda kernel for efficient matrix vector multiplication using this format.
It is targeted at sparsities between 20-90%, commonly seen in Neural Network pruning.

These sparsities were historically very hard to make use of in practice, because existing formats like CSR are not optimized for this range.
We hope this library will help to spark more interest in the field of neural network pruning and find uses outside it as well.
Debate whether quantization is better than pruning is beyond the scope of this work, but we hope we will help to bridge the gap.

You can find more information on [our blog](https://grizzlytech.dev/blog/macko-spmv).
For even more technical information see our [paper](https://arxiv.org/pdf/2511.13061).


# Performance

Performance depends on what GPU you are using, size of the matrix and the density of the matrix.
Tested on NVIDIA: 2080, 3090, 4090.
In general, you should expect following results for consumer GPUs.

fp16 values
- 50% matrix density: 1.5x memory reduction, 1.3-1.5x speedup over cuBLAS
- 10% matrix density: 6.25x memory reduction, 3.5-4.5x speedup over cuBLAS
- Faster with smaller memory footprint compared to CSR format for all densities above 10%.

This translates directly to End2End LLM inference.
For Llama 2-7b model in fp16 pruned with wanda in unstructured mode
- 50% density: Memory goes from 13.59GB to 8.87GB, tokens/sec from 66.53 to 98.60
- 10% density: Memory goes from 13.59GB to 2.67GB, tokens/sec from 66.53 to 255.01

You can find detailed benchmarks in [our blog](https://grizzlytech.dev/blog/macko-spmv).
See `media` directory for extensive number of graphs.
MACKO format works across all GPUs and the memory reduction is the same.
However, the SpMV algorithm is not tuned yet and the performance may vary across GPUS.
A special type of optimization is needed for server GPUs (H100, V100).

# Installation

This repo uses python 3.12, but should work across most versions.
Normally we use `uv`, but you can use whatever virtual env you want.

Basic usage
```bash
git clone https://github.com/vlejd/macko_spmv.git
cd macko_spmv
pip install .
```

If you want to run tests:

```bash
pip install '.[test]'
```

If you want to run end2end LLM stuff or make some graphs:

```bash
pip install '.[dev]'
```

If you use `uv`
```bash
uv sync --extra dev --extra test
uv run pytest
```


# Usage

Call `macko_spmv.compress` to compress your sparse matrix and `macko_spmv.multiply` to perform SpMV.

```python
import macko_spmv
import torch

# Make some random matrix M and random vector V
M = torch.rand((8192, 8192), dtype=torch.float16, device="cuda")
V = torch.rand((8192,), dtype=torch.float16, device="cuda")

# Make The matrix sparse
M[M<0.5] = 0.

# Compress it to MACKO format
compressed_M = macko_spmv.compress(M)

# Run it
print("MACKO output:", macko_spmv.multiply(compressed_M, V))
print("Torch output:", M@V)

# Profile it
for _ in range(10): # Torch needs a little warmup
    M @ V
torch.cuda.synchronize()

with torch.autograd.profiler.profile(use_device = 'cuda') as prof:
    for _ in range(100):
        y = M@V

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


with torch.autograd.profiler.profile(use_device = 'cuda') as prof:
    for _ in range(100):
        y = macko_spmv.multiply(compressed_M, V)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

Run this and you should see clear speedup on all consumer GPUs. 

Note: this library compiles cuda kernels on first import. The first run may take a while.

Compatible with torch tensors and torch compilation.
See `tests/test_macko_spmv.py` for more advanced usage.


# Setup

You can directly install this repository using `pip` thanks to `pyproject.toml`.

This library compiles it's cuda kernels on the fly using `torch.load_inline`.
Including this library for the first time can take a minute.
To prevent recompilation, set `MACKO_SPMV_BUILD_DIRECTORY` environment variable.


# Instructions for End2End model

This process is more involved, please refer to [technical readme](TECHNICAL_README.md).


# Citing

If you use this library, please cite the following paper https://arxiv.org/pdf/2511.13061 (bibtex soon).


We will also appreciate some kind words about it to github issues, hacker news, Yannic Kilcher, two minute papers, discord, linkedin, x, tikto, instagram, mastodont, reddit, 4chan, your investors newsletter, your loved ones or whoever you think would find it interesting.

We are a very small team, every good promo helps :).


# Authors

- Vladimír Macko: GrizzlyTech.dev; Comenius University, Bratislava, Slovakia, 
- Vladimír Boža: Comenius University, Bratislava, Slovakia


# Acknowledgement

This work would not be possible without the following resources:
- GPU MODE community, https://discord.gg/pqm9sJgD . This is hands down the best set of resources regarding GPU programming.
- Simon Boehm with his amazing blog: https://siboehm.com/articles/22/CUDA-MMM . This is a must read if you want to do anything regarding matrix multiplication.
- Lei Mao and all his blogs, especially: https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/
- Very nice visual blog about cuda benchmarking: https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch
- Another very nice blog about cuda benchmarking: https://guillesanbri.com/CUDA-Benchmarks/
- [Vast.ai](https://vast.ai/): the most convenient way to just get a random gpu for a few hours and develop some fast kernels. You are one `ncu` support away from perfection.


# Contributions

Contributions are welcome!

Current challenges:

- Support more data types (bfloat16, fp8, int8, float32) in a maintainable way
- Torch batch routing
- Installation / build system optimization
- Profiling on server GPUs
- Benchmarking other models & other pruning algorithms
- Further kernel optimization

If you want to collaborate and can write GEMV that matches/beats cuBLAS on H100 GPU,
or if you have access to H100 with enabled profiling, please reach out.

Discord: [@vlejd](https://discord.com/users/444267838140579840/)
x.com: [@vlejd](https://x.com/vlejd) and [@bozavlado](https://x.com/bozavlado)
