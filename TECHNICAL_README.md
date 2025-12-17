# Instructions for End2End model inference

These are instructions for Llama-2-7b-hf model pruning to sparsity 60%, density 40%. 
Be carefull, MACKO uses density, wanda pruning uses sparsity.

This is incompatible with `transformers>=4.56.0`

- setup environment
    - `curl -LsSf https://astral.sh/uv/install.sh | sh`
    - `exec bash`
    - `hf auth login`

- prune model  (4 min per pruning)
    - `git clone https://github.com/vlejd/wanda-modern.git`
        - original repo is `https://github.com/locuslab/wanda.git` but it contains data loading bug (PR with fix was opened 1 year ago)
    - `cd wanda-modern`
    - `uv venv; uv sync`
    - `source .venv/bin/activate`
    - `mkdir -p /workspace/pruned_models/`
    - `mkdir -p out/`
    -  `python main.py --model meta-llama/Llama-2-7b-hf --prune_method wanda --sparsity_ratio 0.6 --sparsity_type unstructured --save out/ --save_model /workspace/pruned_models/llama_7b_unstructured_wanda_density_0.4`


- Get this repo
    - Clone the repo, or copy paste the archive `git archive --format=zip --output ../macko_spmv.zip master`
    - `uv venv; uv sync --all-groups`
    - `source .venv/bin/activate`
    - test inside python directory: `pytest test_macko_spmv.py`

- compress model (4 min)
    - `python python_scripts/util_compress_llm.py /workspace/pruned_models/llama_7b_unstructured_wanda_density_0.4/`

- benchmark (un)compressed model
    - `python python_scripts/util_run_pruned_llm.py /workspace/pruned_models/llama_7b_unstructured_wanda_density_0.4 /workspace/pruned_models/llama_7b_unstructured_wanda_density_0.4_compressed --make_sparse=1 | tee run_d0.4_sparse.txt`


# Useful commands

- Compile the benchmarking kernel (you can also add  `-arch=sm_<your gpu>`): `nvcc -O3   benchmark.cu -o benchmark -lcublas -lcusparse --resource-usage`
- Profile code: `sudo /usr/local/cuda/bin/ncu -f --set full -o profiles/benchmark ./benchmark 4096 4096 0 0 1 0 7 47 0.5`
- Set GPU compute frequency: `sudo nvidia-smi -i 0 -lgc 1800,1800`
- Reset GPU compute frequency: `sudo nvidia-smi -i 0 -rgc`
- Query real time GPU stats: `nvidia-smi --query-gpu=index,timestamp,power.draw,clocks.sm,clocks.mem,clocks.gr,memory.used --format=csv -l 1`
- Get gpu name: `nvidia-smi --query-gpu=name --format=csv,noheader | tr ' ' '_'`


# Full SpMV evaluation

- `git clone https://github.com/vlejd/macko_spmv.git ; cd macko_spmv/c_benchmarking`
- `nvcc -O3 benchmark.cu -o benchmark -gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -lcublas -lcusparse --resource-usage`
- Test one size: `./benchmark 16 4096 4096 100 1000 1 0 7 47 0.5`
- Run all tests: `bash run_tests_fp16.sh`

# SpMV Baselines

**Dasp**
- `git clone https://github.com/vlejd/DASP.git ; cd DASP`
- Compile: `nvcc -O3 src/main_f16.cu -o spmv_half -gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -lineinfo -lcusparse -lcublas`
- Test one size: `./spmv_half 4096 4096 0.5`
- Run all tests: `bash run_tests.sh`
- Results are in `dasp_results_16bit` directory

**Sputnik**
- `git clone https://github.com/vlejd/SpInfer.git; cd SpInfer`

- Install (a little more involved) 
    ```bash
    git submodule update --init --recursive
    source Init_SpInfer.sh
    cd $SpInfer_HOME/third_party/sputnik && git apply ../sputnik.patch
    cd $SpInfer_HOME/build && make -j
    cd $SpInfer_HOME/third_party/
    source build_sputnik.sh

    cd $SpInfer_HOME/third_party/
    source preparse_cusparselt.sh

    cd $SpInfer_HOME/kernel_benchmark
    source test_env
    make -j spmm_test_sputnik
    ```
- Test one size: `./spmm_test_sputnik 4096 4096 1 50 1`
- Run all tests
    ```bash
    cd $SpInfer_HOME/kernel_benchmark
    bash run_sputnik.sh
    ```