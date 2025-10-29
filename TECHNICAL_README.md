# Instructions for End2End model inference

These are instructions for Llama-2-7b-hf model pruning to sparsity 60\%, density 40\%. 
Be carefull, MACKO uses density, wanda pruning uses sparsity.

This is incompatible with `transformers>=4.56.0`

- setup environment
    - `curl -LsSf https://astral.sh/uv/install.sh | sh`
    - `exec bash`
    - `hf auth login`

- prune model  (4 min per pruning)
    - `git clone ` TODO fixed wanda repo 
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

- compress model (4 min for 20 workers)
    - `python python_scripts/util_compress_llm.py /workspace/pruned_models/llama_7b_unstructured_wanda_density_0.4/ 20`

- benchmark (un)compressed model
    - `python python_scripts/util_run_pruned_llm.py /workspace/pruned_models/llama_7b_unstructured_wanda_density_0.4 /workspace/pruned_models/llama_7b_unstructured_wanda_density_0.4_compressed --make_sparse=1 | tee run_d0.4_sparse.txt`
