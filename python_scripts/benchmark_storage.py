import macko_spmv
import torch
import numpy as np
import fire
import tqdm

def get_tensor_size_bits(t):
    return t.numel() * t.element_size() * 8


def benchmark(M_rows, M_cols, density, seed, rounds=1000):
    # Make data
    torch.manual_seed(seed)

    effective_densities = []
    for _ in tqdm.trange(rounds):
        M_cpu = torch.rand((M_rows, M_cols), dtype=torch.float16, device="cpu")
        M_cpu[torch.rand_like(M_cpu) > density] = 0
        compressed_M = macko_spmv.compress(M_cpu)

        compressed_size = (
            get_tensor_size_bits(compressed_M[0])
            + get_tensor_size_bits(compressed_M[1])
            + get_tensor_size_bits(compressed_M[2])
            + 64  # (matrix size)
        )

        effective_densities.append((compressed_size) / (M_rows * M_cols * 16))

    eff_d = np.array(effective_densities)

    print(f"Density: {density}")
    print(f"Sample real density: {eff_d[0]}")
    print(f"Effective density random mean +-std: {np.mean(eff_d)} +- {np.std(eff_d)}")
    print(f"Effective density random min max: {np.min(eff_d)} {np.max(eff_d)}")
    print(
        f"Effective density random 5-95 percentile: {np.percentile(eff_d, 5)} {np.percentile(eff_d, 95)}"
    )


if __name__ == "__main__":
    fire.Fire(benchmark)
