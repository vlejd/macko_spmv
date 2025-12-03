import torch
import macko_spmv

# fmt: off
SAMPLE_MATRIX = torch.tensor([
    [
        0.5498, 0.7124, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.7998, 0.0000, 
        0.0000, 0.0000, 0.1611, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.4663, 
        0.0000, 0.0000
    ],
    [
        0.2241, 0.6528, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 
        0.0000, 0.0000, 0.0000, 0.0000, 0.4761, 0.0000, 0.0000, 0.0000, 0.0000, 0.4673, 
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 
        0.0000, 0.0000
    ],
    [
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 
        0.0000, 0.0166, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 
        0.0000, 0.0000
    ],
    [
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1958, 0.0000, 0.0000, 0.0000, 0.0000, 
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 
        0.0000, 0.0000
    ],
    [
        0.0000, 0.0000, 0.0000, 0.0000, 0.5195, 0.5762, 0.0000, 0.8965, 0.0000, 0.0000, 
        0.0000, 0.0000, 0.0000, 0.5625, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 
        0.0000, 0.0000, 0.0000, 0.2041, 0.0000, 0.0000, 0.8770, 0.0000, 0.1123, 0.4966, 
        0.0000, 0.5132
    ]
], device="cpu", dtype=torch.float16)
# fmt: on

SAMPLE_MATRIX_ROWS = 5
SAMPLE_MATRIX_COLS = 32


def test_comressor_fp16_manual():
    for device in ["cpu", "cuda"]:
        M = SAMPLE_MATRIX.clone().to(device)

        # fmt: off
        expected_values = torch.tensor([
            0.5498, 0.7124, 0.7998, 0.1611, 0.0000, 0.4663, 0.2241, 0.6528, 0.4761,
            0.4673, 0.0000, 0.0166, 0.1958, 0.5195, 0.5762, 0.8965, 0.5625, 0.2041,
            0.8770, 0.1123, 0.4966, 0.5132, 0.0000, 0.0000], dtype=torch.float16, device=device)
        # fmt: on

        expected_deltas = torch.tensor(
            [0, 54, 15, 0, 76, 95, 69, 16, 149, 18, 16, 0],
            dtype=torch.uint8,
            device=device,
        )
        expected_row_indices = torch.tensor(
            [0, 6, 10, 12, 13, 22], device=device, dtype=torch.int32
        )

        compressed_M = macko_spmv.compress(M)
        assert len(compressed_M) == 5
        assert torch.allclose(compressed_M[0], expected_values)
        assert torch.allclose(compressed_M[1], expected_deltas)
        assert torch.allclose(compressed_M[2], expected_row_indices)
        assert compressed_M[3] == SAMPLE_MATRIX_ROWS
        assert compressed_M[4] == SAMPLE_MATRIX_COLS


def test_comressor_fp16_manual_empty():
    for device in ["cpu", "cuda"]:
        M = (
            torch.tensor(
                [[0] * 32, [0] * 31 + [1], [0, 2] + [0] * 30],
                device="cpu",
                dtype=torch.float16,
            )
            .clone()
            .to(device)
        )

        # fmt: off
        expected_values = torch.tensor([0,1,2,0,0,0,0,0], dtype=torch.float16, device=device)
        # fmt: on

        expected_deltas = torch.tensor(
            [255, 1, 0, 0],
            dtype=torch.uint8,
            device=device,
        )
        expected_row_indices = torch.tensor(
            [0, 0, 2, 3], device=device, dtype=torch.int32
        )

        compressed_M = macko_spmv.compress(M)
        assert len(compressed_M) == 5
        assert torch.allclose(compressed_M[0], expected_values)
        assert torch.allclose(compressed_M[1], expected_deltas)
        assert torch.allclose(compressed_M[2], expected_row_indices)
        assert compressed_M[3] == 3
        assert compressed_M[4] == 32


def test_spmv_manual():
    M_cuda = SAMPLE_MATRIX.cuda()
    compressed_M = macko_spmv.compress(SAMPLE_MATRIX.cuda())
    V_cuda = torch.rand((SAMPLE_MATRIX_COLS,), device="cuda", dtype=torch.float16)
    expected_result = M_cuda @ V_cuda
    result = macko_spmv.multiply(compressed_M, V_cuda)
    assert torch.allclose(expected_result, result)


def run_test(M_rows, M_cols, density, seed):
    print(M_rows, M_cols, seed, density)
    torch.manual_seed(seed)
    M_cpu = torch.rand((M_rows, M_cols), dtype=torch.float16, device="cpu")
    M_cpu[torch.rand_like(M_cpu) > density] = 0
    M_cuda = M_cpu.cuda()
    V_cuda = torch.rand((M_cols,), device="cuda", dtype=torch.float16)
    compressed_M = macko_spmv.compress(M_cuda)
    result = macko_spmv.multiply(compressed_M, V_cuda)
    expected_result = M_cuda @ V_cuda
    assert torch.allclose(
        expected_result, result, rtol=0.001, atol=0.001
    ), f"Fail at {seed} {density}"


def test_spmv_random():
    M_rows = 4096
    M_cols = 2 * 4096
    for density in (1.0, 0.5, 0.1, 0.01):
        for seed in range(2):
            run_test(M_rows, M_cols, density, seed)


def test_one_spmv():
    run_test(4096, 2 * 4096, 1.0, 0)


def test_compile():
    M_rows, M_cols, seed, density = 4096, 2 * 4096, 1.0, 0
    torch.manual_seed(seed)
    M_cpu = torch.rand((M_rows, M_cols), dtype=torch.float16, device="cpu")
    M_cpu[torch.rand_like(M_cpu) > density] = 0
    M_cuda = M_cpu.cuda()
    V_cuda = torch.rand((M_cols,), device="cuda", dtype=torch.float16)
    compressed_M = macko_spmv.compress(M_cuda)
    args = (
        compressed_M[0],
        compressed_M[1],
        compressed_M[2],
        compressed_M[3],
        compressed_M[4],
        V_cuda,
    )
    result = macko_spmv.multiply(compressed_M, V_cuda)
    expected_result = M_cuda @ V_cuda
    assert torch.allclose(
        expected_result, result, rtol=0.001, atol=0.001
    ), f"Fail at {seed} {density}"

    print(torch.library.opcheck(torch.ops.macko_spmv.multiply, args))

    compiled = torch.compile(
        macko_spmv.multiply, mode="reduce-overhead", fullgraph=True
    )
    compiled_result = compiled(compressed_M, V_cuda)
    assert torch.allclose(
        expected_result, compiled_result, rtol=0.001, atol=0.001
    ), f"Fail at {seed} {density}"

    f = lambda x: macko_spmv.multiply(compressed_M, x + 1) - 1

    f_c = torch.compile(f, mode="reduce-overhead", fullgraph=True)
    print(f_c(V_cuda))
    print((M_cuda @ (V_cuda + 1)) - 1)
