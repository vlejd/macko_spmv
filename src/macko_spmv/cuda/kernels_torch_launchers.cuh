#include <ATen/cuda/CUDAContext.h>

at::Tensor macko_spmv_launcher(
    at::Tensor M_values,
    at::Tensor M_deltas,
    at::Tensor M_row_indices,
    int64_t M_rows, int64_t M_cols,
    at::Tensor V)
{
    TORCH_CHECK(M_values.dtype() == at::kHalf, "M must be torch.half");
    TORCH_CHECK(V.dtype() == at::kHalf, "V must be torch.half");

    auto C = at::empty({M_rows}, M_values.options());

    dim3 grid_size(UPDIV(M_rows, BLOCK_SIZE));
    dim3 block_size(WARP_SIZE, BLOCK_SIZE);
    int mem_size = 0;

    // Raw pointers (reinterpret_cast to __half* is safe with torch.half)
    const __half *M_values_ptr = reinterpret_cast<const __half *>(M_values.data_ptr<at::Half>());
    const unsigned char *M_deltas_ptr = reinterpret_cast<const unsigned char *>(M_deltas.data_ptr<unsigned char>());
    const int *M_row_indices_ptr = reinterpret_cast<int *>(M_row_indices.data_ptr<int>());
    const __half *V_ptr = reinterpret_cast<const __half *>(V.data_ptr<at::Half>());
    __half *C_ptr = reinterpret_cast<__half *>(C.data_ptr<at::Half>());

    // Use proper stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int int_M_rows = M_rows;
    int int_M_cols = M_cols;

    // Call kernel (extern "C")
    float alpha = 1;
    float beta = 0;
    macko_spmv<false><<<grid_size, block_size, mem_size, stream>>>(
        M_values_ptr, M_deltas_ptr, M_row_indices_ptr, V_ptr, int_M_rows, int_M_cols, alpha, beta, C_ptr);

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::string msg = "CUDA kernel launch error: ";
        msg += cudaGetErrorString(err);
        TORCH_CHECK(false, msg);
    }

    return C;
}