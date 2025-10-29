#include <vector>
#include <cuda_fp16.h>

#define FORMATTER_UPDIV(a, b) (a + b - 1) / b

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int, int>
 cpu_compress(at::Tensor M) {
    TORCH_CHECK(M.device() == torch::kCPU, "M must be on cpu");
    TORCH_CHECK(M.dtype() == at::kHalf, "M must be torch.half");

    int *row_indices;
    __half *values;
    unsigned char *deltas;

    const __half* M_ptr = reinterpret_cast<const __half*>(M.data_ptr<at::Half>());
    int M_rows = M.size(0);
    int M_cols = M.size(1);
    int M_sparse_size = 0;
    int M_sparse_padded_size = 0;

    compress_rows_cpu(
        M_ptr, M_rows, M_cols, &row_indices, &deltas, &values, &M_sparse_size, &M_sparse_padded_size);

    auto row_indices_options = torch::TensorOptions()
                       .dtype(torch::kInt32)
                       .device(torch::kCPU);
    auto values_options = torch::TensorOptions()
                       .dtype(torch::kFloat16)
                       .device(torch::kCPU);
    auto deltas_options = torch::TensorOptions()
                       .dtype(torch::kUInt8)
                       .device(torch::kCPU);

    at::Tensor t_values = torch::from_blob(values, {(long int)(M_sparse_padded_size)}, values_options).clone();
    at::Tensor t_deltas = torch::from_blob(deltas, {(long int)(M_sparse_padded_size/2)}, deltas_options).clone();
    at::Tensor t_row_indices = torch::from_blob(row_indices, {(long int)(M_rows+1)}, row_indices_options).clone();

    return std::make_tuple(t_values, t_deltas, t_row_indices, M.size(0), M.size(1));
}
