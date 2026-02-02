#include "cpu_compressor_torch_launcher.cuh"
#include "kernels_torch_launchers.cuh"

TORCH_LIBRARY(macko_spmv, m) {
    m.def("multiply(Tensor M_values, Tensor M_deltas, Tensor M_row_indices, \
           int M_rows, int M_cols, Tensor V) -> Tensor");
}

TORCH_LIBRARY_IMPL(macko_spmv, CUDA, m) {
    m.impl("multiply", &macko_spmv_launcher);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cpu_compress", &cpu_compress, "cpu_compress");

    m.def("macko_spmv_launcher", \
           torch::wrap_pybind_function(macko_spmv_launcher), \
           "macko_spmv_launcher");
}
