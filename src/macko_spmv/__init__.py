from torch.utils.cpp_extension import load_inline
import os
import torch
from importlib import resources
from importlib.metadata import version

LIB_NAME = "macko_spmv"

pkg_version = version(LIB_NAME).replace(".", "_")
cuda_resources = resources.files(f"{LIB_NAME}.cuda")

BUILD_DIRECTORY = os.environ.get("MACKO_SPMV_BUILD_DIRECTORY", "")

def __init_compressor():
    compressor_source_code = cuda_resources.joinpath("cpu_compressor.cuh").read_text()
    launcher_source_code = cuda_resources.joinpath(
        "cpu_compressor_torch_launcher.cuh"
    ).read_text()

    lib_name = f"macko_spmv_compression_{pkg_version}"
    build_directory = None if not BUILD_DIRECTORY else os.path.join(BUILD_DIRECTORY, lib_name)
    os.makedirs(build_directory, exist_ok=True)

    lib = load_inline(
        name=lib_name,
        cpp_sources=[compressor_source_code + "\n" + launcher_source_code],
        functions=["cpu_compress"],
        verbose=False,
        with_cuda=True,
        build_directory=build_directory,
    )
    return lib


__compressor_lib = __init_compressor()


def __init_multiply():
    kernel_source_code = cuda_resources.joinpath("kernels.cuh").read_text()
    launcher_source_code = cuda_resources.joinpath(
        "kernels_torch_launchers.cuh"
    ).read_text()

    full_source_code = kernel_source_code + "\n" + launcher_source_code

    cpp_source = """
at::Tensor macko_spmv_launcher(
    at::Tensor M_values,
    at::Tensor M_deltas,
    at::Tensor M_row_indices,
    int64_t M_rows, int64_t M_cols,
    at::Tensor V
);

TORCH_LIBRARY(macko_spmv, m) {
    m.def("multiply(Tensor M_values, Tensor M_deltas, Tensor M_row_indices, \
            int M_rows, int M_cols, Tensor V) -> Tensor");
}

TORCH_LIBRARY_IMPL(macko_spmv, CUDA, m) {
    m.impl("multiply", &macko_spmv_launcher);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("macko_spmv_launcher", \
            torch::wrap_pybind_function(macko_spmv_launcher), \
            "macko_spmv_launcher");
}
    """

    lib_name = f"macko_spmv_multiplication_{pkg_version}"
    build_directory = None if not BUILD_DIRECTORY else os.path.join(BUILD_DIRECTORY, lib_name)
    os.makedirs(build_directory, exist_ok=True)

    lib = load_inline(
        name=lib_name,
        cpp_sources=cpp_source,
        cuda_sources=full_source_code,
        functions=None,
        verbose=False,
        with_cuda=True,
        extra_cuda_cflags=["-O3"],
        build_directory=build_directory,
    )
    return lib


__multiply_lib = __init_multiply()


def __torch_registration():
    build_directory = __multiply_lib.__file__

    assert (
        os.path.isfile(build_directory) == 1
    ), f"Expected one _C*.so file, found {build_directory}"
    torch.ops.load_library(__multiply_lib.__file__)


__torch_registration()


@torch.library.register_fake("macko_spmv::multiply")
def _(a, b, c, d, e, f):
    return torch.empty((d,), device=a.device, dtype=a.dtype)


def move_to_device(compressed, device):
    return (
        compressed[0].to(device=device),
        compressed[1].to(device=device),
        compressed[2].to(device=device),
        compressed[3],
        compressed[4],
    )


def compress(M):
    assert M.is_contiguous()
    if M.device.type == "cuda":
        # TODO: implement properly fast gpu only compression
        compressed = __compressor_lib.cpu_compress(M.to("cpu"))
        return move_to_device(compressed, "cuda")

    elif M.device.type == "cpu":
        compressed = __compressor_lib.cpu_compress(M)
        return compressed
    else:
        raise NotImplementedError()


def multiply(compressed_M, V):
    assert compressed_M[0].is_cuda
    assert compressed_M[1].is_cuda
    assert compressed_M[2].is_cuda
    assert V.is_cuda
    assert V.is_contiguous()

    # __multiply_lib.macko_spmv_launcher is also usable
    return torch.ops.macko_spmv.multiply.default(
        compressed_M[0],
        compressed_M[1],
        compressed_M[2],
        compressed_M[3],
        compressed_M[4],
        V,
    )
