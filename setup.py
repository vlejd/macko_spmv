from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=[
        CUDAExtension(
            name="macko_spmv._C",
            sources=["src/macko_spmv/cuda/torch_bindings.cu"],
            extra_compile_args={"nvcc": ["-O3"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    include_package_data=True,
)
