import os
import glob
import torch
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

library_name = "ckl_custom_ops"
debug_mode = os.getenv("DEBUG", "0") == "1"
if debug_mode:
    print("Compiling in debug mode")

extra_link_args = []
extra_compile_args = {
    "cxx": [
        "-O3" if not debug_mode else "-O0",
        # "-fdiagnostics-color=always",
        # "-DPy_LIMITED_API=0x03090000",  # min CPython version 3.9
    ],
    
    "nvcc": [
        "-O3" if not debug_mode else "-O0",
        '-std=c++17',
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        '--expt-relaxed-constexpr',
        "--expt-extended-lambda"
        # "-DNDEBUG"
        # "-Xcompiler",
        # "-fPIC",
        # '-gencode=arch=compute_100,code=sm_100',
        # "-gencode=arch=compute_90,code=sm_90"
        # "-DFLASHINFER_ENABLE_F16",
        # "-DCUTE_USE_PACKED_TUPLE=1",
        # "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
        # "-DCUTLASS_VERSIONS_GENERATED",
        # "-DCUTLASS_TEST_LEVEL=0",
        # "-DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1",
        # "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
        # "--expt-relaxed-constexpr",
        # "--expt-extended-lambda",
        # '-gencode=arch=compute_80,code=sm_80',
        # '-U__CUDA_NO_HALF_OPERATORS__',
        # '-U__CUDA_NO_HALF_CONVERSIONS__',
        # '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
        # '-U__CUDA_NO_HALF2_OPERATORS__',
    ],
}
if debug_mode:
    extra_compile_args["cxx"].append("-g")
    extra_compile_args["nvcc"].append("-g")
    extra_link_args.extend(["-O0", "-g"])

this_dir = os.path.curdir
abs_this_dir = os.path.abspath(this_dir)
extensions_dir = os.path.join(this_dir, "csrc")
include_dirs = [
    'include',
    'third_party/cutlass/include',
]
for i in range(len(include_dirs)):
    include_dirs[i] = os.path.join(abs_this_dir, include_dirs[i])

sources = []
# sources = list(glob.glob(os.path.join(extensions_dir, "*.cc")))
# cuda_sources = list(glob.glob(os.path.join(extensions_dir, "*.cu"))
# sources = sources + cuda_sources
sources += [
    f'{extensions_dir}/common_extension.cc',
    # f'{extensions_dir}/add/custom_add_kernel.cu',
    f'{extensions_dir}/gemm/gemm.cu',
]
print(f'sources={sources}')

setup(
    name=library_name,
    ext_modules=[
        CUDAExtension(
            name=library_name,                # 模块名，Python import 时使用
            include_dirs=include_dirs,
            sources=sources, # CUDA 源文件
            extra_compile_args=extra_compile_args, # 可选的编译参数
            extra_link_args=extra_link_args,
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)