import os
import subprocess
from pathlib import Path

from packaging.version import Version, parse
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension

from tools.utils import get_build_cuda_cflags, get_build_sources

# package name managed by pip, which can be remove by `pip uninstall toy-hgemm`
PACKAGE_NAME = "toy-hgemm"

ext_modules = []
generator_flag = []
cc_flag = []
cc_flag.append("-gencode")
cc_flag.append("arch=compute_80,code=sm_80")
cc_flag.append("-gencode")
cc_flag.append("arch=compute_89,code=sm_89")


# helper function to get cuda version
def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output(
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


if CUDA_HOME is not None:
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    if bare_metal_version >= Version("11.8"):
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_90,code=sm_90")

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

# cuda module
# may need export LD_LIBRARY_PATH=PATH-TO/torch/lib:$LD_LIBRARY_PATH
ext_modules.append(
    CUDAExtension(
        # package name for import
        name="toy_hgemm",
        sources=get_build_sources(),
        extra_compile_args={
            # add c compile flags
            "cxx": ["-O3", "-std=c++17"] + generator_flag,
            # add nvcc compile flags
            "nvcc": get_build_cuda_cflags(build_pkg=True) + generator_flag + cc_flag,
        },
        include_dirs=[
            Path(this_dir) / "csrc",
            Path(this_dir) / "include",
            Path(this_dir) / "third_party/cutlass/include",
        ],
    )
)

setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "third_party",
            "tmp",
        )
    ),
    description="My Toy HGEMM implement by CUDA",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
    install_requires=[
        "torch",
        "packaging",
        "ninja",
    ],
)
