# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#!/usr/bin/env python

import glob
import os

import torch
from setuptools import find_packages
from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CUDAExtension

requirements = ["torch", "torchvision"]


def get_extensions():
    print(CUDA_HOME)
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "csrc")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    sources = main_file

    extra_compile_args = {"cxx": []}
    define_macros = []

    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]

        extra_compile_args["nvcc"] = [
            "-arch=sm_60",
            #"-DCUDA_HAS_FP16=1",
            #"-D__CUDA_NO_HALF_OPERATORS__",
            #"-D__CUDA_NO_HALF_CONVERSIONS__",
            #"-D__CUDA_NO_HALF2_OPERATORS__",
        ]

    sources = [os.path.join(extensions_dir, s) for s in sources]

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "Deformable_MatMul_cu3",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


setup(
    name="Deformable_MatMul_cu3",
    version="3.0",
    description="deformable_matmul3",
    #packages=find_packages(exclude=("configs", "tests",)),
    #install_requires=requirements,
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
