# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.deimport io

import io
import os
import os.path as osp

from setuptools import find_packages, setup

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

NAME = 'bvh_distance_queries'
DESCRIPTION = 'BVH operations'
URL = ''
EMAIL = 'wzielonka'
AUTHOR = 'Wojciech Zielonka'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = '1.0.1'

here = os.path.abspath(os.path.dirname(__file__))

about = {}
if not VERSION:
    with open(os.path.join(here, NAME, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION

bvh_src_files = ['src/bvh.cpp', 'src/bvh_cuda_op.cu']
bvh_include_dirs = torch.utils.cpp_extension.include_paths() + [
    'include',
    osp.expandvars('cuda-samples/Common')]

nvcc_args = ["-O3"]
nvcc_args.extend(
    [
        "-gencode=arch=compute_70,code=sm_70",
        "-gencode=arch=compute_75,code=sm_75",
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_86,code=sm_86",
        "-gencode=arch=compute_90,code=sm_90",
    ]
)

bvh_extra_compile_args = {'nvcc': ['-DPRINT_TIMINGS=0',
                                   '-DDEBUG_PRINT=0',
                                   '-DERROR_CHECKING=1',
                                   '-DNUM_THREADS_CUSTOM=256',
                                   '-DBVH_PROFILING=0',
                                   ] + nvcc_args,
                          'cxx': []}

bvh_extension = CUDAExtension('bvh_distance_queries_cuda',
                              bvh_src_files,
                              include_dirs=bvh_include_dirs,
                              extra_compile_args=bvh_extra_compile_args)

setup(name=NAME,
      version=about['__version__'],
      description=DESCRIPTION,
      author=AUTHOR,
      author_email=EMAIL,
      python_requires=REQUIRES_PYTHON,
      url=URL,
      packages=find_packages(),
      ext_modules=[bvh_extension],
      install_requires=[
          'torch>=1.0.1',
      ],
      cmdclass={'build_ext': BuildExtension})
