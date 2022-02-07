import glob
import os

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (BuildExtension, CppExtension,
                                       CUDAExtension)


def get_extensions():
    extensions = []
    ext_name = 'detops._ext'

    # prevent ninja from using too many resources
    os.environ.setdefault('MAX_JOBS', '4')
    define_macros = []
    extra_compile_args = {'cxx': []}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('DETOPS_WITH_CUDA', None)]
        extra_compile_args['nvcc'] = []
        op_files = glob.glob('./detops/csrc/pytorch/*')
        extension = CUDAExtension
    else:
        print(f'Compiling {ext_name} without CUDA')
        op_files = glob.glob('./detops/csrc/pytorch/*.cpp')
        extension = CppExtension

    include_path = os.path.abspath('./detops/csrc/include')
    ext_ops = extension(name=ext_name,
                        sources=op_files,
                        include_dirs=[include_path],
                        define_macros=define_macros,
                        extra_compile_args=extra_compile_args)
    extensions.append(ext_ops)
    return extensions


if __name__ == '__main__':
    setup(
        name='detops',
        version='0.0',
        description='My detection ops lib.',
        author='dwy927',
        author_email='daiwenying927@gmail.com',
        url='https://github.com/dwy927/MyDetOpsLib',
        packages=find_packages(),
        classifiers=[
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
        ],
        cmdclass={'build_ext': BuildExtension},
        ext_modules=get_extensions(),
    )
