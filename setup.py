import os
import os.path as osp
import sys
import glob
import torch
from torch.__config__ import parallel_info
from itertools import product
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import CppExtension, CUDAExtension, CUDA_HOME

WITH_CUDA = torch.cuda.is_available() and CUDA_HOME is not None
suffices = ['cpu', 'cuda'] if WITH_CUDA else ['cpu']
if os.getenv('FORCE_CUDA', '0') == '1':
    suffices = ['cuda', 'cpu']
if os.getenv('FORCE_ONLY_CUDA', '0') == '1':
    suffices = ['cuda']
if os.getenv('FORCE_ONLY_CPU', '0') == '1':
    suffices = ['cpu']


def get_extensions():
    extensions = []

    extensions_dir = osp.join(osp.dirname(osp.abspath(__file__)), 'csrc')
    main_files = glob.glob(osp.join(extensions_dir, '*.cpp'))

    for main, suffix in product(main_files, suffices):
        define_macros = []
        libraries = []

        extra_compile_args = {'cxx': ['-O2']}
        extra_link_args = ['-s']

        info = parallel_info()
        if 'backend: OpenMP' in info and 'OpenMP not found' not in info:
            extra_compile_args['cxx'] += ['-DAT_PARALLEL_OPENMP']
            if sys.platform == 'win32':
                extra_compile_args['cxx'] += ['/openmp']
            else:
                extra_compile_args['cxx'] += ['-fopenmp']
        else:
            print('Compiling without OpenMP...')

        if suffix == 'cuda':
            define_macros += [('WITH_CUDA', None)]
            nvcc_flags = os.getenv('NVCC_FLAGS', '')
            nvcc_flags = [] if nvcc_flags == '' else nvcc_flags.split(' ')
            nvcc_flags += ['-arch=sm_35', '--expt-relaxed-constexpr', '-O2']
            extra_compile_args['nvcc'] = nvcc_flags

            if sys.platform == 'win32':
                extra_link_args += ['cusparse.lib']
            else:
                extra_link_args += ['-lcusparse', '-l', 'cusparse']

        name = main.split(os.sep)[-1][:-4]
        sources = [main]

        path = osp.join(extensions_dir, 'cpu', f'{name}_cpu.cpp')
        if osp.exists(path):
            sources += [path]

        path = osp.join(extensions_dir, 'cuda', f'{name}_cuda.cu')
        if suffix == 'cuda' and osp.exists(path):
            sources += [path]

        Extension = CppExtension if suffix == 'cpu' else CUDAExtension
        extension = Extension(
            f'thgsp._{name}_{suffix}',
            sources,
            include_dirs=[extensions_dir],
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            libraries=libraries,
        )
        extensions += [extension]

    return extensions


install_requires = [
    'torch>=1.7.0',
    'torchvision',
    'ray',
    'sklearn',
    'numpy',
    'scipy',
    'networkx',
    'torch-scatter',
    'torch-sparse',
    'torch-cluster',
    'scikit-sparse',
    'matplotlib',
    'pandas',
    'plotly'
]

tests_require = ['pytest',
                 'pytest-cov',
                 'pytest-pep8',
                 'pytest-xdist']

setup(
    name="thgsp",
    version="0.1.0",
    author="Bowen Deng",
    author_email="bowen.deng20@gmail.com",
    description="A graph signal processing toolbox built on PyTorch",
    long_description_content_type="text/markdown",
    url="https://github.com/bwdeng20/thgsp",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: BSD License",
        "Operating System :: Linux",
    ],
    keywords=[
        'pytorch',
        'graph-signal-processing',
        'graph-wavelet-filterbank'
    ],
    python_requires='>=3.6',
    install_requires=install_requires,
    tests_require=tests_require,
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=False)},
    packages=find_packages(),
)
