from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='custom_kernerls',
    ext_modules=[
        CUDAExtension('custom_kernels', ['add_cuda.cu', 'bindings.cpp'])
    ],
    cmdclass={'build_ext': BuildExtension},
)
