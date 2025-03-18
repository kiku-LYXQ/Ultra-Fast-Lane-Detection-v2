# 导入setuptools的setup函数，用于配置Python包
from setuptools import setup
# 导入PyTorch的C++/CUDA扩展构建工具
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 调用setup函数配置Python包
setup(
    # 扩展模块名称（Python导入时使用的名称）
    name='my_interp',

    # 指定扩展模块列表
    ext_modules=[
        CUDAExtension(
            # 生成的模块名称（与name参数对应）
            'my_interp',
            # 源文件列表（支持C++和CUDA文件）
            [
                'my_interp_cuda.cpp',  # C++包装代码
                'my_interp_cuda_kernel.cu'  # CUDA内核代码
            ]
        ),
    ],

    # 指定自定义命令类（使用PyTorch的构建扩展）
    cmdclass={
        'build_ext': BuildExtension
    }
)
