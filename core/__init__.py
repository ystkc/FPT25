"""
FPT25 核心模块
FPGA 激活函数硬件加速项目的核心功能模块
"""

# 从基础模块导入
from .base import *

__version__ = VERSION_INFO['version']
__author__ = VERSION_INFO['author']
__description__ = VERSION_INFO['description']

# 模块导出
__all__ = [
    # 版本信息
    '__version__', '__author__', '__description__'
]
