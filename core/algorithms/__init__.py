"""
算法模块
提供查找表和数学工具函数
"""

from .math_utils import (
    MathUtils,
    safe_exp,
    safe_log,
    safe_sqrt,
    safe_divide
)

from .lookup_table import (
    LookupTableConfig,
    SamplingStrategy,
    BinarySampling,
    InterpolationMethod,
    DirectLookup,
    LinearInterpolation,
    QuadraticInterpolation,
    LookupTable,
    create_exp_table,
    create_sigmoid_table
)

__all__ = [
    # 数学工具
    'MathUtils', 'safe_exp', 'safe_log', 'safe_sqrt', 'safe_divide',
    
    # 查找表
    'LookupTableConfig', 'SamplingStrategy', 'BinarySampling', 'UniformSampling',
    'AdaptiveSampling', 'LogarithmicSampling', 'QuadraticSampling',
    'InterpolationMethod', 'DirectLookup', 'LinearInterpolation',
    'QuadraticInterpolation', 'LookupTable',
    'create_exp_table', 'create_sigmoid_table'
]
