"""
硬件支持模块
提供 FPGA 硬件实现相关的功能
"""

from .fixed_point import (
    FixedPointConfig,
    FixedPointNumber,
    FixedPointTensor,
    FixedPointConverter,
    FixedPointArithmetic,
    HardwareCompatibility,
    create_fixed_point,
    create_fixed_point_tensor
)

__all__ = [
    'FixedPointConfig',
    'FixedPointNumber',
    'FixedPointTensor',
    'FixedPointConverter',
    'FixedPointArithmetic',
    'HardwareCompatibility',
    'create_fixed_point',
    'create_fixed_point_tensor'
]
