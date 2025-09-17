"""
定点数支持模块
为 FPGA 硬件实现提供定点数计算支持
"""

import math
import torch
import numpy as np
from typing import Union, Tuple, Optional, Dict, Any
from dataclasses import dataclass

from core.base.constants import (
    FIXED_POINT_FORMATS, DEFAULT_FIXED_POINT_FORMAT, 
    FIXED_POINT_PRECISION, NUMERICAL_STABILITY
)
from core.base.exceptions import (
    UnsupportedDataTypeError, NumericalOverflowError, 
    validate_dtype, HardwareNotSupportedError
)


@dataclass
class FixedPointConfig:
    """定点数配置"""
    format: str = DEFAULT_FIXED_POINT_FORMAT
    integer_bits: int = 16
    fractional_bits: int = 16
    signed: bool = True
    rounding_mode: str = 'round'  # 'round', 'floor', 'ceil', 'trunc'


class FixedPointNumber:
    """定点数类"""
    
    def __init__(self, value: Union[float, int], config: FixedPointConfig):
        self.config = config
        self._validate_config()
        
        # 计算定点数参数
        self.total_bits = config.integer_bits + config.fractional_bits
        self.scale_factor = 2 ** config.fractional_bits
        self.max_value = (2 ** (self.total_bits - 1) - 1) / self.scale_factor if config.signed else (2 ** self.total_bits - 1) / self.scale_factor
        self.min_value = -(2 ** (self.total_bits - 1)) / self.scale_factor if config.signed else 0
        
        # 转换并存储定点数值
        self._raw_value = self._float_to_fixed(value)
    
    def _validate_config(self) -> None:
        """验证配置参数"""
        if self.config.format not in FIXED_POINT_FORMATS:
            raise ValueError(f"不支持的定点数格式: {self.config.format}")
        
        if self.config.format in FIXED_POINT_PRECISION:
            expected_int, expected_frac = FIXED_POINT_PRECISION[self.config.format]
            if (self.config.integer_bits, self.config.fractional_bits) != (expected_int, expected_frac):
                raise ValueError(f"格式 {self.config.format} 的位数不匹配: "
                               f"期望 ({expected_int}, {expected_frac}), "
                               f"实际 ({self.config.integer_bits}, {self.config.fractional_bits})")
    
    def _float_to_fixed(self, value: float) -> int:
        """将浮点数转换为定点数"""
        # 检查范围
        if value > self.max_value:
            raise NumericalOverflowError(value, self.max_value)
        if value < self.min_value:
            raise NumericalOverflowError(value, self.min_value)
        
        # 缩放并四舍五入
        scaled_value = value * self.scale_factor
        
        if self.config.rounding_mode == 'round':
            return int(round(scaled_value))
        elif self.config.rounding_mode == 'floor':
            return int(math.floor(scaled_value))
        elif self.config.rounding_mode == 'ceil':
            return int(math.ceil(scaled_value))
        elif self.config.rounding_mode == 'trunc':
            return int(scaled_value)
        else:
            raise ValueError(f"不支持的舍入模式: {self.config.rounding_mode}")
    
    def _fixed_to_float(self) -> float:
        """将定点数转换为浮点数"""
        return self._raw_value / self.scale_factor
    
    def to_float(self) -> float:
        """转换为浮点数"""
        return self._fixed_to_float()
    
    def to_int(self) -> int:
        """转换为整数（原始定点数表示）"""
        return self._raw_value
    
    def __add__(self, other: 'FixedPointNumber') -> 'FixedPointNumber':
        """定点数加法"""
        if not isinstance(other, FixedPointNumber):
            other = FixedPointNumber(other, self.config)
        
        # 检查格式兼容性
        if self.config != other.config:
            raise ValueError("定点数格式不兼容")
        
        # 直接相加定点数
        result_raw = self._raw_value + other._raw_value
        
        # 检查溢出
        max_raw = (2 ** (self.total_bits - 1) - 1) if self.config.signed else (2 ** self.total_bits - 1)
        min_raw = -(2 ** (self.total_bits - 1)) if self.config.signed else 0
        
        if result_raw > max_raw or result_raw < min_raw:
            raise NumericalOverflowError(result_raw / self.scale_factor, self.max_value)
        
        result = FixedPointNumber(0, self.config)
        result._raw_value = result_raw
        return result
    
    def __sub__(self, other: 'FixedPointNumber') -> 'FixedPointNumber':
        """定点数减法"""
        if not isinstance(other, FixedPointNumber):
            other = FixedPointNumber(other, self.config)
        
        # 检查格式兼容性
        if self.config != other.config:
            raise ValueError("定点数格式不兼容")
        
        # 直接相减定点数
        result_raw = self._raw_value - other._raw_value
        
        # 检查溢出
        max_raw = (2 ** (self.total_bits - 1) - 1) if self.config.signed else (2 ** self.total_bits - 1)
        min_raw = -(2 ** (self.total_bits - 1)) if self.config.signed else 0
        
        if result_raw > max_raw or result_raw < min_raw:
            raise NumericalOverflowError(result_raw / self.scale_factor, self.max_value)
        
        result = FixedPointNumber(0, self.config)
        result._raw_value = result_raw
        return result
    
    def __mul__(self, other: 'FixedPointNumber') -> 'FixedPointNumber':
        """定点数乘法"""
        if not isinstance(other, FixedPointNumber):
            other = FixedPointNumber(other, self.config)
        
        # 检查格式兼容性
        if self.config != other.config:
            raise ValueError("定点数格式不兼容")
        
        # 定点数乘法需要额外的缩放
        result_raw = (self._raw_value * other._raw_value) >> self.config.fractional_bits
        
        # 检查溢出
        max_raw = (2 ** (self.total_bits - 1) - 1) if self.config.signed else (2 ** self.total_bits - 1)
        min_raw = -(2 ** (self.total_bits - 1)) if self.config.signed else 0
        
        if result_raw > max_raw or result_raw < min_raw:
            raise NumericalOverflowError(result_raw / self.scale_factor, self.max_value)
        
        result = FixedPointNumber(0, self.config)
        result._raw_value = result_raw
        return result
    
    def __truediv__(self, other: 'FixedPointNumber') -> 'FixedPointNumber':
        """定点数除法"""
        if not isinstance(other, FixedPointNumber):
            other = FixedPointNumber(other, self.config)
        
        # 检查格式兼容性
        if self.config != other.config:
            raise ValueError("定点数格式不兼容")
        
        # 防止除零
        if other._raw_value == 0:
            raise ZeroDivisionError("定点数除法除零")
        
        # 定点数除法需要额外的缩放
        result_raw = (self._raw_value << self.config.fractional_bits) // other._raw_value
        
        # 检查溢出
        max_raw = (2 ** (self.total_bits - 1) - 1) if self.config.signed else (2 ** self.total_bits - 1)
        min_raw = -(2 ** (self.total_bits - 1)) if self.config.signed else 0
        
        if result_raw > max_raw or result_raw < min_raw:
            raise NumericalOverflowError(result_raw / self.scale_factor, self.max_value)
        
        result = FixedPointNumber(0, self.config)
        result._raw_value = result_raw
        return result
    
    def __neg__(self) -> 'FixedPointNumber':
        """定点数取负"""
        result = FixedPointNumber(0, self.config)
        result._raw_value = -self._raw_value
        return result
    
    def __abs__(self) -> 'FixedPointNumber':
        """定点数绝对值"""
        result = FixedPointNumber(0, self.config)
        result._raw_value = abs(self._raw_value)
        return result
    
    def __lt__(self, other: 'FixedPointNumber') -> bool:
        """小于比较"""
        if not isinstance(other, FixedPointNumber):
            other = FixedPointNumber(other, self.config)
        return self._raw_value < other._raw_value
    
    def __le__(self, other: 'FixedPointNumber') -> bool:
        """小于等于比较"""
        if not isinstance(other, FixedPointNumber):
            other = FixedPointNumber(other, self.config)
        return self._raw_value <= other._raw_value
    
    def __gt__(self, other: 'FixedPointNumber') -> bool:
        """大于比较"""
        if not isinstance(other, FixedPointNumber):
            other = FixedPointNumber(other, self.config)
        return self._raw_value > other._raw_value
    
    def __ge__(self, other: 'FixedPointNumber') -> bool:
        """大于等于比较"""
        if not isinstance(other, FixedPointNumber):
            other = FixedPointNumber(other, self.config)
        return self._raw_value >= other._raw_value
    
    def __eq__(self, other: 'FixedPointNumber') -> bool:
        """等于比较"""
        if not isinstance(other, FixedPointNumber):
            other = FixedPointNumber(other, self.config)
        return self._raw_value == other._raw_value
    
    def __ne__(self, other: 'FixedPointNumber') -> bool:
        """不等于比较"""
        return not (self == other)
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"FixedPoint({self.to_float():.6f}, {self.config.format})"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return (f"FixedPointNumber(value={self.to_float():.6f}, "
                f"raw={self._raw_value}, format={self.config.format})")


class FixedPointTensor:
    """定点数张量类"""
    
    def __init__(self, data: torch.Tensor, config: FixedPointConfig):
        self.config = config
        self.original_dtype = data.dtype
        self.original_device = data.device
        
        # 转换为定点数
        self._raw_data = self._tensor_to_fixed(data)
    
    def _tensor_to_fixed(self, tensor: torch.Tensor) -> torch.Tensor:
        """将张量转换为定点数表示"""
        # 检查范围
        max_val = tensor.max().item()
        min_val = tensor.min().item()
        
        if max_val > self.config.max_value or min_val < self.config.min_value:
            raise NumericalOverflowError(max_val, self.config.max_value)
        
        # 缩放并四舍五入
        scale_factor = 2 ** self.config.fractional_bits
        scaled_tensor = tensor * scale_factor
        
        if self.config.rounding_mode == 'round':
            return torch.round(scaled_tensor).to(torch.int32)
        elif self.config.rounding_mode == 'floor':
            return torch.floor(scaled_tensor).to(torch.int32)
        elif self.config.rounding_mode == 'ceil':
            return torch.ceil(scaled_tensor).to(torch.int32)
        elif self.config.rounding_mode == 'trunc':
            return torch.trunc(scaled_tensor).to(torch.int32)
        else:
            raise ValueError(f"不支持的舍入模式: {self.config.rounding_mode}")
    
    def _fixed_to_tensor(self) -> torch.Tensor:
        """将定点数转换回浮点张量"""
        scale_factor = 2 ** self.config.fractional_bits
        return self._raw_data.float() / scale_factor
    
    def to_float_tensor(self) -> torch.Tensor:
        """转换为浮点张量"""
        return self._fixed_to_tensor().to(self.original_dtype).to(self.original_device)
    
    def to_int_tensor(self) -> torch.Tensor:
        """转换为整数张量（原始定点数表示）"""
        return self._raw_data
    
    def __add__(self, other: 'FixedPointTensor') -> 'FixedPointTensor':
        """定点数张量加法"""
        if not isinstance(other, FixedPointTensor):
            other = FixedPointTensor(other, self.config)
        
        if self.config != other.config:
            raise ValueError("定点数格式不兼容")
        
        result_raw = self._raw_data + other._raw_data
        
        # 检查溢出
        max_raw = (2 ** (self.config.integer_bits + self.config.fractional_bits - 1) - 1) if self.config.signed else (2 ** (self.config.integer_bits + self.config.fractional_bits) - 1)
        min_raw = -(2 ** (self.config.integer_bits + self.config.fractional_bits - 1)) if self.config.signed else 0
        
        if torch.any(result_raw > max_raw) or torch.any(result_raw < min_raw):
            raise NumericalOverflowError("定点数张量加法溢出")
        
        result = FixedPointTensor(torch.zeros_like(self.to_float_tensor()), self.config)
        result._raw_data = result_raw
        return result
    
    def __mul__(self, other: 'FixedPointTensor') -> 'FixedPointTensor':
        """定点数张量乘法"""
        if not isinstance(other, FixedPointTensor):
            other = FixedPointTensor(other, self.config)
        
        if self.config != other.config:
            raise ValueError("定点数格式不兼容")
        
        # 定点数乘法需要额外的缩放
        result_raw = (self._raw_data * other._raw_data) >> self.config.fractional_bits
        
        # 检查溢出
        max_raw = (2 ** (self.config.integer_bits + self.config.fractional_bits - 1) - 1) if self.config.signed else (2 ** (self.config.integer_bits + self.config.fractional_bits) - 1)
        min_raw = -(2 ** (self.config.integer_bits + self.config.fractional_bits - 1)) if self.config.signed else 0
        
        if torch.any(result_raw > max_raw) or torch.any(result_raw < min_raw):
            raise NumericalOverflowError("定点数张量乘法溢出")
        
        result = FixedPointTensor(torch.zeros_like(self.to_float_tensor()), self.config)
        result._raw_data = result_raw
        return result


class FixedPointConverter:
    """定点数转换器"""
    
    @staticmethod
    def float_to_fixed(value: float, config: FixedPointConfig) -> FixedPointNumber:
        """浮点数转定点数"""
        return FixedPointNumber(value, config)
    
    @staticmethod
    def tensor_to_fixed(tensor: torch.Tensor, config: FixedPointConfig) -> FixedPointTensor:
        """张量转定点数张量"""
        return FixedPointTensor(tensor, config)
    
    @staticmethod
    def fixed_to_float(fixed: FixedPointNumber) -> float:
        """定点数转浮点数"""
        return fixed.to_float()
    
    @staticmethod
    def fixed_tensor_to_float(fixed_tensor: FixedPointTensor) -> torch.Tensor:
        """定点数张量转浮点张量"""
        return fixed_tensor.to_float_tensor()


class FixedPointArithmetic:
    """定点数算术运算"""
    
    @staticmethod
    def add(a: FixedPointNumber, b: FixedPointNumber) -> FixedPointNumber:
        """定点数加法"""
        return a + b
    
    @staticmethod
    def subtract(a: FixedPointNumber, b: FixedPointNumber) -> FixedPointNumber:
        """定点数减法"""
        return a - b
    
    @staticmethod
    def multiply(a: FixedPointNumber, b: FixedPointNumber) -> FixedPointNumber:
        """定点数乘法"""
        return a * b
    
    @staticmethod
    def divide(a: FixedPointNumber, b: FixedPointNumber) -> FixedPointNumber:
        """定点数除法"""
        return a / b
    
    @staticmethod
    def sqrt(x: FixedPointNumber) -> FixedPointNumber:
        """定点数平方根"""
        # 使用牛顿法计算平方根
        if x < FixedPointNumber(0, x.config):
            raise ValueError("负数不能开平方根")
        
        # 初始猜测
        guess = FixedPointNumber(x.to_float() / 2, x.config)
        
        # 牛顿迭代
        for _ in range(10):  # 最多10次迭代
            if guess == FixedPointNumber(0, x.config):
                break
            new_guess = (guess + x / guess) / FixedPointNumber(2, x.config)
            if abs(new_guess - guess).to_float() < 1e-6:
                break
            guess = new_guess
        
        return guess
    
    @staticmethod
    def exp(x: FixedPointNumber) -> FixedPointNumber:
        """定点数指数函数（近似）"""
        # 使用泰勒级数近似
        result = FixedPointNumber(1, x.config)
        term = FixedPointNumber(1, x.config)
        
        for i in range(1, 10):  # 10项泰勒级数
            term = term * x / FixedPointNumber(i, x.config)
            result = result + term
        
        return result


class HardwareCompatibility:
    """硬件兼容性检查"""
    
    @staticmethod
    def check_fpga_support(config: FixedPointConfig) -> bool:
        """检查 FPGA 是否支持该定点数格式"""
        total_bits = config.integer_bits + config.fractional_bits
        
        # 检查位数是否在 FPGA 支持范围内
        if total_bits > 64:
            raise HardwareNotSupportedError(f"FPGA 不支持 {total_bits} 位定点数")
        
        # 检查是否有足够的 DSP 资源
        if total_bits > 32:
            # 需要多个 DSP 块
            return True
        
        return True
    
    @staticmethod
    def get_optimal_config(target_precision: float) -> FixedPointConfig:
        """根据目标精度获取最优配置"""
        # 计算需要的小数位数
        fractional_bits = int(math.ceil(-math.log2(target_precision)))
        
        # 选择最接近的标准格式
        best_format = DEFAULT_FIXED_POINT_FORMAT
        best_error = float('inf')
        
        for format_name, (int_bits, frac_bits) in FIXED_POINT_PRECISION.items():
            if frac_bits >= fractional_bits:
                error = abs(frac_bits - fractional_bits)
                if error < best_error:
                    best_error = error
                    best_format = format_name
        
        int_bits, frac_bits = FIXED_POINT_PRECISION[best_format]
        return FixedPointConfig(
            format=best_format,
            integer_bits=int_bits,
            fractional_bits=frac_bits
        )


# 便捷函数
def create_fixed_point(value: Union[float, int], 
                      format: str = DEFAULT_FIXED_POINT_FORMAT) -> FixedPointNumber:
    """创建定点数"""
    int_bits, frac_bits = FIXED_POINT_PRECISION[format]
    config = FixedPointConfig(
        format=format,
        integer_bits=int_bits,
        fractional_bits=frac_bits
    )
    return FixedPointNumber(value, config)


def create_fixed_point_tensor(tensor: torch.Tensor, 
                             format: str = DEFAULT_FIXED_POINT_FORMAT) -> FixedPointTensor:
    """创建定点数张量"""
    int_bits, frac_bits = FIXED_POINT_PRECISION[format]
    config = FixedPointConfig(
        format=format,
        integer_bits=int_bits,
        fractional_bits=frac_bits
    )
    return FixedPointTensor(tensor, config)
