"""
数学工具函数模块
提供激活函数计算中需要的各种数学工具函数
"""

import math
import torch
import numpy as np
from typing import Union, Tuple, Optional, List
from functools import lru_cache

from core.base.constants import (
    MATH_CONSTANTS, NUMERICAL_STABILITY, EPSILON_TINY
)
from core.base.exceptions import (
    NumericalOverflowError, NumericalUnderflowError, 
    validate_dtype, UnsupportedDataTypeError
)


class MathUtils:
    """数学工具类"""
    
    @staticmethod
    def safe_exp(x: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        """
        安全的指数函数，防止数值溢出
        
        Args:
            x: 输入值
            
        Returns:
            计算结果
        """
        if isinstance(x, torch.Tensor):
            # 使用 PyTorch 的 clamp 函数限制输入范围
            x_clamped = torch.clamp(x, 
                                  min=NUMERICAL_STABILITY['log_min'],
                                  max=NUMERICAL_STABILITY['log_max'])
            return torch.exp(x_clamped)
        else:
            # 标量计算
            x_clamped = max(NUMERICAL_STABILITY['log_min'], 
                          min(NUMERICAL_STABILITY['log_max'], x))
            return math.exp(x_clamped)
    
    @staticmethod
    def safe_log(x: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        """
        安全的对数函数，防止数值问题
        
        Args:
            x: 输入值
            
        Returns:
            计算结果
        """
        if isinstance(x, torch.Tensor):
            # 确保输入为正数
            x_safe = torch.clamp(x, min=NUMERICAL_STABILITY['min_value'])
            return torch.log(x_safe)
        else:
            # 标量计算
            if x <= 0:
                return math.log(NUMERICAL_STABILITY['min_value'])
            return math.log(x)
    
    @staticmethod
    def safe_sqrt(x: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        """
        安全的平方根函数
        
        Args:
            x: 输入值
            
        Returns:
            计算结果
        """
        if isinstance(x, torch.Tensor):
            # 确保输入为非负数
            x_safe = torch.clamp(x, min=0.0)
            return torch.sqrt(x_safe)
        else:
            # 标量计算
            if x < 0:
                return 0.0
            return math.sqrt(x)
    
    @staticmethod
    def safe_divide(a: Union[float, torch.Tensor], 
                   b: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        """
        安全的除法运算，防止除零
        
        Args:
            a: 被除数
            b: 除数
            
        Returns:
            计算结果
        """
        if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
            # 张量计算
            if isinstance(a, (int, float)):
                a = torch.tensor(a, dtype=torch.float32)
            if isinstance(b, (int, float)):
                b = torch.tensor(b, dtype=torch.float32)
            
            # 防止除零
            b_safe = torch.where(torch.abs(b) < EPSILON_TINY, 
                               torch.tensor(EPSILON_TINY), b)
            return a / b_safe
        else:
            # 标量计算
            if abs(b) < EPSILON_TINY:
                return a / EPSILON_TINY
            return a / b
    
    @staticmethod
    def softmax_stable(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        数值稳定的 Softmax 实现
        
        Args:
            x: 输入张量
            dim: 计算维度
            
        Returns:
            Softmax 结果
        """
        # 减去最大值防止溢出
        x_max = torch.max(x, dim=dim, keepdim=True)[0]
        x_shifted = x - x_max
        
        # 限制指数范围防止溢出
        x_shifted = torch.clamp(x_shifted, min=-20, max=20)
        
        # 计算指数
        exp_x = torch.exp(x_shifted)
        
        # 计算和
        sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
        
        # 归一化，添加小常数防止除零
        return exp_x / (sum_exp + EPSILON_TINY)
    
    @staticmethod
    def layer_norm_stable(x: torch.Tensor, 
                         gamma: Optional[torch.Tensor] = None,
                         beta: Optional[torch.Tensor] = None,
                         eps: float = 1e-5) -> torch.Tensor:
        """
        数值稳定的 LayerNorm 实现
        
        Args:
            x: 输入张量
            gamma: 缩放参数
            beta: 偏移参数
            eps: 数值稳定性常数
            
        Returns:
            LayerNorm 结果
        """
        # 计算均值和方差
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, keepdim=True, unbiased=False)
        
        # 确保方差不为负（数值稳定性）
        var = torch.clamp(var, min=0.0)
        
        # 归一化，使用更稳定的平方根计算
        std = torch.sqrt(var + eps)
        x_norm = (x - mean) / std
        
        # 应用缩放和偏移
        if gamma is not None:
            x_norm = x_norm * gamma
        if beta is not None:
            x_norm = x_norm + beta
        
        return x_norm
    
    @staticmethod
    def rms_norm_stable(x: torch.Tensor,
                       gamma: Optional[torch.Tensor] = None,
                       eps: float = 1e-5) -> torch.Tensor:
        """
        数值稳定的 RMSNorm 实现
        
        Args:
            x: 输入张量
            gamma: 缩放参数
            eps: 数值稳定性常数
            
        Returns:
            RMSNorm 结果
        """
        # 计算 RMS，使用更稳定的方法
        x_squared = x ** 2
        mean_squared = torch.mean(x_squared, dim=-1, keepdim=True)
        
        # 确保均方根不为负
        mean_squared = torch.clamp(mean_squared, min=0.0)
        
        # 计算 RMS
        rms = torch.sqrt(mean_squared + eps)
        
        # 归一化，防止除零
        x_norm = x / (rms + EPSILON_TINY)
        
        # 应用缩放
        if gamma is not None:
            x_norm = x_norm * gamma
        
        return x_norm
    
    @staticmethod
    def sigmoid_stable(x: torch.Tensor) -> torch.Tensor:
        """
        数值稳定的 Sigmoid 实现
        
        Args:
            x: 输入张量
            
        Returns:
            Sigmoid 结果
        """
        # 使用 PyTorch 的稳定实现
        return torch.sigmoid(x)
    
    @staticmethod
    def gelu_approximation(x: torch.Tensor) -> torch.Tensor:
        """
        GELU 近似实现（适合硬件实现）
        
        Args:
            x: 输入张量
            
        Returns:
            GELU 结果
        """
        # 使用 tanh 近似
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * 
                                        (x + 0.044715 * x ** 3)))
    
    @staticmethod
    def erf_approximation(x: torch.Tensor) -> torch.Tensor:
        """
        误差函数近似实现
        
        Args:
            x: 输入张量
            
        Returns:
            误差函数结果
        """
        # 使用有理函数近似
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911
        
        sign = torch.sign(x)
        x = torch.abs(x)
        
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * torch.exp(-x * x)
        
        return sign * y
    
    @staticmethod
    def gelu_exact(x: torch.Tensor) -> torch.Tensor:
        """
        精确的 GELU 实现
        
        Args:
            x: 输入张量
            
        Returns:
            GELU 结果
        """
        erf_x = MathUtils.erf_approximation(x / math.sqrt(2))
        return 0.5 * x * (1 + erf_x)
    
    @staticmethod
    def check_numerical_stability(x: torch.Tensor, 
                                 operation: str = "unknown") -> bool:
        """
        检查数值稳定性
        
        Args:
            x: 输入张量
            operation: 操作名称
            
        Returns:
            是否稳定
        """
        # 检查 NaN
        if torch.isnan(x).any():
            raise ValueError(f"数值不稳定: {operation} 产生 NaN")
        
        # 检查无穷大
        if torch.isinf(x).any():
            raise ValueError(f"数值不稳定: {operation} 产生无穷大")
        
        # 检查数值范围
        abs_x = torch.abs(x)
        max_val = abs_x.max()
        min_val = abs_x.min()
        
        if max_val > NUMERICAL_STABILITY['max_value']:
            raise NumericalOverflowError(
                max_val.item(), 
                NUMERICAL_STABILITY['max_value']
            )
        
        # 只对非零值检查下溢
        non_zero_mask = x != 0
        if non_zero_mask.any():
            non_zero_abs = abs_x[non_zero_mask]
            min_non_zero = non_zero_abs.min()
            if min_non_zero < NUMERICAL_STABILITY['min_value']:
                raise NumericalUnderflowError(
                    min_non_zero.item(), 
                    NUMERICAL_STABILITY['min_value']
                )
        
        return True
    
    @staticmethod
    def safe_softmax_enhanced(x: torch.Tensor, dim: int = -1, 
                             temperature: float = 1.0) -> torch.Tensor:
        """
        增强的数值稳定 Softmax 实现
        
        Args:
            x: 输入张量
            dim: 计算维度
            temperature: 温度参数
            
        Returns:
            Softmax 结果
        """
        # 应用温度缩放
        x_scaled = x / temperature
        
        # 减去最大值防止溢出
        x_max = torch.max(x_scaled, dim=dim, keepdim=True)[0]
        x_shifted = x_scaled - x_max
        
        # 限制指数范围
        x_shifted = torch.clamp(x_shifted, min=-20, max=20)
        
        # 计算指数
        exp_x = torch.exp(x_shifted)
        
        # 计算和
        sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
        
        # 归一化
        result = exp_x / (sum_exp + EPSILON_TINY)
        
        # 检查数值稳定性
        MathUtils.check_numerical_stability(result, "enhanced_softmax")
        
        return result
    
    @staticmethod
    def convert_dtype(x: torch.Tensor, target_dtype: str) -> torch.Tensor:
        """
        转换数据类型
        
        Args:
            x: 输入张量
            target_dtype: 目标数据类型
            
        Returns:
            转换后的张量
        """
        validate_dtype(target_dtype, ['float32', 'bfloat16'])
        
        if target_dtype == 'float32':
            return x.float()
        elif target_dtype == 'bfloat16':
            return x.bfloat16()
        else:
            raise UnsupportedDataTypeError(target_dtype, ['float32', 'bfloat16'])
    
    @staticmethod
    @lru_cache(maxsize=128)
    def factorial(n: int) -> int:
        """
        计算阶乘（带缓存）
        
        Args:
            n: 输入值
            
        Returns:
            阶乘结果
        """
        if n < 0:
            raise ValueError("阶乘不能计算负数")
        if n == 0 or n == 1:
            return 1
        return n * MathUtils.factorial(n - 1)
    
    @staticmethod
    def taylor_exp(x: torch.Tensor, n_terms: int = 10) -> torch.Tensor:
        """
        泰勒级数近似指数函数
        
        Args:
            x: 输入张量
            n_terms: 级数项数
            
        Returns:
            近似结果
        """
        result = torch.ones_like(x)
        x_power = x.clone()
        
        for i in range(1, n_terms + 1):
            result += x_power / MathUtils.factorial(i)
            x_power *= x
        
        return result
    
    @staticmethod
    def chebyshev_approximation(x: torch.Tensor, 
                               coeffs: List[float],
                               a: float = -1.0, 
                               b: float = 1.0) -> torch.Tensor:
        """
        切比雪夫多项式近似
        
        Args:
            x: 输入张量
            coeffs: 切比雪夫系数
            a: 区间下界
            b: 区间上界
            
        Returns:
            近似结果
        """
        # 将 x 映射到 [-1, 1] 区间
        x_mapped = 2 * (x - a) / (b - a) - 1
        
        # 计算切比雪夫多项式
        result = torch.zeros_like(x)
        T_prev = torch.ones_like(x)
        T_curr = x_mapped
        
        result += coeffs[0] * T_prev
        if len(coeffs) > 1:
            result += coeffs[1] * T_curr
        
        for i in range(2, len(coeffs)):
            T_next = 2 * x_mapped * T_curr - T_prev
            result += coeffs[i] * T_next
            T_prev = T_curr
            T_curr = T_next
        
        return result


# 便捷函数
def safe_exp(x: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
    """安全的指数函数"""
    return MathUtils.safe_exp(x)


def safe_log(x: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
    """安全的对数函数"""
    return MathUtils.safe_log(x)


def safe_sqrt(x: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
    """安全的平方根函数"""
    return MathUtils.safe_sqrt(x)


def safe_divide(a: Union[float, torch.Tensor], 
                b: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
    """安全的除法运算"""
    return MathUtils.safe_divide(a, b)
