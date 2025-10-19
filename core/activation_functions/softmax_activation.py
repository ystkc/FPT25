"""
Softmax 激活函数模块
实现高效的 Softmax 激活函数，支持查找表优化和硬件加速
"""

import traceback
import torch
import math
from typing import Optional, Dict, Any
from dataclasses import dataclass

from core.activation_functions.base_activation import BaseActivationFunction
from core.hardware import create_fixed_point, FixedPointNumber
from core.utils.memory_pool import MemoryContext


from config.config import ActivationConfig, SoftmaxConfig


class SoftmaxActivation(BaseActivationFunction):
    """Softmax 激活函数类"""

    def __init__(self, config: SoftmaxConfig):
        super().__init__(config,
                         table_func=torch.exp,
                         name='Softmax', 
                         table_name='Exponential')

    def _forward_reference(self, original_tensor: torch.Tensor, mem_ctx: MemoryContext) -> torch.Tensor:
        """Softmax 参考答案"""
        return torch.softmax(original_tensor, dim=-1)
    
    def _forward_direct(self, original_tensor: torch.Tensor, mem_ctx: MemoryContext) -> torch.Tensor:
        """直接计算 Softmax"""
        # 按照torch内部逻辑，需要先转换成高精度类型以减少精度损失
        compute_tensor = original_tensor.to(self.config.compute_dtype)
        
        # 不使用查找表的 Softmax 实现
        exp_x = torch.exp(compute_tensor)
        result = exp_x / torch.sum(exp_x, dim=-1, keepdim=True)
        
        # 转换回原始数据类型
        return result.to(original_tensor.dtype)
    
    def _forward_with_lookup_table(self, original_tensor: torch.Tensor, mem_ctx: MemoryContext) -> torch.Tensor:
        """使用查找表计算 Softmax"""
        compute_tensor = original_tensor.to(self.config.compute_dtype)
        
        # 数值稳定性：减去最大值
        x_max = torch.max(compute_tensor, dim=-1, keepdim=True)[0]
        x_shifted = compute_tensor - x_max
        
        # 使用查找表计算指数
        if self.config.use_fixed_point:
            exp_x = self._exp_with_fixed_point(x_shifted)
        else:
            exp_x = self.lookup_table.lookup(x_shifted)
        
        # 计算归一化
        sum_exp = torch.sum(exp_x, dim=-1, keepdim=True)
        result = exp_x / sum_exp
        
        # 转换回原始数据类型
        return result.to(self.config.dtype)
    
    def _exp_with_fixed_point(self, x: torch.Tensor) -> torch.Tensor:
        """使用定点数计算指数"""
        # 转换为定点数
        fixed_x = create_fixed_point(0.0, self.config.fixed_point_format)
        
        # 逐元素计算（这里简化实现）
        result = torch.zeros_like(x)
        for i in range(x.numel()):
            flat_x = x.view(-1)
            flat_result = result.view(-1)
            
            # 创建定点数
            fixed_val = create_fixed_point(flat_x[i].item(), self.config.fixed_point_format)
            
            # 使用查找表
            if self.lookup_table:
                exp_val = self.lookup_table.lookup(fixed_val.to_float())
            else:
                exp_val = math.exp(fixed_val.to_float())
            
            flat_result[i] = exp_val
        
        return result
    
    def backward(self, grad_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Softmax 反向传播
        
        Args:
            grad_output: 输出梯度
            x: 原始输入
            
        Returns:
            输入梯度
        """
        # 计算 Softmax 输出
        softmax_output = self.forward(x)
        
        # 计算梯度
        grad_input = softmax_output * (grad_output - torch.sum(grad_output * softmax_output, dim=-1, keepdim=True))
        
        return grad_input


# 便捷函数
def create_softmax(config: Optional[SoftmaxConfig] = None) -> SoftmaxActivation:
    """创建 Softmax 激活函数"""
    if config is None:
        config = SoftmaxConfig()
    return SoftmaxActivation(config)


def softmax_forward(x: torch.Tensor, 
                   use_lookup_table: bool = True,
                   lookup_bit_len: int = 800,
                   interpolation_method: str = 'linear') -> torch.Tensor:
    """Softmax 前向传播便捷函数"""
    config = SoftmaxConfig(
        use_lookup_table=use_lookup_table,
        lookup_table_bitlen=lookup_bit_len,
        interpolation_method=interpolation_method
    )
    
    softmax = SoftmaxActivation(config)
    return softmax.forward(x)


def softmax_benchmark(input_shapes: list = None, 
                     num_runs: int = 100) -> Dict[str, Any]:
    """Softmax 基准测试便捷函数"""
    softmax = SoftmaxActivation(SoftmaxConfig())
    return softmax.benchmark(input_shapes, num_runs)
