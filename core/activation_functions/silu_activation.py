"""
SiLU 激活函数模块
实现高效的 SiLU (Swish) 激活函数
"""

import torch
import math
from typing import Optional, Dict, Any
from dataclasses import dataclass

from core.activation_functions.activation_manager import BaseActivationFunction, ActivationConfig
from core.algorithms import MathUtils
from core.base.logs import get_logger

from config.config import SiLUConfig
from core.utils.memory_pool import MemoryContext

class SiLUActivation(BaseActivationFunction):
    """SiLU 激活函数类"""
    
    def __init__(self, config: SiLUConfig):
        super().__init__(config,
                         table_func=torch.sigmoid,
                         name="SiLU",
                         default_table_name="sigmoid")
        self.config = config
        
    def _forward_direct(self, original_tensor: torch.Tensor, mem_ctx: MemoryContext) -> torch.Tensor:
        """SiLU 前向传播(直接计算)"""
        # 提高精度
        compute_tensor = original_tensor.to(dtype=self.config.compute_dtype)

        sigmoid_x = torch.sigmoid(compute_tensor)
        
        # SiLU(x) = x * sigmoid(x)
        return (compute_tensor * sigmoid_x).to(dtype=self.config.dtype)
    
    def _forward_reference(self, original_tensor: torch.Tensor, mem_ctx: MemoryContext) -> torch.Tensor:
        """SiLU 前向传播(参考实现)"""
        silu = torch.nn.SiLU()
        return silu(original_tensor)
    
    def _forward_with_lookup_table(self, original_tensor: torch.Tensor, mem_ctx: MemoryContext) -> torch.Tensor:
        """SiLU 前向传播(使用查找表)"""
        # 提高精度
        compute_tensor = original_tensor.to(dtype=self.config.compute_dtype)
        # 计算 sigmoid(x)
        sigmoid_x = self.lookup_table.lookup(compute_tensor)

        # SiLU(x) = x * sigmoid(x)
        return (compute_tensor * sigmoid_x).to(dtype=self.config.dtype)
    
    def backward(self, grad_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """SiLU 反向传播"""
        # SiLU 的导数: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        if self.lookup_table is not None:
            sigmoid_x = self.lookup_table.lookup(x)
        else:
            sigmoid_x = MathUtils.sigmoid_stable(x)
        
        sigmoid_derivative = sigmoid_x * (1 - sigmoid_x)
        silu_derivative = sigmoid_x + x * sigmoid_derivative
        
        return grad_output * silu_derivative
    