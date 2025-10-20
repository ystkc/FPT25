"""
GELU 激活函数模块
实现高效的 GELU 激活函数
"""

import torch
import math
from typing import Optional, Dict, Any
from dataclasses import dataclass

from core.activation_functions.activation_manager import BaseActivationFunction, ActivationConfig
from core.algorithms import MathUtils
from core.base.logs import get_logger

from config.config import GELUConfig
from core.utils.memory_pool import MemoryContext



class GELUActivation(BaseActivationFunction):
    """GELU 激活函数类"""
    
    def __init__(self, config: GELUConfig):
        super().__init__(config,
                         table_func=torch.nn.GELU(config.approximation_type),
                         name='GELU',
                         default_table_name='GELU')
        self.config = config
    
    def _forward_reference(self, original_tensor: torch.Tensor, mem_ctx: MemoryContext) -> torch.Tensor:
        """GELU 前向传播(参考实现)"""
        return self.table_func(original_tensor)
    
    def _forward_direct(self, original_tensor: torch.Tensor, mem_ctx: MemoryContext) -> torch.Tensor:
        """GELU 前向传播(直接实现)"""
        compute_tensor = original_tensor.to(dtype=self.config.compute_dtype)
        return (compute_tensor * 0.5 * (1.0 + torch.erf(compute_tensor / math.sqrt(2.0)))).to(dtype=self.config.dtype)
    
    def _forward_with_lookup_table(self, original_tensor: torch.Tensor, mem_ctx: MemoryContext) -> torch.Tensor:
        """GELU 前向传播(使用查找表)"""
        return self.lookup_table.lookup(original_tensor.to(dtype=self.config.compute_dtype)).to(dtype=self.config.dtype)
    
    def backward(self, grad_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """GELU 反向传播"""
        # 简化的反向传播实现
        # 实际应用中需要更复杂的梯度计算
        return grad_output
    