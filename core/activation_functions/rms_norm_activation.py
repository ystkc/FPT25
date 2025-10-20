"""
RMSNorm 激活函数模块
实现高效的 RMSNorm 激活函数
"""

import torch
import math
from typing import Optional, Dict, Any
from dataclasses import dataclass

from core.activation_functions.activation_manager import BaseActivationFunction, ActivationConfig
from core.base.constants import EPSILON_TINY
from core.algorithms import MathUtils
from core.base.logs import get_logger

from config.config import RMSNormConfig
from core.utils.memory_pool import MemoryContext



class RMSNormActivation(BaseActivationFunction):
    """RMSNorm 激活函数类"""
    
    def __init__(self, config: RMSNormConfig):
        super().__init__(config,
                         table_func=torch.sqrt,
                         name='rms_norm',
                         default_table_name='sqrt')
        self.config = config
        
        # 初始化可学习参数
        if self.config.use_learnable_params:
            self.gamma = torch.nn.Parameter(torch.full((768,), self.config.gamma_init))
        else:
            self.gamma = None
    
    def _forward_direct(self, original_tensor: torch.Tensor, mem_ctx: MemoryContext) -> torch.Tensor:
        """RMSNorm 前向传播"""
        # 提高精度
        compute_tensor = original_tensor.to(self.config.compute_dtype)
        # 计算 RMS
        rms = torch.sqrt(torch.mean(compute_tensor ** 2, dim=-1, keepdim=True) + self.config.eps)
        
        # 归一化
        x_norm = compute_tensor / rms
        
        # 应用缩放
        if self.gamma is not None:
            x_norm = x_norm * self.gamma
        
        return x_norm.to(original_tensor.dtype)
    
    def _forward_with_lookup_table(self, original_tensor: torch.Tensor, mem_ctx: MemoryContext) -> torch.Tensor:
        """RMSNorm 前向传播(带查找表)"""
        # 提高精度
        compute_tensor = original_tensor.to(self.config.compute_dtype)
        # 计算 RMS
        rms = self.lookup_table.lookup(torch.mean(compute_tensor ** 2, dim=-1, keepdim=True) + self.config.eps)

        # 归一化
        x_norm = compute_tensor / rms
        
        # 应用缩放
        if self.gamma is not None:
            x_norm = x_norm * self.gamma
        
        return x_norm.to(original_tensor.dtype)
    
    def _forward_reference(self, original_tensor: torch.Tensor, mem_ctx: MemoryContext) -> torch.Tensor:
        """RMSNorm 参考实现(torch库)"""
        rms_norm = torch.nn.RMSNorm(normalized_shape=(768,), eps=self.config.eps)
        return rms_norm(original_tensor)
    
    def backward(self, grad_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """RMSNorm 反向传播"""
        # 简化的反向传播实现
        return grad_output