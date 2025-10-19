"""
LayerNorm 激活函数模块
实现高效的 LayerNorm 激活函数
"""

import torch
import math
from typing import Optional, Dict, Any
from dataclasses import dataclass

from core.activation_functions.activation_manager import BaseActivationFunction, ActivationConfig
from core.base.constants import EPSILON_TINY
from core.algorithms import MathUtils
from core.base.logs import get_logger

from config.config import LayerNormConfig
from core.utils.memory_pool import MemoryContext



class LayerNormActivation(BaseActivationFunction):
    """LayerNorm 激活函数类"""
    
    def __init__(self, config: LayerNormConfig):
        super().__init__(config,
                         table_func=torch.sqrt,
                         name='layer_norm',
                         table_name='sqrt')
        self.config = config
        
        # 初始化可学习参数
        if self.config.use_learnable_params:
            self.gamma = torch.nn.Parameter(torch.full((768,), self.config.gamma_init))
            self.beta = torch.nn.Parameter(torch.full((768,), self.config.beta_init))
        else:
            self.gamma = None
            self.beta = None
    
    def _forward_reference(self, original_tensor: torch.Tensor, mem_ctx: MemoryContext) -> torch.Tensor:
        """LayerNorm 前向传播，使用torch库"""
        return torch.nn.functional.layer_norm(original_tensor, normalized_shape=(768,), eps=self.config.eps, weight=self.gamma, bias=self.beta)
        

    def _forward_direct(self, original_tensor: torch.Tensor, mem_ctx: MemoryContext) -> torch.Tensor:
        """LayerNorm 前向传播，使用直接计算"""
        # 转换成更高精度的向量
        compute_tensor = original_tensor.to(self.config.compute_dtype)

        # 计算均值和方差
        mean = torch.mean(compute_tensor, dim=-1, keepdim=True)
        var = torch.var(compute_tensor, dim=-1, keepdim=True, unbiased=False)
        
        # 归一化
        x_norm = (compute_tensor - mean) / torch.sqrt(var + self.config.eps)
        
        # 应用缩放和偏移
        if self.gamma is not None:
            x_norm = x_norm * self.gamma
        if self.beta is not None:
            x_norm = x_norm + self.beta
        
        return x_norm.to(original_tensor.dtype)
    
    def _forward_with_lookup_table(self, original_tensor: torch.Tensor, mem_ctx: MemoryContext) -> torch.Tensor:
        """LayerNorm 前向传播，使用查找表(sqrt函数)"""
        # 转换成更高精度的向量
        compute_tensor = original_tensor.to(self.config.compute_dtype)

        # 计算均值和方差
        mean = torch.mean(compute_tensor, dim=-1, keepdim=True)
        var = torch.var(compute_tensor, dim=-1, keepdim=True, unbiased=False)
        
        # 归一化
        x_norm = (compute_tensor - mean) / self.lookup_table.lookup(var + self.config.eps)

        # 应用缩放和偏移
        if self.gamma is not None:
            x_norm = x_norm * self.gamma
        if self.beta is not None:
            x_norm = x_norm + self.beta
        
        return x_norm.to(original_tensor.dtype)

    
    def backward(self, grad_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """LayerNorm 反向传播"""
        # 简化的反向传播实现
        # 实际应用中需要更复杂的梯度计算
        return grad_output
