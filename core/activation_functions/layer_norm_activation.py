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



class LayerNormActivation(BaseActivationFunction):
    """LayerNorm 激活函数类"""
    
    def __init__(self, config: LayerNormConfig):
        super().__init__(config)
        self.config = config
        
        # 初始化可学习参数
        if self.config.use_learnable_params:
            self.gamma = torch.nn.Parameter(torch.full((768,), self.config.gamma_init))
            self.beta = torch.nn.Parameter(torch.full((768,), self.config.beta_init))
        else:
            self.gamma = None
            self.beta = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """LayerNorm 前向传播"""
        # 计算均值和方差
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, keepdim=True, unbiased=False)
        
        # 归一化
        x_norm = (x - mean) / torch.sqrt(var + self.config.eps)
        
        # 应用缩放和偏移
        if self.gamma is not None:
            x_norm = x_norm * self.gamma
        if self.beta is not None:
            x_norm = x_norm + self.beta
        
        return x_norm
    
    def backward(self, grad_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """LayerNorm 反向传播"""
        # 简化的反向传播实现
        # 实际应用中需要更复杂的梯度计算
        return grad_output
    
    def get_config(self) -> Dict[str, Any]:
        """获取配置信息"""
        return {
            'eps': self.config.eps,
            'use_learnable_params': self.config.use_learnable_params,
            'gamma_init': self.config.gamma_init,
            'beta_init': self.config.beta_init,
            'dtype': self.config.dtype
        }
