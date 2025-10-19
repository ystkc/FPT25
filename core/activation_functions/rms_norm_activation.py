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



class RMSNormActivation(BaseActivationFunction):
    """RMSNorm 激活函数类"""
    
    def __init__(self, config: RMSNormConfig):
        super().__init__(config)
        self.config = config
        
        # 初始化可学习参数
        if self.config.use_learnable_params:
            self.gamma = torch.nn.Parameter(torch.full((768,), self.config.gamma_init))
        else:
            self.gamma = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """RMSNorm 前向传播"""
        # 计算 RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.config.eps)
        
        # 归一化
        x_norm = x / rms
        
        # 应用缩放
        if self.gamma is not None:
            x_norm = x_norm * self.gamma
        
        return x_norm
    
    def backward(self, grad_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """RMSNorm 反向传播"""
        # 简化的反向传播实现
        return grad_output
    
    def get_config(self) -> Dict[str, Any]:
        """获取配置信息"""
        return {
            'eps': self.config.eps,
            'use_learnable_params': self.config.use_learnable_params,
            'gamma_init': self.config.gamma_init,
            'dtype': self.config.dtype
        }
