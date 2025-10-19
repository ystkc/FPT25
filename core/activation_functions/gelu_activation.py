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



class GELUActivation(BaseActivationFunction):
    """GELU 激活函数类"""
    
    def __init__(self, config: GELUConfig):
        super().__init__(config)
        self.config = config
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """GELU 前向传播"""
        if self.config.use_approximation:
            if self.config.approximation_type == 'tanh':
                # 使用 tanh 近似
                return MathUtils.gelu_approximation(x)
            else:
                # 使用精确实现
                return MathUtils.gelu_exact(x)
        else:
            # 使用 PyTorch 标准实现
            return torch.nn.functional.gelu(x)
    
    def backward(self, grad_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """GELU 反向传播"""
        # 简化的反向传播实现
        # 实际应用中需要更复杂的梯度计算
        return grad_output
    
    def get_config(self) -> Dict[str, Any]:
        """获取配置信息"""
        return {
            'use_approximation': self.config.use_approximation,
            'approximation_type': self.config.approximation_type,
            'dtype': self.config.dtype
        }
