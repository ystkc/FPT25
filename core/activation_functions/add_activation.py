"""
Add 激活函数模块
实现元素级加法激活函数
"""

import torch
from typing import Optional, Dict, Any
from dataclasses import dataclass

from core.activation_functions.activation_manager import BaseActivationFunction, ActivationConfig
from core.base.logs import get_logger


@dataclass
class AddConfig(ActivationConfig):
    """Add 配置"""
    pass


class AddActivation(BaseActivationFunction):
    """Add 激活函数类"""
    
    def __init__(self, config: AddConfig):
        super().__init__(config)
        self.config = config
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add 前向传播"""
        # 对于 Add 激活函数，通常需要两个输入
        # 这里简化实现，假设输入是 [x, y] 的元组
        if isinstance(x, tuple) and len(x) == 2:
            return x[0] + x[1]
        else:
            # 如果只有一个输入，返回自身
            return x
    
    def backward(self, grad_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Add 反向传播"""
        # Add 的梯度是 1
        return grad_output
    
    def get_config(self) -> Dict[str, Any]:
        """获取配置信息"""
        return {
            'dtype': self.config.dtype
        }
