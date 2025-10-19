"""
Multiply 激活函数模块
实现元素级乘法激活函数
"""

import torch
from typing import Optional, Dict, Any
from dataclasses import dataclass

from core.activation_functions.activation_manager import BaseActivationFunction, ActivationConfig
from core.base.logs import get_logger


from config.config import MultiplyConfig


class MultiplyActivation(BaseActivationFunction):
    """Multiply 激活函数类"""
    
    def __init__(self, config: MultiplyConfig):
        super().__init__(config)
        self.config = config
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Multiply 前向传播"""
        # 对于 Multiply 激活函数，通常需要两个输入
        # 这里简化实现，假设输入是 [x, y] 的元组
        if isinstance(x, tuple) and len(x) == 2:
            return x[0] * x[1]
        else:
            # 如果只有一个输入，返回自身
            return x
    
    def backward(self, grad_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Multiply 反向传播"""
        # Multiply 的梯度需要根据输入计算
        if isinstance(x, tuple) and len(x) == 2:
            # 对于 x * y，梯度是 [y, x]
            return (grad_output * x[1], grad_output * x[0])
        else:
            return grad_output
    
    def get_config(self) -> Dict[str, Any]:
        """获取配置信息"""
        return {
            'dtype': self.config.dtype
        }
