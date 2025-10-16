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


@dataclass
class SiLUConfig(ActivationConfig):
    """SiLU 配置"""
    use_lookup_table: bool = True
    lookup_bit_len: int = 800
    interpolation_method: str = 'linear'


class SiLUActivation(BaseActivationFunction):
    """SiLU 激活函数类"""
    
    def __init__(self, config: SiLUConfig):
        super().__init__(config)
        self.config = config
        
        # 初始化查找表
        self.lookup_table = None
        if self.config.use_lookup_table:
            self._initialize_lookup_table()
    
    def _initialize_lookup_table(self):
        """初始化 Sigmoid 查找表"""
        try:
            from core.algorithms import create_sigmoid_table
            self.lookup_table = create_sigmoid_table(
                name="silu_sigmoid",
                bit_len=self.config.lookup_bit_len,
                interpolation_method=self.config.interpolation_method
            )
            self.logger.info(f"SiLU 查找表初始化完成: {self.config.lookup_bit_len} 点")
        except Exception as e:
            self.logger.warning(f"查找表初始化失败，将使用直接计算: {e}")
            self.lookup_table = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SiLU 前向传播"""
        if self.lookup_table is not None:
            # 使用查找表计算 Sigmoid
            sigmoid_x = self.lookup_table.lookup_sigmoid(x)
        else:
            # 直接计算 Sigmoid
            sigmoid_x = MathUtils.sigmoid_stable(x)
        
        # SiLU(x) = x * sigmoid(x)
        return x * sigmoid_x
    
    def backward(self, grad_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """SiLU 反向传播"""
        # SiLU 的导数: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        if self.lookup_table is not None:
            sigmoid_x = self.lookup_table.lookup_sigmoid(x)
        else:
            sigmoid_x = MathUtils.sigmoid_stable(x)
        
        sigmoid_derivative = sigmoid_x * (1 - sigmoid_x)
        silu_derivative = sigmoid_x + x * sigmoid_derivative
        
        return grad_output * silu_derivative
    
    def get_config(self) -> Dict[str, Any]:
        """获取配置信息"""
        return {
            'use_lookup_table': self.config.use_lookup_table,
            'lookup_bit_len': self.config.lookup_bit_len,
            'interpolation_method': self.config.interpolation_method,
            'dtype': self.config.dtype
        }
