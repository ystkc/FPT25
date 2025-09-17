"""
激活函数测试模块
提供7种激活函数的功能测试
"""

import pytest
import torch
import numpy as np
from typing import Dict, Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.activation_functions import get_activation_manager, create_activation_function
from core.base.constants import TENSOR_SHAPE, DEFAULT_DTYPE


class TestActivationFunctions:
    """激活函数测试类"""
    
    def setup_method(self):
        """测试前准备"""
        self.activation_manager = get_activation_manager()
        self.input_tensor = torch.randn(TENSOR_SHAPE, dtype=getattr(torch, DEFAULT_DTYPE))
    
    def test_softmax_activation(self):
        """测试 Softmax 激活函数"""
        softmax = create_activation_function('softmax')
        
        # 测试前向传播
        output = softmax.forward(self.input_tensor)
        
        # 验证输出形状
        assert output.shape == self.input_tensor.shape
        
        # 验证输出范围
        assert torch.all(output >= 0)
        assert torch.all(output <= 1)
        
        # 验证归一化
        row_sums = torch.sum(output, dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)
    
    def test_layer_norm_activation(self):
        """测试 LayerNorm 激活函数"""
        layer_norm = create_activation_function('layer_norm')
        
        # 测试前向传播
        output = layer_norm.forward(self.input_tensor)
        
        # 验证输出形状
        assert output.shape == self.input_tensor.shape
    
    def test_rms_norm_activation(self):
        """测试 RMSNorm 激活函数"""
        rms_norm = create_activation_function('rms_norm')
        
        # 测试前向传播
        output = rms_norm.forward(self.input_tensor)
        
        # 验证输出形状
        assert output.shape == self.input_tensor.shape
    
    def test_silu_activation(self):
        """测试 SiLU 激活函数"""
        silu = create_activation_function('silu')
        
        # 测试前向传播
        output = silu.forward(self.input_tensor)
        
        # 验证输出形状
        assert output.shape == self.input_tensor.shape
    
    def test_gelu_activation(self):
        """测试 GELU 激活函数"""
        gelu = create_activation_function('gelu')
        
        # 测试前向传播
        output = gelu.forward(self.input_tensor)
        
        # 验证输出形状
        assert output.shape == self.input_tensor.shape
    
    def test_add_activation(self):
        """测试 Add 激活函数"""
        add = create_activation_function('add')
        
        # 创建第二个输入
        input2 = torch.randn_like(self.input_tensor)
        
        # 测试前向传播
        output = add.forward((self.input_tensor, input2))
        
        # 验证输出形状
        assert output.shape == self.input_tensor.shape
        
        # 验证加法结果
        expected = self.input_tensor + input2
        assert torch.allclose(output, expected)
    
    def test_multiply_activation(self):
        """测试 Multiply 激活函数"""
        multiply = create_activation_function('multiply')
        
        # 创建第二个输入
        input2 = torch.randn_like(self.input_tensor)
        
        # 测试前向传播
        output = multiply.forward((self.input_tensor, input2))
        
        # 验证输出形状
        assert output.shape == self.input_tensor.shape
        
        # 验证乘法结果
        expected = self.input_tensor * input2
        assert torch.allclose(output, expected)
    
    def test_all_activation_functions(self):
        """测试所有激活函数"""
        functions = ['softmax', 'layer_norm', 'rms_norm', 'silu', 'gelu', 'add', 'multiply']
        
        for func_name in functions:
            try:
                func = create_activation_function(func_name)
                
                if func_name in ['add', 'multiply']:
                    # 二元操作需要两个输入
                    input2 = torch.randn_like(self.input_tensor)
                    output = func.forward((self.input_tensor, input2))
                else:
                    # 一元操作
                    output = func.forward(self.input_tensor)
                
                # 验证输出形状
                assert output.shape == self.input_tensor.shape
                
            except Exception as e:
                pytest.fail(f"激活函数 {func_name} 测试失败: {e}")


if __name__ == '__main__':
    pytest.main([__file__])
