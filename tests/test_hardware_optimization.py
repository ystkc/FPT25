"""
硬件优化测试模块
提供硬件优化功能测试
"""

import pytest
import torch
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.hardware import create_fixed_point, create_fixed_point_tensor, FixedPointConfig
from core.algorithms import create_exp_table, create_sigmoid_table
from core.optimization import get_memory_optimizer, get_parallel_processor


class TestHardwareOptimization:
    """硬件优化测试类"""
    
    def setup_method(self):
        """测试前准备"""
        self.input_tensor = torch.randn(64, 768, dtype=torch.bfloat16)
    
    def test_fixed_point_conversion(self):
        """测试定点数转换"""
        # 测试标量转换
        fixed_point = create_fixed_point(3.14159, 'Q16_16')
        assert isinstance(fixed_point, type(create_fixed_point(0.0, 'Q16_16')))
        
        # 测试转换回浮点数
        converted_back = fixed_point.to_float()
        assert abs(converted_back - 3.14159) < 1e-4
    
    def test_fixed_point_arithmetic(self):
        """测试定点数算术运算"""
        a = create_fixed_point(2.5, 'Q16_16')
        b = create_fixed_point(1.5, 'Q16_16')
        
        # 测试加法
        c = a + b
        assert abs(c.to_float() - 4.0) < 1e-4
        
        # 测试乘法
        d = a * b
        assert abs(d.to_float() - 3.75) < 1e-4
    
    def test_fixed_point_tensor_conversion(self):
        """测试定点数张量转换"""
        fixed_tensor = create_fixed_point_tensor(self.input_tensor, 'Q16_16')
        
        # 测试转换回浮点张量
        converted_back = fixed_tensor.to_float_tensor()
        assert converted_back.shape == self.input_tensor.shape
        assert converted_back.dtype == self.input_tensor.dtype
    
    def test_lookup_table_generation(self):
        """测试查找表生成"""
        # 测试指数查找表
        exp_table = create_exp_table(table_name="test_exp", bit_len=100)
        assert exp_table is not None
        
        # 测试 Sigmoid 查找表
        sigmoid_table = create_sigmoid_table(name="test_sigmoid", bit_len=100)
        assert sigmoid_table is not None
    
    def test_memory_optimization(self):
        """测试内存优化"""
        memory_optimizer = get_memory_optimizer()
        
        # 测试内存统计
        stats = memory_optimizer.get_memory_stats()
        assert 'total_memory' in stats
        assert 'used_memory' in stats
        assert stats['used_memory'] > 0
    
    def test_parallel_processing(self):
        """测试并行处理"""
        parallel_processor = get_parallel_processor()
        
        # 测试并行映射
        def square(x):
            return x ** 2
        
        data = list(range(10))
        results = parallel_processor.parallel_map(square, data)
        
        assert len(results) == len(data)
        assert all(r == x ** 2 for r, x in zip(results, data))


if __name__ == '__main__':
    pytest.main([__file__])
