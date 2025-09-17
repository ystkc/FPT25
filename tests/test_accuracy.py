"""
精度测试模块
提供精度评估测试功能
"""

import pytest
import torch
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation import get_accuracy_evaluator, get_reference_generator
from core.base.constants import TENSOR_SHAPE, DEFAULT_DTYPE


class TestAccuracyEvaluation:
    """精度评估测试类"""
    
    def setup_method(self):
        """测试前准备"""
        self.accuracy_evaluator = get_accuracy_evaluator()
        self.reference_generator = get_reference_generator()
        self.input_tensor = torch.randn(TENSOR_SHAPE, dtype=getattr(torch, DEFAULT_DTYPE))
    
    def test_relative_l2_error_calculation(self):
        """测试相对 L2 误差计算"""
        # 创建测试数据
        fpga_output = torch.randn(TENSOR_SHAPE, dtype=getattr(torch, DEFAULT_DTYPE))
        reference_output = torch.randn(TENSOR_SHAPE, dtype=getattr(torch, DEFAULT_DTYPE))
        
        # 计算相对 L2 误差
        error = self.accuracy_evaluator.calculate_relative_l2_error(fpga_output, reference_output)
        
        # 验证误差为非负数
        assert error >= 0
    
    def test_accuracy_score_calculation(self):
        """测试精度评分计算"""
        # 测试满分情况
        perfect_score = self.accuracy_evaluator.calculate_accuracy_score(1e-4)
        assert perfect_score == 1.0
        
        # 测试零分情况
        zero_score = self.accuracy_evaluator.calculate_accuracy_score(1e-1)
        assert zero_score == 0.0
        
        # 测试中间情况
        middle_score = self.accuracy_evaluator.calculate_accuracy_score(1e-2)
        assert 0 < middle_score < 1
    
    def test_softmax_accuracy_evaluation(self):
        """测试 Softmax 精度评估"""
        # 生成参考结果
        reference_output = self.reference_generator.generate_softmax_reference(self.input_tensor)
        
        # 模拟 FPGA 输出（添加小误差）
        fpga_output = reference_output + torch.randn_like(reference_output) * 1e-4
        
        # 评估精度
        result = self.accuracy_evaluator.evaluate_activation_function(
            'softmax', fpga_output, reference_output
        )
        
        # 验证结果
        assert result['function_name'] == 'softmax'
        assert result['weight'] == 15  # Softmax 权重
        assert 0 <= result['metrics'].accuracy_score <= 1
        assert result['metrics'].passed_threshold is not None
    
    def test_layer_norm_accuracy_evaluation(self):
        """测试 LayerNorm 精度评估"""
        # 生成参考结果
        reference_output = self.reference_generator.generate_layer_norm_reference(self.input_tensor)
        
        # 模拟 FPGA 输出
        fpga_output = reference_output + torch.randn_like(reference_output) * 1e-4
        
        # 评估精度
        result = self.accuracy_evaluator.evaluate_activation_function(
            'layer_norm', fpga_output, reference_output
        )
        
        # 验证结果
        assert result['function_name'] == 'layer_norm'
        assert result['weight'] == 12  # LayerNorm 权重
        assert 0 <= result['metrics'].accuracy_score <= 1


if __name__ == '__main__':
    pytest.main([__file__])
