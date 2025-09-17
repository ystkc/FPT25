"""
精度评估模块
实现竞赛标准的精度评估功能
"""

import torch
import math
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass

from core.base.constants import EPSILON_STAR, EPSILON_MAX, EPSILON_TINY, ACTIVATION_FUNCTION_WEIGHTS
from core.base.exceptions import AccuracyError, AccuracyThresholdExceededError
from core.base.logs import get_logger, get_accuracy_logger


@dataclass
class AccuracyMetrics:
    """精度指标"""
    l2_error: float
    relative_l2_error: float
    mse: float
    mae: float
    max_error: float
    min_error: float
    accuracy_score: float
    passed_threshold: bool


class AccuracyEvaluator:
    """精度评估器"""
    
    def __init__(self):
        self.logger = get_logger()
        self.accuracy_logger = get_accuracy_logger()
    
    def calculate_relative_l2_error(self, fpga_output: torch.Tensor, 
                                  reference_output: torch.Tensor) -> float:
        """
        计算相对 L2 误差（竞赛标准）
        
        Args:
            fpga_output: FPGA 输出张量
            reference_output: 参考输出张量
            
        Returns:
            相对 L2 误差
        """
        # 确保张量形状一致
        if fpga_output.shape != reference_output.shape:
            raise ValueError(f"张量形状不匹配: {fpga_output.shape} vs {reference_output.shape}")
        
        # 计算 L2 范数
        numerator = torch.norm(fpga_output - reference_output, p=2)
        denominator = torch.norm(reference_output, p=2) + EPSILON_TINY
        
        relative_l2_error = numerator / denominator
        
        return float(relative_l2_error)
    
    def calculate_accuracy_score(self, relative_l2_error: float) -> float:
        """
        计算精度评分（竞赛标准）
        
        Args:
            relative_l2_error: 相对 L2 误差
            
        Returns:
            精度评分 (0-1)
        """
        if relative_l2_error <= EPSILON_STAR:
            # 误差在阈值内，得满分
            return 1.0
        elif relative_l2_error <= EPSILON_MAX:
            # 误差在阈值和最大值之间，对数衰减
            return (math.log(EPSILON_MAX) - math.log(relative_l2_error)) / math.log(100)
        else:
            # 误差超过最大值，得零分
            return 0.0
    
    def evaluate_accuracy(self, fpga_output: torch.Tensor, 
                         reference_output: torch.Tensor) -> AccuracyMetrics:
        """
        评估精度
        
        Args:
            fpga_output: FPGA 输出张量
            reference_output: 参考输出张量
            
        Returns:
            精度指标
        """
        # 计算各种误差指标
        l2_error = torch.norm(fpga_output - reference_output, p=2).item()
        relative_l2_error = self.calculate_relative_l2_error(fpga_output, reference_output)
        
        mse = torch.mean((fpga_output - reference_output) ** 2).item()
        mae = torch.mean(torch.abs(fpga_output - reference_output)).item()
        
        max_error = torch.max(torch.abs(fpga_output - reference_output)).item()
        min_error = torch.min(torch.abs(fpga_output - reference_output)).item()
        
        # 计算精度评分
        accuracy_score = self.calculate_accuracy_score(relative_l2_error)
        
        # 检查是否通过阈值
        passed_threshold = relative_l2_error <= EPSILON_STAR
        
        metrics = AccuracyMetrics(
            l2_error=l2_error,
            relative_l2_error=relative_l2_error,
            mse=mse,
            mae=mae,
            max_error=max_error,
            min_error=min_error,
            accuracy_score=accuracy_score,
            passed_threshold=passed_threshold
        )
        
        # 记录日志
        self.accuracy_logger.log_accuracy_test(
            "activation_function", relative_l2_error, EPSILON_STAR, passed_threshold
        )
        
        return metrics
    
    def evaluate_activation_function(self, function_name: str,
                                   fpga_output: torch.Tensor,
                                   reference_output: torch.Tensor) -> Dict[str, Any]:
        """
        评估激活函数精度
        
        Args:
            function_name: 激活函数名称
            fpga_output: FPGA 输出张量
            reference_output: 参考输出张量
            
        Returns:
            评估结果
        """
        metrics = self.evaluate_accuracy(fpga_output, reference_output)
        
        # 获取函数权重
        weight = ACTIVATION_FUNCTION_WEIGHTS.get(function_name, 1)
        
        # 计算加权评分
        weighted_score = metrics.accuracy_score * weight
        
        result = {
            'function_name': function_name,
            'weight': weight,
            'metrics': metrics,
            'weighted_score': weighted_score,
            'evaluation_time': torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        }
        
        self.logger.info(f"激活函数 {function_name} 精度评估完成: "
                        f"相对L2误差={metrics.relative_l2_error:.2e}, "
                        f"精度评分={metrics.accuracy_score:.4f}, "
                        f"加权评分={weighted_score:.4f}")
        
        return result
    
    def batch_evaluate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        批量评估
        
        Args:
            results: 评估结果列表
            
        Returns:
            批量评估结果
        """
        total_weighted_score = 0.0
        total_weight = 0.0
        function_results = {}
        
        for result in results:
            function_name = result['function_name']
            weighted_score = result['weighted_score']
            weight = result['weight']
            
            total_weighted_score += weighted_score
            total_weight += weight
            
            function_results[function_name] = {
                'accuracy_score': result['metrics'].accuracy_score,
                'weighted_score': weighted_score,
                'relative_l2_error': result['metrics'].relative_l2_error,
                'passed_threshold': result['metrics'].passed_threshold
            }
        
        # 计算总体评分
        overall_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
        
        batch_result = {
            'overall_score': overall_score,
            'total_weighted_score': total_weighted_score,
            'total_weight': total_weight,
            'function_results': function_results,
            'evaluation_summary': self._generate_evaluation_summary(function_results)
        }
        
        self.logger.info(f"批量评估完成: 总体评分={overall_score:.4f}")
        
        return batch_result
    
    def _generate_evaluation_summary(self, function_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成评估摘要"""
        passed_functions = [name for name, result in function_results.items() 
                          if result['passed_threshold']]
        failed_functions = [name for name, result in function_results.items() 
                          if not result['passed_threshold']]
        
        avg_accuracy = sum(result['accuracy_score'] for result in function_results.values()) / len(function_results)
        avg_error = sum(result['relative_l2_error'] for result in function_results.values()) / len(function_results)
        
        return {
            'total_functions': len(function_results),
            'passed_functions': len(passed_functions),
            'failed_functions': len(failed_functions),
            'passed_function_names': passed_functions,
            'failed_function_names': failed_functions,
            'average_accuracy_score': avg_accuracy,
            'average_relative_l2_error': avg_error,
            'pass_rate': len(passed_functions) / len(function_results) if function_results else 0.0
        }


class ReferenceGenerator:
    """参考结果生成器"""
    
    def __init__(self):
        self.logger = get_logger()
    
    def generate_softmax_reference(self, x: torch.Tensor) -> torch.Tensor:
        """生成 Softmax 参考结果"""
        return torch.softmax(x, dim=-1)
    
    def generate_layer_norm_reference(self, x: torch.Tensor, 
                                    gamma: Optional[torch.Tensor] = None,
                                    beta: Optional[torch.Tensor] = None,
                                    eps: float = 1e-5) -> torch.Tensor:
        """生成 LayerNorm 参考结果"""
        return torch.nn.functional.layer_norm(x, x.shape[-1:], gamma, beta, eps)
    
    def generate_rms_norm_reference(self, x: torch.Tensor,
                                  gamma: Optional[torch.Tensor] = None,
                                  eps: float = 1e-5) -> torch.Tensor:
        """生成 RMSNorm 参考结果"""
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
        x_norm = x / rms
        if gamma is not None:
            x_norm = x_norm * gamma
        return x_norm
    
    def generate_silu_reference(self, x: torch.Tensor) -> torch.Tensor:
        """生成 SiLU 参考结果"""
        return x * torch.sigmoid(x)
    
    def generate_gelu_reference(self, x: torch.Tensor) -> torch.Tensor:
        """生成 GELU 参考结果"""
        return torch.nn.functional.gelu(x)
    
    def generate_add_reference(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """生成 Add 参考结果"""
        return x + y
    
    def generate_multiply_reference(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """生成 Multiply 参考结果"""
        return x * y


class AccuracyTestSuite:
    """精度测试套件"""
    
    def __init__(self):
        self.logger = get_logger()
        self.evaluator = AccuracyEvaluator()
        self.reference_generator = ReferenceGenerator()
    
    def run_comprehensive_test(self, input_tensor: torch.Tensor) -> Dict[str, Any]:
        """运行综合精度测试"""
        results = {}
        
        # 测试 Softmax
        try:
            fpga_softmax = self._test_softmax(input_tensor)
            reference_softmax = self.reference_generator.generate_softmax_reference(input_tensor)
            results['softmax'] = self.evaluator.evaluate_activation_function(
                'softmax', fpga_softmax, reference_softmax
            )
        except Exception as e:
            self.logger.error(f"Softmax 测试失败: {e}")
            results['softmax'] = {'error': str(e)}
        
        # 测试其他激活函数...
        # 这里可以添加更多测试
        
        return results
    
    def _test_softmax(self, x: torch.Tensor) -> torch.Tensor:
        """测试 Softmax"""
        from ..core.activation_functions import create_softmax
        softmax = create_softmax()
        return softmax.forward(x)
    
    def run_stress_test(self, num_tests: int = 100) -> Dict[str, Any]:
        """运行压力测试"""
        results = {
            'num_tests': num_tests,
            'test_results': [],
            'summary': {}
        }
        
        for i in range(num_tests):
            # 生成随机测试数据
            x = torch.randn(64, 768, dtype=torch.bfloat16)
            
            try:
                test_result = self.run_comprehensive_test(x)
                results['test_results'].append({
                    'test_id': i,
                    'success': True,
                    'result': test_result
                })
            except Exception as e:
                results['test_results'].append({
                    'test_id': i,
                    'success': False,
                    'error': str(e)
                })
        
        # 计算摘要
        successful_tests = [r for r in results['test_results'] if r['success']]
        results['summary'] = {
            'successful_tests': len(successful_tests),
            'failed_tests': num_tests - len(successful_tests),
            'success_rate': len(successful_tests) / num_tests
        }
        
        return results


# 全局评估器
_accuracy_evaluator: Optional[AccuracyEvaluator] = None
_reference_generator: Optional[ReferenceGenerator] = None
_test_suite: Optional[AccuracyTestSuite] = None


def get_accuracy_evaluator() -> AccuracyEvaluator:
    """获取全局精度评估器"""
    global _accuracy_evaluator
    if _accuracy_evaluator is None:
        _accuracy_evaluator = AccuracyEvaluator()
    return _accuracy_evaluator


def get_reference_generator() -> ReferenceGenerator:
    """获取全局参考结果生成器"""
    global _reference_generator
    if _reference_generator is None:
        _reference_generator = ReferenceGenerator()
    return _reference_generator


def get_test_suite() -> AccuracyTestSuite:
    """获取全局测试套件"""
    global _test_suite
    if _test_suite is None:
        _test_suite = AccuracyTestSuite()
    return _test_suite


def evaluate_accuracy(fpga_output: torch.Tensor, 
                     reference_output: torch.Tensor) -> AccuracyMetrics:
    """精度评估便捷函数"""
    evaluator = get_accuracy_evaluator()
    return evaluator.evaluate_accuracy(fpga_output, reference_output)
