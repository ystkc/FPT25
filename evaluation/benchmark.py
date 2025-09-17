"""
性能基准测试模块
实现激活函数的性能基准测试功能
"""

import time
import math
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass
from contextlib import contextmanager

from core.base.constants import (
    TENSOR_SHAPE, BATCH_SIZE, BENCHMARK_CONFIG, 
    ACTIVATION_FUNCTION_WEIGHTS, ACTIVATION_FUNCTIONS
)
from core.base.exceptions import PerformanceError, BenchmarkError
from core.base.logs import get_logger, get_performance_logger
from core.optimization import get_performance_profiler, performance_monitoring


@dataclass
class BenchmarkConfig:
    """基准测试配置"""
    warmup_runs: int = BENCHMARK_CONFIG['warmup_runs']
    measurement_runs: int = BENCHMARK_CONFIG['measurement_runs']
    timeout_seconds: int = BENCHMARK_CONFIG['timeout_seconds']
    memory_limit_mb: int = BENCHMARK_CONFIG['memory_limit_mb']
    input_shapes: List[Tuple[int, int]] = None
    batch_sizes: List[int] = None
    dtypes: List[str] = None


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    function_name: str
    input_shape: Tuple[int, int]
    batch_size: int
    dtype: str
    execution_times: List[float]
    average_time: float
    min_time: float
    max_time: float
    std_time: float
    throughput: float
    memory_usage: int
    peak_memory: int
    accuracy_score: float = 0.0
    success: bool = True
    error_message: Optional[str] = None


class BenchmarkRunner:
    """基准测试运行器"""
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.logger = get_logger()
        self.performance_logger = get_performance_logger()
        self.profiler = get_performance_profiler()
        
        # 设置默认值
        if self.config.input_shapes is None:
            self.config.input_shapes = [TENSOR_SHAPE, (32, 128), (16, 256)]
        if self.config.batch_sizes is None:
            self.config.batch_sizes = [1, 4, 8, 16]
        if self.config.dtypes is None:
            self.config.dtypes = ['float32', 'bfloat16']
    
    def benchmark_function(self, function: Callable, 
                         function_name: str,
                         input_shape: Tuple[int, int],
                         batch_size: int = 1,
                         dtype: str = 'float32') -> BenchmarkResult:
        """
        基准测试单个函数
        
        Args:
            function: 要测试的函数
            function_name: 函数名称
            input_shape: 输入张量形状
            batch_size: 批处理大小
            dtype: 数据类型
            
        Returns:
            基准测试结果
        """
        self.logger.info(f"开始基准测试: {function_name}, 形状: {input_shape}, 批次: {batch_size}, 类型: {dtype}")
        
        try:
            # 创建测试数据
            test_data = self._create_test_data(input_shape, batch_size, dtype)
            
            # 预热运行
            self._warmup_runs(function, test_data)
            
            # 测量运行
            execution_times = self._measurement_runs(function, test_data)
            
            # 计算统计信息
            stats = self._calculate_statistics(execution_times)
            
            # 测量内存使用
            memory_stats = self._measure_memory_usage(function, test_data)
            
            # 计算精度（与参考实现对比）
            # 如果函数有benchmark方法，从结果中获取精度分数
            if hasattr(function, 'benchmark'):
                try:
                    benchmark_result = function.benchmark([input_shape], 1)
                    if benchmark_result and 'results' in benchmark_result and benchmark_result['results']:
                        accuracy_score = benchmark_result['results'][0].get('accuracy_score', 0.0)
                    else:
                        accuracy_score = 0.0
                except Exception as e:
                    self.logger.warning(f"无法从benchmark方法获取精度: {e}")
                    accuracy_score = 0.0
            else:
                accuracy_score = self._calculate_accuracy(function, test_data, function_name)
            
            result = BenchmarkResult(
                function_name=function_name,
                input_shape=input_shape,
                batch_size=batch_size,
                dtype=dtype,
                execution_times=execution_times,
                average_time=stats['average'],
                min_time=stats['min'],
                max_time=stats['max'],
                std_time=stats['std'],
                throughput=stats['throughput'],
                memory_usage=memory_stats['memory_usage'],
                peak_memory=memory_stats['peak_memory'],
                accuracy_score=accuracy_score,
                success=True
            )
            
            self.logger.info(f"基准测试完成: {function_name}, 平均时间: {stats['average']:.6f}s, "
                           f"吞吐量: {stats['throughput']:.2f} ops/s, 精度: {accuracy_score:.6f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"基准测试失败: {function_name}, 错误: {e}")
            return BenchmarkResult(
                function_name=function_name,
                input_shape=input_shape,
                batch_size=batch_size,
                dtype=dtype,
                execution_times=[],
                average_time=0.0,
                min_time=0.0,
                max_time=0.0,
                std_time=0.0,
                throughput=0.0,
                memory_usage=0,
                peak_memory=0,
                success=False,
                error_message=str(e)
            )
    
    def _create_test_data(self, input_shape: Tuple[int, int], 
                         batch_size: int, dtype: str) -> torch.Tensor:
        """创建测试数据"""
        if dtype == 'bfloat16':
            try:
                torch_dtype = torch.bfloat16
            except:
                # 如果系统不支持 bfloat16，使用 float32 但记录警告
                torch_dtype = torch.float32
                self.logger.warning("系统不支持 bfloat16，使用 float32 作为输入")
        else:
            torch_dtype = getattr(torch, dtype)
        return torch.randn(batch_size, *input_shape, dtype=torch_dtype)
    
    def _warmup_runs(self, function: Callable, test_data: torch.Tensor) -> None:
        """预热运行"""
        for _ in range(self.config.warmup_runs):
            try:
                _ = function(test_data)
            except Exception as e:
                self.logger.warning(f"预热运行失败: {e}")
    
    def _measurement_runs(self, function: Callable, 
                         test_data: torch.Tensor) -> List[float]:
        """测量运行"""
        execution_times = []
        
        for _ in range(self.config.measurement_runs):
            start_time = time.perf_counter()
            try:
                _ = function(test_data)
                end_time = time.perf_counter()
                execution_times.append(end_time - start_time)
            except Exception as e:
                self.logger.error(f"测量运行失败: {e}")
                break
        
        return execution_times
    
    def _calculate_statistics(self, execution_times: List[float]) -> Dict[str, float]:
        """计算统计信息"""
        if not execution_times:
            return {'average': 0.0, 'min': 0.0, 'max': 0.0, 'std': 0.0, 'throughput': 0.0}
        
        average = np.mean(execution_times)
        min_time = np.min(execution_times)
        max_time = np.max(execution_times)
        std = np.std(execution_times)
        throughput = 1.0 / average if average > 0 else 0.0
        
        return {
            'average': average,
            'min': min_time,
            'max': max_time,
            'std': std,
            'throughput': throughput
        }
    
    def _measure_memory_usage(self, function: Callable, 
                             test_data: torch.Tensor) -> Dict[str, int]:
        """测量内存使用"""
        import psutil
        import gc
        
        process = psutil.Process()
        
        # 清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 记录开始内存
        start_memory = process.memory_info().rss
        
        # 执行函数
        try:
            _ = function(test_data)
        except Exception:
            pass
        
        # 记录结束内存
        end_memory = process.memory_info().rss
        
        return {
            'memory_usage': end_memory - start_memory,
            'peak_memory': end_memory
        }
    
    def _calculate_accuracy(self, function: Callable, test_data: torch.Tensor, 
                           function_name: str) -> float:
        """计算精度（与参考实现对比）"""
        try:
            # 获取我们的实现结果
            our_result = function(test_data)
            
            # 获取参考实现结果
            reference_result = self._get_reference_result(function_name, test_data)
            
            # 计算相对L2误差
            if our_result.shape != reference_result.shape:
                self.logger.warning(f"形状不匹配: {our_result.shape} vs {reference_result.shape}")
                return 0.0
            
            # 计算相对L2误差
            l2_error = torch.norm(our_result - reference_result, p=2)
            reference_norm = torch.norm(reference_result, p=2)
            relative_l2_error = l2_error / (reference_norm + 1e-12)
            
            # 转换为精度分数（竞赛标准）
            epsilon_star = 1e-3
            if relative_l2_error <= epsilon_star:
                accuracy_score = 1.0
            elif relative_l2_error <= 100 * epsilon_star:
                accuracy_score = (math.log(100 * epsilon_star) - math.log(relative_l2_error)) / math.log(100)
            else:
                accuracy_score = 0.0
            
            return accuracy_score.item()
            
        except Exception as e:
            self.logger.error(f"精度计算失败: {e}")
            return 0.0
    
    def _get_reference_result(self, function_name: str, test_data: torch.Tensor) -> torch.Tensor:
        """获取参考实现结果"""
        if function_name == 'softmax':
            return torch.softmax(test_data, dim=-1)
        elif function_name == 'layer_norm':
            return torch.nn.functional.layer_norm(test_data, test_data.shape[-1:])
        elif function_name == 'rms_norm':
            # RMSNorm 参考实现
            variance = test_data.pow(2).mean(-1, keepdim=True)
            return test_data * torch.rsqrt(variance + 1e-5)
        elif function_name == 'silu':
            return torch.nn.functional.silu(test_data)
        elif function_name == 'gelu':
            return torch.nn.functional.gelu(test_data)
        elif function_name == 'add':
            # Add 需要两个输入，这里返回原数据
            return test_data
        elif function_name == 'multiply':
            # Multiply 需要两个输入，这里返回原数据
            return test_data
        else:
            # 默认返回原数据
            return test_data
    
    def benchmark_activation_functions(self, 
                                     activation_functions: Dict[str, Callable]) -> Dict[str, Any]:
        """基准测试所有激活函数"""
        results = {
            'config': {
                'warmup_runs': self.config.warmup_runs,
                'measurement_runs': self.config.measurement_runs,
                'input_shapes': self.config.input_shapes,
                'batch_sizes': self.config.batch_sizes,
                'dtypes': self.config.dtypes
            },
            'results': {}
        }
        
        for function_name, function in activation_functions.items():
            self.logger.info(f"开始测试激活函数: {function_name}")
            
            function_results = []
            
            for input_shape in self.config.input_shapes:
                for batch_size in self.config.batch_sizes:
                    for dtype in self.config.dtypes:
                        result = self.benchmark_function(
                            function, function_name, input_shape, batch_size, dtype
                        )
                        function_results.append(result)
            
            results['results'][function_name] = function_results
            
            # 记录性能日志
            successful_results = [r for r in function_results if r.success]
            if successful_results:
                avg_time = np.mean([r.average_time for r in successful_results])
                self.performance_logger.log_benchmark(function_name, {
                    'average_time': avg_time,
                    'total_tests': len(function_results),
                    'successful_tests': len(successful_results)
                })
        
        return results
    
    def compare_with_reference(self, fpga_function: Callable,
                             reference_function: Callable,
                             function_name: str,
                             input_shape: Tuple[int, int]) -> Dict[str, Any]:
        """与参考实现比较性能"""
        # 测试 FPGA 实现
        fpga_result = self.benchmark_function(fpga_function, f"{function_name}_fpga", input_shape)
        
        # 测试参考实现
        reference_result = self.benchmark_function(reference_function, f"{function_name}_reference", input_shape)
        
        # 计算性能比较
        speedup = reference_result.average_time / fpga_result.average_time if fpga_result.average_time > 0 else 0
        throughput_ratio = fpga_result.throughput / reference_result.throughput if reference_result.throughput > 0 else 0
        
        comparison = {
            'function_name': function_name,
            'fpga_result': fpga_result,
            'reference_result': reference_result,
            'speedup': speedup,
            'throughput_ratio': throughput_ratio,
            'memory_ratio': fpga_result.memory_usage / reference_result.memory_usage if reference_result.memory_usage > 0 else 0
        }
        
        self.logger.info(f"性能比较完成: {function_name}, 加速比: {speedup:.2f}x, "
                        f"吞吐量比: {throughput_ratio:.2f}x")
        
        return comparison


class PerformanceAnalyzer:
    """性能分析器"""
    
    def __init__(self):
        self.logger = get_logger()
    
    def analyze_benchmark_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """分析基准测试结果"""
        analysis = {
            'summary': {},
            'rankings': {},
            'recommendations': []
        }
        
        # 计算总体统计
        all_results = []
        for function_name, function_results in results['results'].items():
            successful_results = [r for r in function_results if r.success]
            if successful_results:
                avg_time = np.mean([r.average_time for r in successful_results])
                avg_throughput = np.mean([r.throughput for r in successful_results])
                all_results.append({
                    'function_name': function_name,
                    'average_time': avg_time,
                    'average_throughput': avg_throughput,
                    'success_rate': len(successful_results) / len(function_results)
                })
        
        # 按性能排序
        all_results.sort(key=lambda x: x['average_time'])
        
        analysis['rankings'] = {
            'fastest': all_results[0] if all_results else None,
            'slowest': all_results[-1] if all_results else None,
            'by_throughput': sorted(all_results, key=lambda x: x['average_throughput'], reverse=True),
            'by_success_rate': sorted(all_results, key=lambda x: x['success_rate'], reverse=True)
        }
        
        # 生成建议
        analysis['recommendations'] = self._generate_recommendations(all_results)
        
        return analysis
    
    def _generate_recommendations(self, results: List[Dict[str, Any]]) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        if not results:
            return recommendations
        
        # 找出最慢的函数
        slowest = max(results, key=lambda x: x['average_time'])
        if slowest['average_time'] > 0.1:  # 100ms
            recommendations.append(f"函数 {slowest['function_name']} 执行时间较长，建议优化")
        
        # 找出成功率低的函数
        low_success = [r for r in results if r['success_rate'] < 0.8]
        for result in low_success:
            recommendations.append(f"函数 {result['function_name']} 成功率较低，建议检查实现")
        
        # 检查内存使用
        recommendations.append("建议监控内存使用情况，避免内存泄漏")
        
        return recommendations


class BenchmarkReportGenerator:
    """基准测试报告生成器"""
    
    def __init__(self):
        self.logger = get_logger()
    
    def generate_report(self, results: Dict[str, Any], 
                      analysis: Dict[str, Any]) -> Dict[str, Any]:
        """生成基准测试报告"""
        report = {
            'timestamp': time.time(),
            'config': results['config'],
            'summary': {
                'total_functions': len(results['results']),
                'total_tests': sum(len(function_results) for function_results in results['results'].values()),
                'successful_tests': sum(
                    len([r for r in function_results if r.success]) 
                    for function_results in results['results'].values()
                )
            },
            'results': results['results'],
            'analysis': analysis,
            'recommendations': analysis.get('recommendations', [])
        }
        
        return report


# 全局基准测试器
_benchmark_runner: Optional[BenchmarkRunner] = None
_performance_analyzer: Optional[PerformanceAnalyzer] = None
_report_generator: Optional[BenchmarkReportGenerator] = None


def get_benchmark_runner(config: Optional[BenchmarkConfig] = None) -> BenchmarkRunner:
    """获取全局基准测试器"""
    global _benchmark_runner
    if _benchmark_runner is None:
        _benchmark_runner = BenchmarkRunner(config)
    return _benchmark_runner


def get_performance_analyzer() -> PerformanceAnalyzer:
    """获取性能分析器"""
    global _performance_analyzer
    if _performance_analyzer is None:
        _performance_analyzer = PerformanceAnalyzer()
    return _performance_analyzer


def get_report_generator() -> BenchmarkReportGenerator:
    """获取报告生成器"""
    global _report_generator
    if _report_generator is None:
        _report_generator = BenchmarkReportGenerator()
    return _report_generator


def run_benchmark(function: Callable, function_name: str,
                 input_shape: Tuple[int, int] = TENSOR_SHAPE) -> BenchmarkResult:
    """运行基准测试便捷函数"""
    runner = get_benchmark_runner()
    return runner.benchmark_function(function, function_name, input_shape)
