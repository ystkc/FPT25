"""
Softmax 激活函数模块
实现高效的 Softmax 激活函数，支持查找表优化和硬件加速
"""

import traceback
import torch
import math
from typing import Union, Optional, Dict, Any, Tuple
from dataclasses import dataclass

from core.base.constants import (
    TENSOR_SHAPE, DEFAULT_DTYPE, EPSILON_TINY, 
    ACTIVATION_FUNCTION_WEIGHTS, INTERPOLATION_METHODS
)
from core.base.exceptions import (
    InvalidTensorShapeError, UnsupportedDataTypeError,
    validate_tensor_shape, validate_dtype
)
from core.algorithms import MathUtils, create_exp_table, get_table_manager
from core.hardware import create_fixed_point, FixedPointNumber
from core.optimization import get_performance_profiler, performance_monitoring
from core.base.logs import get_logger, get_performance_logger
from core.utils.data_type_manager import get_data_type_manager, ensure_dtype, safe_bf16_operation
from core.utils.memory_pool import get_memory_manager, memory_context


@dataclass
class SoftmaxConfig:
    """Softmax 配置"""
    use_lookup_table: bool = True
    lookup_table_bitlen: int = 14  # 增加查找表大小
    interpolation_method: str = 'quadratic'  # 使用二次插值
    use_fixed_point: bool = False
    fixed_point_format: str = 'Q16_16'
    numerical_stability: bool = True
    dtype: str = DEFAULT_DTYPE


class SoftmaxActivation:
    """Softmax 激活函数类"""
    
    def __init__(self, config: SoftmaxConfig):
        self.config = config
        self.logger = get_logger()
        self.performance_logger = get_performance_logger()
        self.profiler = get_performance_profiler()
        self.data_type_manager = get_data_type_manager()
        self.memory_manager = get_memory_manager()
        
        # 验证配置
        self._validate_config()
        
        # 初始化查找表
        self.lookup_table = None
        if self.config.use_lookup_table:
            self._initialize_lookup_table()
    
    def _validate_config(self) -> None:
        """验证配置参数"""
        validate_dtype(self.config.dtype, ['float32', 'bfloat16'])
        
        if self.config.interpolation_method not in INTERPOLATION_METHODS:
            raise ValueError(f"不支持的插值方法: {self.config.interpolation_method}")
    
    def _initialize_lookup_table(self) -> None:
        """初始化查找表"""
        try:
            self.lookup_table = create_exp_table(
                name="softmax_exp",
                bit_len=self.config.lookup_table_bitlen,
                interpolation_method=self.config.interpolation_method
            )
            self.logger.info(f"Softmax 查找表初始化完成: {self.config.lookup_table_bitlen} 点")
        except Exception as e:
            self.logger.warning(f"查找表初始化失败，将使用直接计算: {e}")
            traceback.print_exc()
            self.lookup_table = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Softmax 前向传播
        
        Args:
            x: 输入张量，形状为 (N, D)
            
        Returns:
            Softmax 输出张量
        """
        # 验证输入
        self._validate_input(x)
        
        # 使用内存池进行内存管理
        with memory_context() as mem_ctx:
            # 记录性能
            with performance_monitoring("Softmax Forward") as timer:
                if self.config.use_lookup_table and self.lookup_table is not None:
                    result = self._forward_with_lookup_table(x, mem_ctx)
                else:
                    result = self._forward_direct(x, mem_ctx)
            
            # 记录性能日志 - 使用实际测量的时间
            execution_time = getattr(timer, 'execution_time', 0.0)
            
            # 计算精度（与参考实现对比）- 在float32精度下评估
            with torch.no_grad():
                # 使用相同的输入计算参考结果
                x_fp32 = x.to(torch.float32)
                reference_output = torch.softmax(x_fp32, dim=-1)
                
                # 重新计算我们的结果，但保持在float32精度下
                if self.config.use_lookup_table and self.lookup_table is not None:
                    # 数值稳定性：减去最大值
                    if self.config.numerical_stability:
                        x_max = x_fp32.max(dim=-1, keepdim=True)[0]
                        x_stable = x_fp32 - x_max
                    else:
                        x_stable = x_fp32
                    
                    # 使用查找表计算指数
                    exp_values = self.lookup_table.lookup_exp(x_stable)
                    
                    # 计算Softmax
                    sum_exp = exp_values.sum(dim=-1, keepdim=True)
                    our_output_fp32 = exp_values / sum_exp
                else:
                    # 直接计算
                    our_output_fp32 = torch.softmax(x_fp32, dim=-1)
                
                # 计算相对L2误差
                l2_error = torch.norm(our_output_fp32 - reference_output, p=2) / (torch.norm(reference_output, p=2) + 1e-12)
                relative_l2_error = float(l2_error.item())
                
                # 计算精度分数（竞赛标准）
                epsilon_star = 1e-3
                if relative_l2_error <= epsilon_star:
                    accuracy_score = 1.0
                elif relative_l2_error <= 100 * epsilon_star:
                    accuracy_score = (math.log(100 * epsilon_star) - math.log(relative_l2_error)) / math.log(100)
                else:
                    accuracy_score = 0.0
            
            self.performance_logger.log_activation_function(
                "softmax", x.shape, execution_time, accuracy_score
            )
            
            return result
    
    def _validate_input(self, x: torch.Tensor) -> None:
        """验证输入张量"""
        if not isinstance(x, torch.Tensor):
            raise TypeError("输入必须是 torch.Tensor")
        
        if len(x.shape) != 2:
            raise InvalidTensorShapeError(x.shape, (None, None))
        
        # 使用统一的数据类型管理器处理数据类型转换
        x = ensure_dtype(x, self.config.dtype)
    
    def _forward_direct(self, x: torch.Tensor, mem_ctx) -> torch.Tensor:
        """直接计算 Softmax"""
        # 使用数据类型管理器进行安全的类型转换
        compute_tensor, original_tensor = self.data_type_manager.convert_for_computation(x)
        
        if self.config.numerical_stability:
            # 数值稳定的 Softmax 实现
            result = MathUtils.softmax_stable(compute_tensor, dim=-1)
        else:
            # 标准 Softmax 实现
            exp_x = torch.exp(compute_tensor)
            result = exp_x / torch.sum(exp_x, dim=-1, keepdim=True)
        
        # 转换回原始数据类型
        return self.data_type_manager.convert_back(result, original_tensor)
    
    def _forward_with_lookup_table(self, x: torch.Tensor, mem_ctx) -> torch.Tensor:
        """使用查找表计算 Softmax"""
        # 使用数据类型管理器进行安全的类型转换
        compute_tensor, original_tensor = self.data_type_manager.convert_for_computation(x)
        
        # 数值稳定性：减去最大值
        x_max = torch.max(compute_tensor, dim=-1, keepdim=True)[0]
        x_shifted = compute_tensor - x_max
        
        # 使用查找表计算指数
        if self.config.use_fixed_point:
            exp_x = self._exp_with_fixed_point(x_shifted)
        else:
            exp_x = self.lookup_table.lookup_exp(x_shifted)
        
        # 计算归一化
        sum_exp = torch.sum(exp_x, dim=-1, keepdim=True)
        result = exp_x / sum_exp
        
        # 转换回原始数据类型
        return self.data_type_manager.convert_back(result, original_tensor)
    
    def _exp_with_fixed_point(self, x: torch.Tensor) -> torch.Tensor:
        """使用定点数计算指数"""
        # 转换为定点数
        fixed_x = create_fixed_point(0.0, self.config.fixed_point_format)
        
        # 逐元素计算（这里简化实现）
        result = torch.zeros_like(x)
        for i in range(x.numel()):
            flat_x = x.view(-1)
            flat_result = result.view(-1)
            
            # 创建定点数
            fixed_val = create_fixed_point(flat_x[i].item(), self.config.fixed_point_format)
            
            # 使用查找表
            if self.lookup_table:
                exp_val = self.lookup_table.lookup_exp(fixed_val.to_float())
            else:
                exp_val = math.exp(fixed_val.to_float())
            
            flat_result[i] = exp_val
        
        return result
    
    def backward(self, grad_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Softmax 反向传播
        
        Args:
            grad_output: 输出梯度
            x: 原始输入
            
        Returns:
            输入梯度
        """
        # 计算 Softmax 输出
        softmax_output = self.forward(x)
        
        # 计算梯度
        grad_input = softmax_output * (grad_output - torch.sum(grad_output * softmax_output, dim=-1, keepdim=True))
        
        return grad_input
    
    def get_config(self) -> Dict[str, Any]:
        """获取配置信息"""
        return {
            'use_lookup_table': self.config.use_lookup_table,
            'lookup_bit_len': self.config.lookup_table_bitlen,
            'interpolation_method': self.config.interpolation_method,
            'use_fixed_point': self.config.use_fixed_point,
            'fixed_point_format': self.config.fixed_point_format,
            'numerical_stability': self.config.numerical_stability,
            'dtype': self.config.dtype,
            'lookup_table_available': self.lookup_table is not None
        }
    
    def benchmark(self, input_shapes: list = None, 
                 num_runs: int = 100) -> Dict[str, Any]:
        """基准测试"""
        if input_shapes is None:
            input_shapes = [TENSOR_SHAPE, (32, 128), (16, 256)]
        
        results = {
            'function_name': 'softmax',
            'config': self.get_config(),
            'input_shapes': input_shapes,
            'num_runs': num_runs,
            'results': []
        }
        
        for shape in input_shapes:
            # 使用数据类型管理器创建测试数据
            x = self.data_type_manager.create_tensor(
                torch.randn(shape), 
                self.config.dtype
            )
            
            # 预热
            for _ in range(10):
                _ = self.forward(x)
            
            # 基准测试 - 使用更高精度的时间测量
            import time
            times = []
            for _ in range(num_runs):
                start_time = time.perf_counter_ns()  # 使用纳秒精度
                output = self.forward(x)
                end_time = time.perf_counter_ns()
                times.append((end_time - start_time) / 1e9)  # 转换为秒
            
            # 计算精度 - 在float32精度下评估
            with torch.no_grad():
                # 使用 fp32 计算参考结果
                x_fp32 = x.to(torch.float32)
                reference_output = torch.softmax(x_fp32, dim=-1)
                
                # 重新计算我们的结果，但保持在float32精度下
                if self.config.use_lookup_table and self.lookup_table is not None:
                    # 数值稳定性：减去最大值
                    if self.config.numerical_stability:
                        x_max = x_fp32.max(dim=-1, keepdim=True)[0]
                        x_stable = x_fp32 - x_max
                    else:
                        x_stable = x_fp32
                    
                    # 使用查找表计算指数
                    exp_values = self.lookup_table.lookup_exp(x_stable)
                    
                    # 计算Softmax
                    sum_exp = exp_values.sum(dim=-1, keepdim=True)
                    our_output_fp32 = exp_values / sum_exp
                else:
                    # 直接计算
                    our_output_fp32 = torch.softmax(x_fp32, dim=-1)
                
                # 计算相对 L2 误差
                l2_error = torch.norm(our_output_fp32 - reference_output, p=2) / (torch.norm(reference_output, p=2) + 1e-12)
                
                # 计算精度分数（竞赛标准）
                epsilon_star = 1e-3
                relative_l2_error = float(l2_error.item())
                if relative_l2_error <= epsilon_star:
                    accuracy_score = 1.0
                elif relative_l2_error <= 100 * epsilon_star:
                    accuracy_score = (math.log(100 * epsilon_star) - math.log(relative_l2_error)) / math.log(100)
                else:
                    accuracy_score = 0.0
            
            # 计算统计信息
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            std_time = math.sqrt(sum((t - avg_time) ** 2 for t in times) / len(times))
            
            results['results'].append({
                'shape': shape,
                'average_time': avg_time,
                'min_time': min_time,
                'max_time': max_time,
                'std_time': std_time,
                'throughput': 1.0 / avg_time if avg_time > 0 else 0,
                'accuracy_score': accuracy_score,
                'l2_error': relative_l2_error
            })
        
        return results
    
    def accuracy_test(self, reference_func: callable = None) -> Dict[str, Any]:
        """精度测试"""
        if reference_func is None:
            # 使用 PyTorch 标准实现作为参考
            reference_func = lambda x: torch.softmax(x, dim=-1)
        
        # 创建测试数据
        x = self.data_type_manager.create_tensor(
            torch.randn(TENSOR_SHAPE), 
            self.config.dtype
        )
        
        # 计算结果
        our_result = self.forward(x)
        reference_result = reference_func(x)
        
        # 计算误差
        mse = torch.mean((our_result - reference_result) ** 2)
        mae = torch.mean(torch.abs(our_result - reference_result))
        
        # 计算相对 L2 误差（竞赛标准）
        l2_error = torch.norm(our_result - reference_result, p=2) / (torch.norm(reference_result, p=2) + EPSILON_TINY)
        
        return {
            'mse': mse.item(),
            'mae': mae.item(),
            'l2_error': l2_error.item(),
            'relative_l2_error': l2_error.item(),
            'max_error': torch.max(torch.abs(our_result - reference_result)).item(),
            'min_error': torch.min(torch.abs(our_result - reference_result)).item()
        }


class SoftmaxOptimizer:
    """Softmax 优化器"""
    
    def __init__(self):
        self.logger = get_logger()
    
    def optimize_lookup_bit_len(self, input_shape: tuple = TENSOR_SHAPE,
                                 bit_lens: list = None) -> Dict[str, Any]:
        """优化查找表大小"""
        if bit_lens is None:
            bit_lens = [400, 600, 800, 1000]
        
        results = []
        
        for bit_len in bit_lens:
            config = SoftmaxConfig(
                use_lookup_table=True,
                lookup_table_bitlen=bit_len,
                interpolation_method='linear'
            )
            
            softmax = SoftmaxActivation(config)
            
            # 基准测试
            benchmark_result = softmax.benchmark([input_shape], num_runs=50)
            accuracy_result = softmax.accuracy_test()
            
            results.append({
                'bit_len': bit_len,
                'average_time': benchmark_result['results'][0]['average_time'],
                'l2_error': accuracy_result['l2_error']
            })
        
        # 找到最优配置
        best_result = min(results, key=lambda x: x['l2_error'])
        
        return {
            'results': results,
            'best_bit_len': best_result['bit_len'],
            'best_l2_error': best_result['l2_error'],
            'best_time': best_result['average_time']
        }
    
    def optimize_interpolation_method(self, input_shape: tuple = TENSOR_SHAPE) -> Dict[str, Any]:
        """优化插值方法"""
        results = []
        
        for method in INTERPOLATION_METHODS:
            config = SoftmaxConfig(
                use_lookup_table=True,
                lookup_table_bitlen=800,
                interpolation_method=method
            )
            
            softmax = SoftmaxActivation(config)
            
            # 基准测试
            benchmark_result = softmax.benchmark([input_shape], num_runs=50)
            accuracy_result = softmax.accuracy_test()
            
            results.append({
                'interpolation_method': method,
                'average_time': benchmark_result['results'][0]['average_time'],
                'l2_error': accuracy_result['l2_error']
            })
        
        # 找到最优配置
        best_result = min(results, key=lambda x: x['l2_error'])
        
        return {
            'results': results,
            'best_method': best_result['interpolation_method'],
            'best_l2_error': best_result['l2_error'],
            'best_time': best_result['average_time']
        }


# 便捷函数
def create_softmax(config: Optional[SoftmaxConfig] = None) -> SoftmaxActivation:
    """创建 Softmax 激活函数"""
    if config is None:
        config = SoftmaxConfig()
    return SoftmaxActivation(config)


def softmax_forward(x: torch.Tensor, 
                   use_lookup_table: bool = True,
                   lookup_bit_len: int = 800,
                   interpolation_method: str = 'linear') -> torch.Tensor:
    """Softmax 前向传播便捷函数"""
    config = SoftmaxConfig(
        use_lookup_table=use_lookup_table,
        lookup_table_bitlen=lookup_bit_len,
        interpolation_method=interpolation_method
    )
    
    softmax = SoftmaxActivation(config)
    return softmax.forward(x)


def softmax_benchmark(input_shapes: list = None, 
                     num_runs: int = 100) -> Dict[str, Any]:
    """Softmax 基准测试便捷函数"""
    softmax = SoftmaxActivation(SoftmaxConfig())
    return softmax.benchmark(input_shapes, num_runs)
