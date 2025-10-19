from abc import ABC, abstractmethod
import pickle
import torch
import math
import time
import traceback
from typing import Any, Tuple, Dict, Callable

from config.config import ActivationConfig, LookupTableConfig
from core.base.constants import (
    TENSOR_SHAPE, INTERPOLATION_METHODS
)
from core.base.logs import get_logger, get_performance_logger
from core.optimization.performance_monitor import get_performance_logger, get_performance_profiler
from core.utils import get_memory_manager, MemoryContext, memory_context, ensure_dtype
from core.algorithms import LookupTable, create_exp_table
from core.base.exceptions import (
    InvalidTensorShapeError,
    validate_dtype
)
class BaseActivationFunction(ABC):
    """激活函数基类"""
    
    def __init__(self, config: ActivationConfig, table_func: Callable, name: str = None, table_name: str = None):
        self.config = config
        self.logger = get_logger()
        self.performance_logger = get_performance_logger()
        self.profiler = get_performance_profiler()
        self.memory_manager = get_memory_manager()

        self.name = name
        self.table_func = table_func
        self.table_name = table_name
        
        # 验证配置
        self._validate_config()
        
        # 初始化查找表
        self.lookup_table: LookupTable = None

    def _validate_input(self, x: torch.Tensor) -> None:
        """验证输入张量"""
        if not isinstance(x, torch.Tensor):
            raise TypeError("输入必须是 torch.Tensor")
        
        if len(x.shape) != 2:
            raise InvalidTensorShapeError(x.shape, (None, None))
        
        # 使用统一的数据类型管理器处理数据类型转换
        x = ensure_dtype(x, self.config.dtype_str)
    
    def _validate_config(self) -> None:
        """验证配置参数"""
        validate_dtype(self.config.dtype_str, ['float32', 'bfloat16'])
        
        if self.config.interpolation_method not in INTERPOLATION_METHODS:
            raise ValueError(f"不支持的插值方法: {self.config.interpolation_method}")
        
    def initialize_lookup_table(self, tableConfig: LookupTableConfig) -> None:
        """初始化查找表"""
        if not self.config.use_lookup_table:
            return
        try:
            self.lookup_table = LookupTable(tableConfig) # 由于历史遗留原因，此处直接销毁重新生成LookupTable对象
            self.lookup_table.generate_table(self.table_func)
            self.logger.info(f"{self.name} 查找表初始化完成: {tableConfig.bit_len} 位共 {len(self.lookup_table.y_points)}")
        except Exception as e:
            self.logger.warning(f"{self.name} 查找表初始化失败，将使用直接计算: {e}")
            traceback.print_exc()
            self.lookup_table = None

    @abstractmethod
    def _forward_with_lookup_table(self, original_tensor: torch.Tensor, mem_ctx: MemoryContext) -> torch.Tensor:
        """使用查找表进行前向传播"""
    
    @abstractmethod
    def _forward_direct(self, original_tensor: torch.Tensor, mem_ctx: MemoryContext) -> torch.Tensor:
        """直接进行前向传播"""

    @abstractmethod
    def _forward_reference(self, original_tensor: torch.Tensor, mem_ctx: MemoryContext) -> torch.Tensor:
        """(torch库的)参考实现"""
    
    def forward(self, original_tensor: torch.Tensor, ref: bool = False) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (N, D)
            ref: 是否使用(torch库的)参考实现
            
        Returns:
            输出张量
        """
        # 验证输入
        self._validate_input(original_tensor)
        
        # 使用内存池进行内存管理
        with memory_context() as mem_ctx:
            if ref:
                result = self._forward_reference(original_tensor, mem_ctx)
            elif self.config.use_lookup_table and self.lookup_table is not None:
                result = self._forward_with_lookup_table(original_tensor, mem_ctx)
            else:
                result = self._forward_direct(original_tensor, mem_ctx)
            
        return result
        
    def benchmark(self, num_runs: int = 100) -> Dict[str, Any]:
        """基准测试
        Args:
            shape: 输入张量形状
            num_runs: 运行次数

        Returns:
            results: {
                 'results': [
                     {
                         'time': 0.000123,  # 运行时间（秒）
                         'throughput': 8333.33,  # 吞吐量（次/秒）
                         'l2_error': 0.0001,  # L2 误差
                         'accuracy_score': 1.0  # 精度分数
                     },
                    ...
                 ],
                 'avg_time': 0.000123,  # 平均运行时间（秒）
                'min_time': 0.000123,  # 最小运行时间（秒）
                'max_time': 0.000123,  # 最大运行时间（秒）
                 'avg_score': 1.0,  # 平均精度分数
                'min_score': 1.0,  # 最小精度分数
                'max_score': 1.0  # 最大精度分数
            }
        """
        
        # 基准测试 - 使用更高精度的时间测量
        shape: Tuple[int, int] = TENSOR_SHAPE
        results = {'results': []}
        
        # 计算精度 - 在float32精度下评估
        times = []
        scores = []
        with torch.no_grad():
            for _ in range(num_runs):
                
                # 创建测试数据
                x = torch.randn(shape, dtype=self.config.dtype)

                start_time = time.perf_counter_ns()  # 使用纳秒精度
                output = self.forward(x)
                end_time = time.perf_counter_ns()
                time_elapse = (end_time - start_time) / 1e9
                times.append(time_elapse)  # 转换为秒
                
                # 使用 fp32 计算参考结果
                reference_output = self.forward(x, ref=True)
                
                # 计算相对 L2 误差
                l2_error = torch.norm(output - reference_output, p=2) / (torch.norm(reference_output, p=2) + 1e-12)
                
                # 计算精度分数（竞赛标准）
                epsilon_star = 1e-3
                relative_l2_error = float(l2_error.item())
                if relative_l2_error <= epsilon_star:
                    accuracy_score = 1.0
                elif relative_l2_error <= 100 * epsilon_star:
                    accuracy_score = (math.log(100 * epsilon_star) - math.log(relative_l2_error)) / math.log(100)
                else:
                    accuracy_score = 0.0
                scores.append(accuracy_score)

                self.performance_logger.log_activation_function(
                    self.name, x.shape, time_elapse, accuracy_score
                )
                
                results['results'].append({
                    'time': time_elapse,
                    'throughput': 1.0 / time_elapse if time_elapse > 0 else 0,
                    'l2_error': relative_l2_error,
                    'accuracy_score': accuracy_score
                })
        # 计算统计信息
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        results['avg_time'] = avg_time
        results['min_time'] = min_time
        results['max_time'] = max_time

        # 计算分数
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)
        results['avg_score'] = avg_score
        results['min_score'] = min_score
        results['max_score'] = max_score
        return results

    def get_config(self) -> Dict[str, Any]:
        """获取配置信息"""
        return {
            'use_lookup_table': self.config.use_lookup_table,
            'interpolation_method': self.config.interpolation_method,
            'use_fixed_point': self.config.use_fixed_point,
            'fixed_point_format': self.config.fixed_point_format,
            'dtype': self.config.dtype,
            'lookup_table_available': self.lookup_table is not None
        }
    
    @abstractmethod
    def backward(self, grad_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """反向传播"""

        
class ActivationScanner:
    """激活函数 参数扫描测试器"""
    
    def __init__(self, activation_function: BaseActivationFunction):
        self.logger = get_logger()
        self.acfun = activation_function
        self.acfun_name = activation_function.name
    
    def optimize_lookup_bit_len(self, bit_lens: list = None):
        """优化查找表大小"""
        if bit_lens is None:
            bit_lens = [15, 16, 17, 18, 19, 20]

        table_config = LookupTableConfig(table_name=self.acfun_name+'-table')
        
        results = []
        
        for bit_len in bit_lens:
            # 生成查找表
            table_config.bit_len = bit_len
            if bit_len > 16: # 从bfloat16开始，如果位宽超过16了再转换成float32
                table_config.dtype_str = 'float32'
                table_config.dtype_len = 32
                table_config.unsigned_type_str = 'uint32' # 避免符号位影响正负
                table_config.x_struct_min_int = 0x00000000
                table_config.x_struct_max_int = 0xffffffff
            else:
                table_config.dtype_str = 'bfloat16'
                table_config.dtype_len = 16
                table_config.unsigned_type_str = 'uint16'
                table_config.x_struct_min_int = 0x0000
                table_config.x_struct_max_int = 0xffff
            table_config.__post_init__()

            self.acfun.initialize_lookup_table(table_config)
            # 基准测试
            benchmark_result = self.acfun.benchmark(num_runs=5)
            
            results.append({
                'bit_len': bit_len,
                'avg_time': benchmark_result['avg_time'],
                'avg_score': benchmark_result['avg_score'],
                'max_score': benchmark_result['max_score'],
                'min_score': benchmark_result['min_score'],
                'result': benchmark_result['results']
            })

        # 对照组：direct
        self.acfun.config.use_lookup_table = False
        # 对照组1：bfloat16 - direct
        self.acfun.config.compute_dtype_str = 'bfloat16'
        self.acfun.config.compute_dtype = torch.bfloat16
        table_config.bit_len = 16
        table_config.dtype_str = 'bfloat16'
        table_config.dtype_len = 16
        table_config.unsigned_type_str = 'uint16'
        table_config.x_struct_min_int = 0x0000
        table_config.x_struct_max_int = 0xffff
        table_config.__post_init__()
        self.acfun.initialize_lookup_table(table_config)
        benchmark_result = self.acfun.benchmark(num_runs=5)
        results.append({
            'bit_len': 'BF16-direct',
            'avg_time': benchmark_result['avg_time'],
            'avg_score': benchmark_result['avg_score'],
            'max_score': benchmark_result['max_score'],
            'min_score': benchmark_result['min_score'],
            'result': benchmark_result['results']
        })
        # 对照组2：float32 - direct
        self.acfun.config.compute_dtype_str = 'float32'
        self.acfun.config.compute_dtype = torch.float32
        table_config.bit_len = 16
        table_config.dtype_str = 'float32'
        table_config.dtype_len = 32
        table_config.unsigned_type_str = 'uint32'
        table_config.x_struct_min_int = 0x00000000
        table_config.x_struct_max_int = 0xffffffff
        table_config.__post_init__()
        self.acfun.initialize_lookup_table(table_config)
        benchmark_result = self.acfun.benchmark(num_runs=5)
        results.append({
            'bit_len': 'FP32-direct',
            'avg_time': benchmark_result['avg_time'],
            'avg_score': benchmark_result['avg_score'],
            'max_score': benchmark_result['max_score'],
            'min_score': benchmark_result['min_score'],
            'result': benchmark_result['results']
        })
        
        # 打印结果
        print("|   bitlen   | avg_time | avg_score | max_score | min_score |")
        print("|------------|----------|-----------|-----------|-----------|")
        for result in results:
            print(f"|{result['bit_len']:<12}| {result['avg_time']:.5f}s | {result['avg_score']:.7f} | {result['max_score']:.7f} | {result['min_score']:.7f} |")
        print("|------------|----------|-----------|-----------|-----------|")