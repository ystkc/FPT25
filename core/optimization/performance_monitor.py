"""
性能监控模块
提供性能监控、分析和优化建议功能
"""

import time
import psutil
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import contextmanager
import threading

from core.base.constants import BENCHMARK_CONFIG, NUMERICAL_STABILITY
from core.base.exceptions import PerformanceError
from core.base.logs import get_logger, get_performance_logger


@dataclass
class PerformanceMetrics:
    """性能指标"""
    execution_time: float = 0.0
    memory_usage: int = 0
    peak_memory: int = 0
    cpu_usage: float = 0.0
    gpu_usage: float = 0.0
    tensor_operations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    throughput: float = 0.0
    latency: float = 0.0


@dataclass
class PerformanceSnapshot:
    """性能快照"""
    timestamp: float
    metrics: PerformanceMetrics
    context: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceTimer:
    """性能计时器"""
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.logger = get_logger()
    
    def start(self):
        """开始计时"""
        self.start_time = time.perf_counter()
    
    def stop(self) -> float:
        """停止计时并返回执行时间"""
        if self.start_time is None:
            raise PerformanceError("计时器未启动")
        
        self.end_time = time.perf_counter()
        execution_time = self.end_time - self.start_time
        
        self.logger.debug(f"执行时间: {execution_time:.6f} 秒")
        return execution_time
    
    def reset(self):
        """重置计时器"""
        self.start_time = None
        self.end_time = None
    
    @contextmanager
    def time_context(self, operation_name: str = ""):
        """计时上下文管理器"""
        self.start()
        try:
            yield self
        finally:
            execution_time = self.stop()
            if operation_name:
                self.logger.debug(f"{operation_name} 执行时间: {execution_time:.6f} 秒")


class SystemMonitor:
    """系统监控器"""
    
    def __init__(self):
        self.logger = get_logger()
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitoring_data: deque = deque(maxlen=1000)
    
    def start_monitoring(self, interval: float = 0.1):
        """开始监控"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(interval,)
        )
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        
        self.logger.info("开始系统监控")
    
    def stop_monitoring(self):
        """停止监控"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
        
        self.logger.info("停止系统监控")
    
    def _monitor_loop(self, interval: float):
        """监控循环"""
        while self._monitoring:
            try:
                snapshot = self._take_snapshot()
                self._monitoring_data.append(snapshot)
                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"监控错误: {e}")
    
    def _take_snapshot(self) -> Dict[str, Any]:
        """拍摄系统快照"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        snapshot = {
            'timestamp': time.time(),
            'cpu_percent': process.cpu_percent(),
            'memory_rss': memory_info.rss,
            'memory_vms': memory_info.vms,
            'num_threads': process.num_threads(),
            'num_fds': process.num_fds() if hasattr(process, 'num_fds') else 0
        }
        
        # GPU 监控
        if torch.cuda.is_available():
            snapshot.update({
                'gpu_memory_allocated': torch.cuda.memory_allocated(),
                'gpu_memory_reserved': torch.cuda.memory_reserved(),
                'gpu_utilization': self._get_gpu_utilization()
            })
        
        return snapshot
    
    def _get_gpu_utilization(self) -> float:
        """获取 GPU 使用率"""
        try:
            import nvidia_ml_py3 as nvml  # type: ignore
            nvml.nvmlInit()
            handle = nvml.nvmlDeviceGetHandleByIndex(0)
            utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
            return utilization.gpu
        except ImportError:
            return 0.0
        except Exception:
            return 0.0
    
    def get_monitoring_data(self) -> List[Dict[str, Any]]:
        """获取监控数据"""
        return list(self._monitoring_data)
    
    def get_average_metrics(self) -> Dict[str, float]:
        """获取平均指标"""
        if not self._monitoring_data:
            return {}
        
        data = list(self._monitoring_data)
        
        return {
            'avg_cpu_percent': np.mean([d['cpu_percent'] for d in data]),
            'avg_memory_rss': np.mean([d['memory_rss'] for d in data]),
            'max_memory_rss': max([d['memory_rss'] for d in data]),
            'avg_gpu_utilization': np.mean([d.get('gpu_utilization', 0) for d in data]),
            'max_gpu_memory': max([d.get('gpu_memory_allocated', 0) for d in data])
        }


class PerformanceProfiler:
    """性能分析器"""
    
    def __init__(self):
        self.logger = get_logger()
        self.performance_logger = get_performance_logger()
        self.snapshots: List[PerformanceSnapshot] = []
        self.function_times: Dict[str, List[float]] = defaultdict(list)
        self.memory_usage: List[int] = []
        self.system_monitor = SystemMonitor()
    
    def profile_function(self, func: Callable, *args, **kwargs) -> Any:
        """分析函数性能"""
        timer = PerformanceTimer()
        
        # 开始监控
        self.system_monitor.start_monitoring()
        
        # 记录开始状态
        start_memory = psutil.Process().memory_info().rss
        start_time = time.perf_counter()
        
        try:
            # 执行函数
            with timer.time_context(f"函数 {func.__name__}"):
                result = func(*args, **kwargs)
            
            # 记录结束状态
            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss
            
            # 计算指标
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            # 记录性能数据
            self.function_times[func.__name__].append(execution_time)
            self.memory_usage.append(end_memory)
            
            # 创建快照
            metrics = PerformanceMetrics(
                execution_time=execution_time,
                memory_usage=end_memory,
                peak_memory=max(self.memory_usage) if self.memory_usage else end_memory,
                cpu_usage=self.system_monitor.get_average_metrics().get('avg_cpu_percent', 0),
                gpu_usage=self.system_monitor.get_average_metrics().get('avg_gpu_utilization', 0),
                throughput=1.0 / execution_time if execution_time > 0 else 0,
                latency=execution_time
            )
            
            snapshot = PerformanceSnapshot(
                timestamp=time.time(),
                metrics=metrics,
                context=f"函数 {func.__name__}",
                metadata={'args': str(args), 'kwargs': str(kwargs)}
            )
            
            self.snapshots.append(snapshot)
            
            # 记录日志
            self.performance_logger.log_activation_function(
                func.__name__, 
                getattr(args[0], 'shape', 'unknown') if args else 'unknown',
                execution_time,
                0.0  # 精度需要单独计算
            )
            
            return result
            
        finally:
            # 停止监控
            self.system_monitor.stop_monitoring()
    
    def profile_activation_function(self, func_name: str, 
                                  input_shape: tuple,
                                  func: Callable, 
                                  *args, **kwargs) -> tuple:
        """分析激活函数性能"""
        timer = PerformanceTimer()
        
        # 记录开始状态
        start_memory = psutil.Process().memory_info().rss
        start_time = time.perf_counter()
        
        try:
            # 执行函数
            with timer.time_context(f"激活函数 {func_name}"):
                result = func(*args, **kwargs)
            
            # 记录结束状态
            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss
            
            # 计算指标
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            # 记录性能数据
            self.function_times[func_name].append(execution_time)
            self.memory_usage.append(end_memory)
            
            # 创建快照
            metrics = PerformanceMetrics(
                execution_time=execution_time,
                memory_usage=end_memory,
                peak_memory=max(self.memory_usage) if self.memory_usage else end_memory,
                throughput=1.0 / execution_time if execution_time > 0 else 0,
                latency=execution_time
            )
            
            snapshot = PerformanceSnapshot(
                timestamp=time.time(),
                metrics=metrics,
                context=f"激活函数 {func_name}",
                metadata={'input_shape': input_shape}
            )
            
            self.snapshots.append(snapshot)
            
            # 记录日志
            self.performance_logger.log_activation_function(
                func_name, input_shape, execution_time, 0.0
            )
            
            return result, execution_time
            
        except Exception as e:
            self.logger.error(f"激活函数 {func_name} 性能分析失败: {e}")
            raise
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.snapshots:
            return {}
        
        # 计算统计信息
        execution_times = [s.metrics.execution_time for s in self.snapshots]
        memory_usage = [s.metrics.memory_usage for s in self.snapshots]
        
        summary = {
            'total_operations': len(self.snapshots),
            'total_execution_time': sum(execution_times),
            'average_execution_time': np.mean(execution_times),
            'min_execution_time': min(execution_times),
            'max_execution_time': max(execution_times),
            'std_execution_time': np.std(execution_times),
            'average_memory_usage': np.mean(memory_usage),
            'peak_memory_usage': max(memory_usage),
            'function_breakdown': {}
        }
        
        # 按函数分组统计
        for func_name, times in self.function_times.items():
            if times:
                summary['function_breakdown'][func_name] = {
                    'count': len(times),
                    'total_time': sum(times),
                    'average_time': np.mean(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'std_time': np.std(times)
                }
        
        return summary
    
    def get_optimization_suggestions(self) -> List[str]:
        """获取优化建议"""
        suggestions = []
        
        if not self.snapshots:
            return suggestions
        
        summary = self.get_performance_summary()
        
        # 基于执行时间的建议
        avg_time = summary.get('average_execution_time', 0)
        if avg_time > 1.0:
            suggestions.append("平均执行时间较长，考虑使用并行处理或优化算法")
        
        # 基于内存使用的建议
        peak_memory = summary.get('peak_memory_usage', 0)
        if peak_memory > 1024 * 1024 * 1024:  # 1GB
            suggestions.append("内存使用较高，考虑使用内存池或批处理")
        
        # 基于函数性能的建议
        function_breakdown = summary.get('function_breakdown', {})
        for func_name, stats in function_breakdown.items():
            if stats['average_time'] > 0.1:  # 100ms
                suggestions.append(f"函数 {func_name} 执行时间较长，考虑优化")
        
        # 基于标准差的建议
        std_time = summary.get('std_execution_time', 0)
        if std_time > avg_time * 0.5:  # 标准差大于平均值的50%
            suggestions.append("执行时间变化较大，考虑检查输入数据一致性")
        
        return suggestions
    
    def clear_data(self):
        """清空性能数据"""
        self.snapshots.clear()
        self.function_times.clear()
        self.memory_usage.clear()
        self.logger.info("清空性能数据")


class BenchmarkRunner:
    """基准测试运行器"""
    
    def __init__(self):
        self.logger = get_logger()
        self.profiler = PerformanceProfiler()
    
    def run_benchmark(self, func: Callable, 
                     test_cases: List[Dict[str, Any]],
                     warmup_runs: int = BENCHMARK_CONFIG['warmup_runs'],
                     measurement_runs: int = BENCHMARK_CONFIG['measurement_runs']) -> Dict[str, Any]:
        """运行基准测试"""
        self.logger.info(f"开始基准测试: {func.__name__}")
        
        results = {
            'function_name': func.__name__,
            'test_cases': len(test_cases),
            'warmup_runs': warmup_runs,
            'measurement_runs': measurement_runs,
            'results': []
        }
        
        for i, test_case in enumerate(test_cases):
            self.logger.info(f"运行测试用例 {i + 1}/{len(test_cases)}")
            
            # 预热运行
            for _ in range(warmup_runs):
                try:
                    func(**test_case)
                except Exception as e:
                    self.logger.warning(f"预热运行失败: {e}")
            
            # 测量运行
            execution_times = []
            for run in range(measurement_runs):
                try:
                    start_time = time.perf_counter()
                    result = func(**test_case)
                    end_time = time.perf_counter()
                    
                    execution_times.append(end_time - start_time)
                except Exception as e:
                    self.logger.error(f"测量运行失败: {e}")
                    break
            
            if execution_times:
                case_result = {
                    'test_case': i,
                    'execution_times': execution_times,
                    'average_time': np.mean(execution_times),
                    'min_time': min(execution_times),
                    'max_time': max(execution_times),
                    'std_time': np.std(execution_times),
                    'throughput': 1.0 / np.mean(execution_times) if np.mean(execution_times) > 0 else 0
                }
                results['results'].append(case_result)
        
        # 计算总体统计
        all_times = [r['average_time'] for r in results['results']]
        if all_times:
            results['overall_stats'] = {
                'average_time': np.mean(all_times),
                'min_time': min(all_times),
                'max_time': max(all_times),
                'std_time': np.std(all_times),
                'total_time': sum(all_times)
            }
        
        self.logger.info(f"基准测试完成: {func.__name__}")
        return results


# 全局性能监控器
_performance_profiler: Optional[PerformanceProfiler] = None
_benchmark_runner: Optional[BenchmarkRunner] = None


def get_performance_profiler() -> PerformanceProfiler:
    """获取全局性能分析器"""
    global _performance_profiler
    if _performance_profiler is None:
        _performance_profiler = PerformanceProfiler()
    return _performance_profiler


def get_benchmark_runner() -> BenchmarkRunner:
    """获取基准测试运行器"""
    global _benchmark_runner
    if _benchmark_runner is None:
        _benchmark_runner = BenchmarkRunner()
    return _benchmark_runner


def profile_function(func: Callable, *args, **kwargs) -> Any:
    """函数性能分析装饰器"""
    profiler = get_performance_profiler()
    return profiler.profile_function(func, *args, **kwargs)


@contextmanager
def performance_monitoring(operation_name: str = ""):
    """性能监控上下文管理器"""
    profiler = get_performance_profiler()
    timer = PerformanceTimer()
    
    timer.start()
    try:
        yield timer
    finally:
        execution_time = timer.stop()
        if operation_name:
            profiler.logger.debug(f"{operation_name} 执行时间: {execution_time:.6f} 秒")
        # 将执行时间存储到timer对象中，供外部访问
        timer.execution_time = execution_time
