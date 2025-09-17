"""
激活函数管理器
统一管理所有激活函数的创建、调用和优化
"""

import torch
from typing import Dict, List, Optional, Any, Union, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass

from core.base.constants import ACTIVATION_FUNCTIONS, ACTIVATION_FUNCTION_WEIGHTS
from core.base.exceptions import ActivationFunctionNotImplementedError
from core.base.logs import get_logger, get_performance_logger
from core.activation_functions.softmax_activation import SoftmaxActivation, SoftmaxConfig


@dataclass
class ActivationConfig:
    """激活函数配置基类"""
    dtype: str = 'bfloat16'
    use_fixed_point: bool = False
    fixed_point_format: str = 'Q16_16'


class BaseActivationFunction(ABC):
    """激活函数基类"""
    
    def __init__(self, config: ActivationConfig):
        self.config = config
        self.logger = get_logger()
        self.performance_logger = get_performance_logger()
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        pass
    
    @abstractmethod
    def backward(self, grad_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """反向传播"""
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """获取配置信息"""
        pass
    
    def benchmark(self, input_shapes: List[tuple] = None, 
                 num_runs: int = 100) -> Dict[str, Any]:
        """基准测试"""
        if input_shapes is None:
            input_shapes = [(64, 768)]
        
        results = {
            'function_name': self.__class__.__name__.lower().replace('activation', ''),
            'config': self.get_config(),
            'input_shapes': input_shapes,
            'num_runs': num_runs,
            'results': []
        }
        
        for shape in input_shapes:
            # 创建测试数据
            x = torch.randn(shape, dtype=getattr(torch, self.config.dtype))
            
            # 预热
            for _ in range(10):
                _ = self.forward(x)
            
            # 基准测试
            import time
            times = []
            for _ in range(num_runs):
                start_time = time.perf_counter()
                _ = self.forward(x)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            
            # 计算统计信息
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            results['results'].append({
                'shape': shape,
                'average_time': avg_time,
                'min_time': min_time,
                'max_time': max_time,
                'throughput': 1.0 / avg_time if avg_time > 0 else 0
            })
        
        return results
    
    def accuracy_test(self, reference_func: Optional[Callable] = None) -> Dict[str, Any]:
        """精度测试"""
        if reference_func is None:
            raise NotImplementedError("需要提供参考函数")
        
        # 创建测试数据
        x = torch.randn((64, 768), dtype=getattr(torch, self.config.dtype))
        
        # 计算结果
        our_result = self.forward(x)
        reference_result = reference_func(x)
        
        # 计算误差
        mse = torch.mean((our_result - reference_result) ** 2)
        mae = torch.mean(torch.abs(our_result - reference_result))
        
        # 计算相对 L2 误差
        l2_error = torch.norm(our_result - reference_result, p=2) / (torch.norm(reference_result, p=2) + 1e-12)
        
        return {
            'mse': mse.item(),
            'mae': mae.item(),
            'l2_error': l2_error.item(),
            'max_error': torch.max(torch.abs(our_result - reference_result)).item()
        }


class ActivationFunctionManager:
    """激活函数管理器"""
    
    def __init__(self):
        self.logger = get_logger()
        self.activation_functions: Dict[str, BaseActivationFunction] = {}
        self._register_default_functions()
    
    def _register_default_functions(self):
        """注册默认激活函数"""
        # 注册 Softmax
        self.register_function('softmax', SoftmaxActivation, SoftmaxConfig)
        
        # 注册其他激活函数
        from .layer_norm_activation import LayerNormActivation, LayerNormConfig
        from .rms_norm_activation import RMSNormActivation, RMSNormConfig
        from .silu_activation import SiLUActivation, SiLUConfig
        from .gelu_activation import GELUActivation, GELUConfig
        from .add_activation import AddActivation, AddConfig
        from .multiply_activation import MultiplyActivation, MultiplyConfig
        
        self.register_function('layer_norm', LayerNormActivation, LayerNormConfig)
        self.register_function('rms_norm', RMSNormActivation, RMSNormConfig)
        self.register_function('silu', SiLUActivation, SiLUConfig)
        self.register_function('gelu', GELUActivation, GELUConfig)
        self.register_function('add', AddActivation, AddConfig)
        self.register_function('multiply', MultiplyActivation, MultiplyConfig)
    
    def register_function(self, name: str, 
                         function_class: type, 
                         config_class: type):
        """注册激活函数"""
        self.activation_functions[name] = {
            'class': function_class,
            'config_class': config_class
        }
        self.logger.info(f"注册激活函数: {name}")
    
    def create_function(self, name: str, 
                       config: Optional[ActivationConfig] = None) -> BaseActivationFunction:
        """创建激活函数实例"""
        if name not in self.activation_functions:
            raise ActivationFunctionNotImplementedError(name)
        
        function_info = self.activation_functions[name]
        function_class = function_info['class']
        config_class = function_info['config_class']
        
        if config is None:
            # 尝试从全局配置中读取激活函数配置
            config = self._load_config_from_global(name, config_class)
        
        return function_class(config)
    
    def _load_config_from_global(self, function_name: str, config_class: type) -> ActivationConfig:
        """从全局配置中加载激活函数配置"""
        try:
            from config import get_config
            global_config = get_config()
            
            # 获取激活函数配置
            if hasattr(global_config, 'activation_functions') and function_name in global_config.activation_functions:
                function_config = global_config.activation_functions[function_name]
                
                # 创建配置对象
                config_dict = {}
                for field in config_class.__dataclass_fields__:
                    if field in function_config:
                        config_dict[field] = function_config[field]
                
                return config_class(**config_dict)
            else:
                self.logger.warning(f"未找到 {function_name} 的全局配置，使用默认配置")
                return config_class()
                
        except Exception as e:
            self.logger.warning(f"从全局配置加载 {function_name} 配置失败: {e}，使用默认配置")
            return config_class()
    
    def get_function(self, name: str) -> BaseActivationFunction:
        """获取激活函数实例"""
        if name not in self.activation_functions:
            raise ActivationFunctionNotImplementedError(name)
        
        return self.activation_functions[name]['class']
    
    def list_functions(self) -> List[str]:
        """列出所有可用的激活函数"""
        return list(self.activation_functions.keys())
    
    def benchmark_all_functions(self, input_shapes: List[tuple] = None,
                               num_runs: int = 100) -> Dict[str, Any]:
        """基准测试所有激活函数"""
        results = {}
        
        for name in self.list_functions():
            try:
                function = self.create_function(name)
                benchmark_result = function.benchmark(input_shapes, num_runs)
                results[name] = benchmark_result
                self.logger.info(f"完成 {name} 基准测试")
            except Exception as e:
                self.logger.error(f"{name} 基准测试失败: {e}")
                results[name] = {'error': str(e)}
        
        return results
    
    def accuracy_test_all_functions(self, reference_functions: Dict[str, Callable] = None) -> Dict[str, Any]:
        """精度测试所有激活函数"""
        results = {}
        
        for name in self.list_functions():
            try:
                function = self.create_function(name)
                reference_func = reference_functions.get(name) if reference_functions else None
                accuracy_result = function.accuracy_test(reference_func)
                results[name] = accuracy_result
                self.logger.info(f"完成 {name} 精度测试")
            except Exception as e:
                self.logger.error(f"{name} 精度测试失败: {e}")
                results[name] = {'error': str(e)}
        
        return results
    
    def get_optimization_suggestions(self) -> Dict[str, List[str]]:
        """获取优化建议"""
        suggestions = {}
        
        for name in self.list_functions():
            try:
                function = self.create_function(name)
                # 这里可以添加具体的优化建议逻辑
                suggestions[name] = [
                    f"考虑优化 {name} 的查找表大小",
                    f"尝试不同的插值方法",
                    f"启用定点数计算以提高硬件兼容性"
                ]
            except Exception as e:
                suggestions[name] = [f"无法获取 {name} 的优化建议: {e}"]
        
        return suggestions


# 全局激活函数管理器
_activation_manager: Optional[ActivationFunctionManager] = None


def get_activation_manager() -> ActivationFunctionManager:
    """获取全局激活函数管理器"""
    global _activation_manager
    if _activation_manager is None:
        _activation_manager = ActivationFunctionManager()
    return _activation_manager


def create_activation_function(name: str, 
                              config: Optional[ActivationConfig] = None) -> BaseActivationFunction:
    """创建激活函数便捷函数"""
    manager = get_activation_manager()
    return manager.create_function(name, config)


def benchmark_activation_functions(input_shapes: List[tuple] = None,
                                 num_runs: int = 100) -> Dict[str, Any]:
    """基准测试激活函数便捷函数"""
    manager = get_activation_manager()
    return manager.benchmark_all_functions(input_shapes, num_runs)
