"""
激活函数管理器
统一管理所有激活函数的创建、调用和优化
"""

import math
import time
import traceback
import torch
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass

from config.config import ActivationConfig
from core.algorithms.lookup_table import LookupTable, create_exp_table
from core.base.constants import ACTIVATION_FUNCTIONS, ACTIVATION_FUNCTION_WEIGHTS, DEFAULT_DTYPE, INTERPOLATION_METHODS, TENSOR_SHAPE
from core.base.exceptions import ActivationFunctionNotImplementedError
from core.base.logs import get_logger, get_performance_logger
from core.optimization import get_performance_profiler, performance_monitoring
from core.utils.data_type_manager import get_data_type_manager, ensure_dtype, safe_bf16_operation
from core.utils.memory_pool import MemoryContext, get_memory_manager, memory_context
from core.base.exceptions import (
    InvalidTensorShapeError, UnsupportedDataTypeError,
    validate_tensor_shape, validate_dtype
)
from config.config import SoftmaxConfig, LayerNormConfig, RMSNormConfig, SiLUConfig, GELUConfig, AddConfig, MultiplyConfig
from core.activation_functions.base_activation import BaseActivationFunction
from core.activation_functions.softmax_activation import SoftmaxActivation
from core.activation_functions.layer_norm_activation import LayerNormActivation
from core.activation_functions.rms_norm_activation import RMSNormActivation
from core.activation_functions.silu_activation import SiLUActivation
from core.activation_functions.gelu_activation import GELUActivation
from core.activation_functions.add_activation import AddActivation
from core.activation_functions.multiply_activation import MultiplyActivation

    
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
            if hasattr(global_config, 'activation_functions') and hasattr(global_config.activation_functions, function_name):
                function_config = getattr(global_config.activation_functions, function_name)
                if isinstance(function_config, config_class):
                    return function_config
                elif isinstance(function_config, dict):
                    return config_class(**function_config)
                else:
                    self.logger.warning(f"未知的 {function_name} 配置格式 {type(function_config)}，使用默认配置")
                    return config_class()
            else:
                self.logger.warning(f"未找到 {function_name} 的全局配置，使用默认配置")
                return config_class()
                
        except Exception as e:
            self.logger.warning(f"从全局配置加载 {function_name} 配置失败: {e}，使用默认配置")
            traceback.print_exc()
            exit(0)
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
