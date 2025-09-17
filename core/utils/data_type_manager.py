"""
数据类型管理器
统一管理bf16数据类型处理，确保精度和兼容性
"""

import torch
import numpy as np
from typing import Union, Optional, Tuple, Any
from enum import Enum

from core.base.constants import SUPPORTED_DTYPES, DEFAULT_DTYPE, EPSILON_TINY
from core.base.exceptions import UnsupportedDataTypeError, validate_dtype
from core.base.logs import get_logger


class DataType(Enum):
    """支持的数据类型枚举"""
    FLOAT32 = "float32"
    BFLOAT16 = "bfloat16"


class DataTypeManager:
    """数据类型管理器"""
    
    def __init__(self):
        self.logger = get_logger()
        self._bf16_supported = self._check_bf16_support()
        
        if not self._bf16_supported:
            self.logger.warning("系统不支持bfloat16，将使用float32作为替代")
    
    def _check_bf16_support(self) -> bool:
        """检查系统是否支持bfloat16"""
        try:
            # 尝试创建bfloat16张量
            test_tensor = torch.tensor([1.0], dtype=torch.bfloat16)
            return True
        except (RuntimeError, TypeError):
            return False
    
    def ensure_dtype(self, tensor: torch.Tensor, target_dtype: str) -> torch.Tensor:
        """
        确保张量具有目标数据类型
        
        Args:
            tensor: 输入张量
            target_dtype: 目标数据类型
            
        Returns:
            转换后的张量
        """
        validate_dtype(target_dtype, SUPPORTED_DTYPES)
        
        if target_dtype == 'bfloat16':
            return self._ensure_bf16(tensor)
        elif target_dtype == 'float32':
            return self._ensure_float32(tensor)
        else:
            raise UnsupportedDataTypeError(target_dtype, SUPPORTED_DTYPES)
    
    def _ensure_bf16(self, tensor: torch.Tensor) -> torch.Tensor:
        """确保张量为bf16类型"""
        if tensor.dtype == torch.bfloat16:
            return tensor
        
        if not self._bf16_supported:
            self.logger.warning("系统不支持bfloat16，使用float32作为替代")
            return tensor.to(torch.float32)
        
        # 先转换为fp32进行精确计算，然后转换为bf16
        if tensor.dtype != torch.float32:
            tensor = tensor.to(torch.float32)
        
        return tensor.to(torch.bfloat16)
    
    def _ensure_float32(self, tensor: torch.Tensor) -> torch.Tensor:
        """确保张量为float32类型"""
        if tensor.dtype == torch.float32:
            return tensor
        
        return tensor.to(torch.float32)
    
    def create_tensor(self, data: Union[list, np.ndarray, torch.Tensor], 
                     dtype: str, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        创建指定类型的张量
        
        Args:
            data: 张量数据
            dtype: 数据类型
            device: 设备
            
        Returns:
            创建的张量
        """
        validate_dtype(dtype, SUPPORTED_DTYPES)
        
        if dtype == 'bfloat16':
            return self._create_bf16_tensor(data, device)
        elif dtype == 'float32':
            return self._create_float32_tensor(data, device)
        else:
            raise UnsupportedDataTypeError(dtype, SUPPORTED_DTYPES)
    
    def _create_bf16_tensor(self, data: Union[list, np.ndarray, torch.Tensor], 
                           device: Optional[torch.device] = None) -> torch.Tensor:
        """创建bf16张量"""
        if not self._bf16_supported:
            self.logger.warning("系统不支持bfloat16，使用float32作为替代")
            return self._create_float32_tensor(data, device)
        
        # 先创建float32张量，然后转换为bf16
        if isinstance(data, torch.Tensor):
            tensor = data.to(torch.float32)
        else:
            tensor = torch.tensor(data, dtype=torch.float32)
        
        if device is not None:
            tensor = tensor.to(device)
        
        return tensor.to(torch.bfloat16)
    
    def _create_float32_tensor(self, data: Union[list, np.ndarray, torch.Tensor], 
                              device: Optional[torch.device] = None) -> torch.Tensor:
        """创建float32张量"""
        if isinstance(data, torch.Tensor):
            tensor = data.to(torch.float32)
        else:
            tensor = torch.tensor(data, dtype=torch.float32)
        
        if device is not None:
            tensor = tensor.to(device)
        
        return tensor
    
    def convert_for_computation(self, tensor: torch.Tensor, 
                              preserve_original: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        为计算转换张量类型
        
        Args:
            tensor: 输入张量
            preserve_original: 是否保留原始张量
            
        Returns:
            (计算用张量, 原始张量)
        """
        original_tensor = tensor if preserve_original else None
        
        # 对于bf16，转换为fp32进行计算
        if tensor.dtype == torch.bfloat16:
            compute_tensor = tensor.to(torch.float32)
        else:
            compute_tensor = tensor
        
        return compute_tensor, original_tensor
    
    def convert_back(self, compute_tensor: torch.Tensor, 
                    original_tensor: torch.Tensor) -> torch.Tensor:
        """
        将计算结果转换回原始类型
        
        Args:
            compute_tensor: 计算结果张量
            original_tensor: 原始张量
            
        Returns:
            转换后的张量
        """
        if original_tensor is None:
            return compute_tensor
        
        return compute_tensor.to(original_tensor.dtype)
    
    def safe_bf16_operation(self, operation: callable, *args, **kwargs) -> torch.Tensor:
        """
        安全的bf16操作
        
        Args:
            operation: 操作函数
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            操作结果
        """
        # 检查是否有bf16张量
        has_bf16 = any(isinstance(arg, torch.Tensor) and arg.dtype == torch.bfloat16 
                      for arg in args)
        
        if not has_bf16:
            return operation(*args, **kwargs)
        
        # 转换所有张量为fp32
        converted_args = []
        original_tensors = []
        
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.dtype == torch.bfloat16:
                converted_args.append(arg.to(torch.float32))
                original_tensors.append(arg)
            else:
                converted_args.append(arg)
                original_tensors.append(None)
        
        # 执行操作
        result = operation(*converted_args, **kwargs)
        
        # 转换结果回bf16
        if isinstance(result, torch.Tensor):
            # 找到第一个bf16张量作为参考
            for orig_tensor in original_tensors:
                if orig_tensor is not None:
                    result = result.to(orig_tensor.dtype)
                    break
        
        return result
    
    def get_optimal_dtype(self, precision_requirement: float = 1e-6) -> str:
        """
        根据精度要求获取最优数据类型
        
        Args:
            precision_requirement: 精度要求
            
        Returns:
            最优数据类型
        """
        if precision_requirement <= 1e-4 and self._bf16_supported:
            return 'bfloat16'
        else:
            return 'float32'
    
    def is_bf16_supported(self) -> bool:
        """检查是否支持bf16"""
        return self._bf16_supported
    
    def get_dtype_info(self, dtype: str) -> dict:
        """
        获取数据类型信息
        
        Args:
            dtype: 数据类型
            
        Returns:
            数据类型信息
        """
        validate_dtype(dtype, SUPPORTED_DTYPES)
        
        if dtype == 'bfloat16':
            return {
                'torch_dtype': torch.bfloat16,
                'supported': self._bf16_supported,
                'precision': 3.4e-38,  # bf16最小正数
                'max_value': 3.4e38,   # bf16最大数
                'bits': 16,
                'exponent_bits': 8,
                'mantissa_bits': 7
            }
        elif dtype == 'float32':
            return {
                'torch_dtype': torch.float32,
                'supported': True,
                'precision': 1.2e-38,  # fp32最小正数
                'max_value': 3.4e38,   # fp32最大数
                'bits': 32,
                'exponent_bits': 8,
                'mantissa_bits': 23
            }
        else:
            raise UnsupportedDataTypeError(dtype, SUPPORTED_DTYPES)


# 全局数据类型管理器
_data_type_manager: Optional[DataTypeManager] = None


def get_data_type_manager() -> DataTypeManager:
    """获取全局数据类型管理器"""
    global _data_type_manager
    if _data_type_manager is None:
        _data_type_manager = DataTypeManager()
    return _data_type_manager


def ensure_dtype(tensor: torch.Tensor, target_dtype: str) -> torch.Tensor:
    """确保张量类型的便捷函数"""
    manager = get_data_type_manager()
    return manager.ensure_dtype(tensor, target_dtype)


def create_tensor(data: Union[list, np.ndarray, torch.Tensor], 
                 dtype: str, device: Optional[torch.device] = None) -> torch.Tensor:
    """创建张量的便捷函数"""
    manager = get_data_type_manager()
    return manager.create_tensor(data, dtype, device)


def safe_bf16_operation(operation: callable, *args, **kwargs) -> torch.Tensor:
    """安全bf16操作的便捷函数"""
    manager = get_data_type_manager()
    return manager.safe_bf16_operation(operation, *args, **kwargs)
