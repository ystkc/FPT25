"""
内存优化模块
提供内存使用优化和管理功能
"""

import gc
import psutil
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Union, Callable
from contextlib import contextmanager
from dataclasses import dataclass
from collections import defaultdict

from core.base.constants import MEMORY_OPTIMIZATION, NUMERICAL_STABILITY
from core.base.exceptions import MemoryAllocationError, MemoryError
from core.base.logs import get_logger


@dataclass
class MemoryStats:
    """内存统计信息"""
    total_memory: int
    available_memory: int
    used_memory: int
    memory_percentage: float
    peak_memory: int
    tensor_count: int
    tensor_memory: int


class MemoryPool:
    """内存池管理器"""
    
    def __init__(self, max_size: int = MEMORY_OPTIMIZATION['max_memory_usage']):
        self.max_size = max_size
        self.allocated_memory = 0
        self.tensor_pool: Dict[tuple, List[torch.Tensor]] = defaultdict(list)
        self.logger = get_logger()
    
    def get_tensor(self, shape: tuple, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """从内存池获取张量"""
        key = (shape, dtype, device)
        
        if key in self.tensor_pool and self.tensor_pool[key]:
            tensor = self.tensor_pool[key].pop()
            self.logger.debug(f"从内存池获取张量: {shape}, {dtype}, {device}")
            return tensor
        
        # 检查内存限制
        required_memory = self._calculate_tensor_memory(shape, dtype)
        if self.allocated_memory + required_memory > self.max_size:
            self._cleanup_pool()
        
        # 创建新张量
        tensor = torch.zeros(shape, dtype=dtype, device=device)
        self.allocated_memory += required_memory
        self.logger.debug(f"创建新张量: {shape}, {dtype}, {device}, 内存: {required_memory}")
        
        return tensor
    
    def return_tensor(self, tensor: torch.Tensor) -> None:
        """将张量返回到内存池"""
        if tensor is None:
            return
        
        key = (tuple(tensor.shape), tensor.dtype, tensor.device)
        
        # 清理张量数据
        tensor.zero_()
        
        # 添加到池中
        self.tensor_pool[key].append(tensor)
        self.logger.debug(f"张量返回到内存池: {tensor.shape}, {tensor.dtype}, {tensor.device}")
    
    def _calculate_tensor_memory(self, shape: tuple, dtype: torch.dtype) -> int:
        """计算张量内存使用量"""
        element_size = torch.tensor(0, dtype=dtype).element_size()
        return np.prod(shape) * element_size
    
    def _cleanup_pool(self) -> None:
        """清理内存池"""
        self.logger.info("清理内存池")
        
        # 清空所有池中的张量
        for tensors in self.tensor_pool.values():
            for tensor in tensors:
                del tensor
            tensors.clear()
        
        # 强制垃圾回收
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.allocated_memory = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """获取内存池统计信息"""
        return {
            'max_size': self.max_size,
            'allocated_memory': self.allocated_memory,
            'pool_size': sum(len(tensors) for tensors in self.tensor_pool.values()),
            'tensor_types': len(self.tensor_pool)
        }


class MemoryOptimizer:
    """内存优化器"""
    
    def __init__(self, enable_pooling: bool = MEMORY_OPTIMIZATION['enable_memory_pooling']):
        self.enable_pooling = enable_pooling
        self.memory_pool = MemoryPool() if enable_pooling else None
        self.tensor_views: List[torch.Tensor] = []
        self.logger = get_logger()
    
    def get_memory_stats(self) -> MemoryStats:
        """获取内存统计信息"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # 获取系统内存信息
        system_memory = psutil.virtual_memory()
        
        # 计算张量内存使用
        tensor_count = 0
        tensor_memory = 0
        
        for obj in gc.get_objects():
            if isinstance(obj, torch.Tensor):
                tensor_count += 1
                tensor_memory += obj.element_size() * obj.nelement()
        
        return MemoryStats(
            total_memory=system_memory.total,
            available_memory=system_memory.available,
            used_memory=memory_info.rss,
            memory_percentage=system_memory.percent,
            peak_memory=memory_info.peak,
            tensor_count=tensor_count,
            tensor_memory=tensor_memory
        )
    
    def optimize_tensor_operations(self, tensor: torch.Tensor) -> torch.Tensor:
        """优化张量操作"""
        # 使用张量视图减少内存拷贝
        if self.enable_pooling and self.memory_pool:
            # 尝试从内存池获取相同形状的张量
            return self.memory_pool.get_tensor(
                tensor.shape, tensor.dtype, tensor.device
            )
        
        return tensor
    
    def create_tensor_view(self, tensor: torch.Tensor, 
                          start: int, end: int) -> torch.Tensor:
        """创建张量视图（避免内存拷贝）"""
        view = tensor[start:end]
        self.tensor_views.append(view)
        return view
    
    def cleanup_tensor_views(self) -> None:
        """清理张量视图"""
        self.tensor_views.clear()
        gc.collect()
    
    def optimize_memory_layout(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """优化内存布局"""
        # 按大小排序张量，提高缓存命中率
        sorted_tensors = sorted(tensors, key=lambda x: x.numel())
        
        # 确保张量在连续内存中
        optimized_tensors = []
        for tensor in sorted_tensors:
            if not tensor.is_contiguous():
                optimized_tensors.append(tensor.contiguous())
            else:
                optimized_tensors.append(tensor)
        
        return optimized_tensors
    
    def batch_operations(self, operation: Callable, 
                        data: List[torch.Tensor], 
                        batch_size: int = 1000) -> List[torch.Tensor]:
        """批处理操作以减少内存使用"""
        results = []
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batch_results = operation(batch)
            results.extend(batch_results)
            
            # 清理中间结果
            del batch
            gc.collect()
        
        return results
    
    def monitor_memory_usage(self, operation: Callable, *args, **kwargs) -> Any:
        """监控内存使用"""
        initial_stats = self.get_memory_stats()
        
        try:
            result = operation(*args, **kwargs)
            return result
        finally:
            final_stats = self.get_memory_stats()
            
            memory_delta = final_stats.used_memory - initial_stats.used_memory
            self.logger.info(f"操作内存使用: {memory_delta / 1024 / 1024:.2f} MB")
            
            if memory_delta > MEMORY_OPTIMIZATION['max_memory_usage'] * 0.8:
                self.logger.warning("内存使用接近限制，建议优化")
    
    @contextmanager
    def memory_context(self):
        """内存管理上下文"""
        initial_stats = self.get_memory_stats()
        
        try:
            yield
        finally:
            final_stats = self.get_memory_stats()
            memory_delta = final_stats.used_memory - initial_stats.used_memory
            
            if memory_delta > 0:
                self.logger.debug(f"上下文内存使用: {memory_delta / 1024 / 1024:.2f} MB")
            
            # 清理
            self.cleanup_tensor_views()
            if self.memory_pool:
                self.memory_pool._cleanup_pool()


class TensorMemoryManager:
    """张量内存管理器"""
    
    def __init__(self):
        self.tensors: Dict[str, torch.Tensor] = {}
        self.tensor_metadata: Dict[str, Dict[str, Any]] = {}
        self.logger = get_logger()
    
    def store_tensor(self, name: str, tensor: torch.Tensor, 
                    metadata: Optional[Dict[str, Any]] = None) -> None:
        """存储张量"""
        if name in self.tensors:
            self.logger.warning(f"张量 {name} 已存在，将被覆盖")
        
        self.tensors[name] = tensor
        self.tensor_metadata[name] = metadata or {}
        
        self.logger.debug(f"存储张量: {name}, 形状: {tensor.shape}, 类型: {tensor.dtype}")
    
    def get_tensor(self, name: str) -> Optional[torch.Tensor]:
        """获取张量"""
        return self.tensors.get(name)
    
    def remove_tensor(self, name: str) -> None:
        """移除张量"""
        if name in self.tensors:
            del self.tensors[name]
            del self.tensor_metadata[name]
            self.logger.debug(f"移除张量: {name}")
    
    def clear_tensors(self) -> None:
        """清空所有张量"""
        self.tensors.clear()
        self.tensor_metadata.clear()
        gc.collect()
        self.logger.info("清空所有张量")
    
    def get_tensor_info(self, name: str) -> Optional[Dict[str, Any]]:
        """获取张量信息"""
        if name not in self.tensors:
            return None
        
        tensor = self.tensors[name]
        return {
            'shape': tensor.shape,
            'dtype': tensor.dtype,
            'device': tensor.device,
            'memory_usage': tensor.element_size() * tensor.nelement(),
            'is_contiguous': tensor.is_contiguous(),
            'metadata': self.tensor_metadata[name]
        }
    
    def list_tensors(self) -> List[str]:
        """列出所有张量"""
        return list(self.tensors.keys())
    
    def get_memory_usage(self) -> Dict[str, int]:
        """获取内存使用情况"""
        total_memory = 0
        tensor_memory = {}
        
        for name, tensor in self.tensors.items():
            memory = tensor.element_size() * tensor.nelement()
            tensor_memory[name] = memory
            total_memory += memory
        
        return {
            'total_memory': total_memory,
            'tensor_count': len(self.tensors),
            'tensor_memory': tensor_memory
        }


class MemoryProfiler:
    """内存分析器"""
    
    def __init__(self):
        self.memory_snapshots: List[MemoryStats] = []
        self.logger = get_logger()
    
    def take_snapshot(self, label: str = "") -> MemoryStats:
        """拍摄内存快照"""
        optimizer = MemoryOptimizer()
        stats = optimizer.get_memory_stats()
        
        self.memory_snapshots.append(stats)
        
        if label:
            self.logger.debug(f"内存快照 [{label}]: {stats.used_memory / 1024 / 1024:.2f} MB")
        
        return stats
    
    def compare_snapshots(self, index1: int, index2: int) -> Dict[str, Any]:
        """比较两个快照"""
        if index1 >= len(self.memory_snapshots) or index2 >= len(self.memory_snapshots):
            raise IndexError("快照索引超出范围")
        
        snap1 = self.memory_snapshots[index1]
        snap2 = self.memory_snapshots[index2]
        
        return {
            'memory_delta': snap2.used_memory - snap1.used_memory,
            'tensor_count_delta': snap2.tensor_count - snap1.tensor_count,
            'tensor_memory_delta': snap2.tensor_memory - snap1.tensor_memory,
            'memory_percentage_delta': snap2.memory_percentage - snap1.memory_percentage
        }
    
    def get_memory_trend(self) -> List[Dict[str, Any]]:
        """获取内存使用趋势"""
        if len(self.memory_snapshots) < 2:
            return []
        
        trends = []
        for i in range(1, len(self.memory_snapshots)):
            comparison = self.compare_snapshots(i - 1, i)
            trends.append(comparison)
        
        return trends
    
    def clear_snapshots(self) -> None:
        """清空快照"""
        self.memory_snapshots.clear()
        self.logger.info("清空内存快照")


# 全局内存管理器
_memory_optimizer: Optional[MemoryOptimizer] = None
_tensor_manager: Optional[TensorMemoryManager] = None
_memory_profiler: Optional[MemoryProfiler] = None


def get_memory_optimizer() -> MemoryOptimizer:
    """获取全局内存优化器"""
    global _memory_optimizer
    if _memory_optimizer is None:
        _memory_optimizer = MemoryOptimizer()
    return _memory_optimizer


def get_tensor_manager() -> TensorMemoryManager:
    """获取全局张量管理器"""
    global _tensor_manager
    if _tensor_manager is None:
        _tensor_manager = TensorMemoryManager()
    return _tensor_manager


def get_memory_profiler() -> MemoryProfiler:
    """获取全局内存分析器"""
    global _memory_profiler
    if _memory_profiler is None:
        _memory_profiler = MemoryProfiler()
    return _memory_profiler


@contextmanager
def memory_optimized():
    """内存优化上下文管理器"""
    optimizer = get_memory_optimizer()
    with optimizer.memory_context():
        yield optimizer
