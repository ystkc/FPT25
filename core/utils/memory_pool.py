"""
内存池管理模块
提供高效的内存分配和回收机制，减少内存碎片
"""

import torch
import gc
import threading
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from collections import defaultdict, deque
import weakref

from core.base.constants import MEMORY_OPTIMIZATION
from core.base.logs import get_logger


@dataclass
class MemoryBlock:
    """内存块"""
    tensor: torch.Tensor
    size: int
    dtype: torch.dtype
    device: torch.device
    allocated_time: float
    last_used_time: float
    is_allocated: bool = True


class MemoryPool:
    """内存池管理器"""
    
    def __init__(self, max_pool_size: int = MEMORY_OPTIMIZATION['max_memory_usage']):
        self.max_pool_size = max_pool_size
        self.logger = get_logger()
        self._lock = threading.RLock()
        
        # 按大小和类型分组的空闲内存块
        self._free_blocks: Dict[Tuple[torch.dtype, torch.device], deque] = defaultdict(deque)
        
        # 活跃的内存块
        self._allocated_blocks: Dict[int, MemoryBlock] = {}
        
        # 内存使用统计
        self._total_allocated = 0
        self._total_freed = 0
        self._peak_usage = 0
        
        # 弱引用跟踪，用于自动回收
        self._weak_refs: Dict[int, weakref.ref] = {}
    
    def allocate(self, shape: Tuple[int, ...], dtype: torch.dtype, 
                device: torch.device, requires_grad: bool = False) -> torch.Tensor:
        """
        分配内存块
        
        Args:
            shape: 张量形状
            dtype: 数据类型
            device: 设备
            requires_grad: 是否需要梯度
            
        Returns:
            分配的张量
        """
        with self._lock:
            # 计算所需大小
            size = self._calculate_size(shape, dtype)
            key = (dtype, device)
            
            # 尝试从池中获取合适的内存块
            tensor = self._get_from_pool(shape, dtype, device, requires_grad)
            
            if tensor is None:
                # 池中没有合适的内存块，创建新的
                tensor = self._create_new_tensor(shape, dtype, device, requires_grad)
            
            # 记录分配
            block_id = id(tensor)
            block = MemoryBlock(
                tensor=tensor,
                size=size,
                dtype=dtype,
                device=device,
                allocated_time=self._get_current_time(),
                last_used_time=self._get_current_time()
            )
            
            self._allocated_blocks[block_id] = block
            self._total_allocated += size
            self._peak_usage = max(self._peak_usage, self._total_allocated - self._total_freed)
            
            # 设置弱引用用于自动回收
            self._weak_refs[block_id] = weakref.ref(tensor, self._auto_reclaim)
            
            self.logger.debug(f"分配内存块: {shape}, {dtype}, {device}, 大小: {size}")
            return tensor
    
    def deallocate(self, tensor: torch.Tensor) -> bool:
        """
        释放内存块
        
        Args:
            tensor: 要释放的张量
            
        Returns:
            是否成功释放
        """
        with self._lock:
            block_id = id(tensor)
            
            if block_id not in self._allocated_blocks:
                return False
            
            block = self._allocated_blocks[block_id]
            
            # 检查是否可以重用
            if self._can_reuse(block):
                # 重置张量并放入池中
                self._reset_tensor(block.tensor)
                key = (block.dtype, block.device)
                self._free_blocks[key].append(block)
                block.is_allocated = False
            else:
                # 直接删除
                del block.tensor
                self._total_freed += block.size
            
            # 清理记录
            del self._allocated_blocks[block_id]
            if block_id in self._weak_refs:
                del self._weak_refs[block_id]
            
            self.logger.debug(f"释放内存块: {block.size} bytes")
            return True
    
    def _get_from_pool(self, shape: Tuple[int, ...], dtype: torch.dtype, 
                      device: torch.device, requires_grad: bool) -> Optional[torch.Tensor]:
        """从池中获取合适的内存块"""
        key = (dtype, device)
        
        if key not in self._free_blocks:
            return None
        
        # 查找合适大小的内存块
        for i, block in enumerate(self._free_blocks[key]):
            if (block.tensor.shape == shape and 
                block.tensor.requires_grad == requires_grad):
                # 找到合适的内存块
                tensor = self._free_blocks[key].pop(i)
                tensor.is_allocated = True
                tensor.last_used_time = self._get_current_time()
                return tensor.tensor
        
        return None
    
    def _create_new_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype, 
                          device: torch.device, requires_grad: bool) -> torch.Tensor:
        """创建新的张量"""
        return torch.empty(shape, dtype=dtype, device=device, requires_grad=requires_grad)
    
    def _calculate_size(self, shape: Tuple[int, ...], dtype: torch.dtype) -> int:
        """计算张量大小（字节）"""
        element_size = torch.tensor(0, dtype=dtype).element_size()
        return element_size * torch.tensor(shape).prod().item()
    
    def _can_reuse(self, block: MemoryBlock) -> bool:
        """检查内存块是否可以重用"""
        # 检查内存池大小限制
        current_usage = self._total_allocated - self._total_freed
        if current_usage + block.size > self.max_pool_size:
            return False
        
        # 检查内存块是否过大
        if block.size > self.max_pool_size // 4:  # 单个块不超过总池的1/4
            return False
        
        return True
    
    def _reset_tensor(self, tensor: torch.Tensor):
        """重置张量"""
        tensor.zero_()
        tensor.requires_grad_(False)
    
    def _auto_reclaim(self, weak_ref):
        """自动回收内存"""
        # 当张量被垃圾回收时自动调用
        pass
    
    def _get_current_time(self) -> float:
        """获取当前时间"""
        import time
        return time.time()
    
    def clear_pool(self):
        """清空内存池"""
        with self._lock:
            # 清空所有空闲块
            for blocks in self._free_blocks.values():
                for block in blocks:
                    del block.tensor
            self._free_blocks.clear()
            
            # 强制垃圾回收
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.logger.info("内存池已清空")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取内存池统计信息"""
        with self._lock:
            current_usage = self._total_allocated - self._total_freed
            free_blocks_count = sum(len(blocks) for blocks in self._free_blocks.values())
            allocated_blocks_count = len(self._allocated_blocks)
            
            return {
                'total_allocated': self._total_allocated,
                'total_freed': self._total_freed,
                'current_usage': current_usage,
                'peak_usage': self._peak_usage,
                'free_blocks_count': free_blocks_count,
                'allocated_blocks_count': allocated_blocks_count,
                'pool_utilization': current_usage / self.max_pool_size if self.max_pool_size > 0 else 0
            }
    
    def optimize_pool(self):
        """优化内存池"""
        with self._lock:
            # 清理长时间未使用的内存块
            current_time = self._get_current_time()
            cleanup_threshold = 60.0  # 60秒
            
            for key, blocks in self._free_blocks.items():
                # 从后往前遍历，移除长时间未使用的块
                i = len(blocks) - 1
                while i >= 0:
                    block = blocks[i]
                    if current_time - block.last_used_time > cleanup_threshold:
                        del block.tensor
                        blocks.pop(i)
                    i -= 1
            
            # 强制垃圾回收
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.logger.info("内存池优化完成")


class TensorMemoryManager:
    """张量内存管理器"""
    
    def __init__(self, memory_pool: Optional[MemoryPool] = None):
        self.memory_pool = memory_pool or MemoryPool()
        self.logger = get_logger()
    
    def create_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype, 
                     device: torch.device, requires_grad: bool = False) -> torch.Tensor:
        """创建张量"""
        return self.memory_pool.allocate(shape, dtype, device, requires_grad)
    
    def release_tensor(self, tensor: torch.Tensor) -> bool:
        """释放张量"""
        return self.memory_pool.deallocate(tensor)
    
    def create_like(self, reference_tensor: torch.Tensor, 
                   shape: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
        """创建与参考张量相似的张量"""
        if shape is None:
            shape = reference_tensor.shape
        
        return self.create_tensor(
            shape, 
            reference_tensor.dtype, 
            reference_tensor.device,
            reference_tensor.requires_grad
        )
    
    def clone_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """克隆张量"""
        new_tensor = self.create_like(tensor)
        new_tensor.copy_(tensor)
        return new_tensor
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取内存统计"""
        return self.memory_pool.get_stats()
    
    def optimize_memory(self):
        """优化内存"""
        self.memory_pool.optimize_pool()


class MemoryContext:
    """内存上下文管理器"""
    
    def __init__(self, memory_manager: TensorMemoryManager):
        self.memory_manager = memory_manager
        self.allocated_tensors: List[torch.Tensor] = []
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 自动释放所有分配的张量
        for tensor in self.allocated_tensors:
            self.memory_manager.release_tensor(tensor)
        self.allocated_tensors.clear()
    
    def allocate(self, shape: Tuple[int, ...], dtype: torch.dtype, 
                device: torch.device, requires_grad: bool = False) -> torch.Tensor:
        """在上下文中分配张量"""
        tensor = self.memory_manager.create_tensor(shape, dtype, device, requires_grad)
        self.allocated_tensors.append(tensor)
        return tensor


# 全局内存管理器
_memory_manager: Optional[TensorMemoryManager] = None


def get_memory_manager() -> TensorMemoryManager:
    """获取全局内存管理器"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = TensorMemoryManager()
    return _memory_manager


def create_tensor_with_pool(shape: Tuple[int, ...], dtype: torch.dtype, 
                           device: torch.device, requires_grad: bool = False) -> torch.Tensor:
    """使用内存池创建张量"""
    manager = get_memory_manager()
    return manager.create_tensor(shape, dtype, device, requires_grad)


def release_tensor_to_pool(tensor: torch.Tensor) -> bool:
    """将张量释放到内存池"""
    manager = get_memory_manager()
    return manager.release_tensor(tensor)


def memory_context():
    """内存上下文管理器"""
    manager = get_memory_manager()
    return MemoryContext(manager)
