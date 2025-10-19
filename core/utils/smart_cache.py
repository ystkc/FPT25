"""
智能缓存模块
提供高效的缓存机制，减少重复计算
"""

import hashlib
import pickle
import httpx
import time
import threading
from typing import Dict, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict
import weakref
import torch
import numpy as np

from core.base.logs import get_logger


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    created_time: float
    last_accessed_time: float
    access_count: int = 0
    size_bytes: int = 0
    is_valid: bool = True


class SmartCache:
    """智能缓存"""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.logger = get_logger()
        self._lock = threading.RLock()
        
        # 使用OrderedDict实现LRU缓存
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # 内存使用统计
        self._current_memory_usage = 0
        
        # 缓存统计
        self._hits = 0
        self._misses = 0
        self._evictions = 0
    
    def _generate_key(self, *args, **kwargs) -> str:
        """生成缓存键"""
        # 将参数序列化为字符串
        key_data = []
        
        for arg in args:
            if isinstance(arg, torch.Tensor):
                # 对张量使用哈希
                key_data.append(f"tensor_{self._tensor_hash(arg)}")
            elif isinstance(arg, np.ndarray):
                key_data.append(f"array_{self._array_hash(arg)}")
            else:
                key_data.append(str(arg))
        
        # 添加关键字参数
        for k, v in sorted(kwargs.items()):
            if isinstance(v, torch.Tensor):
                key_data.append(f"{k}_tensor_{self._tensor_hash(v)}")
            elif isinstance(v, np.ndarray):
                key_data.append(f"{k}_array_{self._array_hash(v)}")
            else:
                key_data.append(f"{k}_{v}")
        
        # 生成MD5哈希
        key_string = "|".join(key_data)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _tensor_hash(self, tensor: torch.Tensor) -> str:
        """计算张量哈希"""
        # 使用张量的形状、数据类型和部分数据计算哈希
        shape_str = str(tensor.shape)
        dtype_str = str(tensor.dtype)
        
        # 对大数据张量，只使用部分数据计算哈希
        if tensor.numel() > 1000:
            # 使用前100个元素
            data_sample = tensor.flatten()[:100].detach().cpu().numpy()
        else:
            data_sample = tensor.detach().cpu().numpy()
        
        data_str = hashlib.md5(data_sample.tobytes()).hexdigest()
        return f"{shape_str}_{dtype_str}_{data_str}"
    
    def _array_hash(self, array: np.ndarray) -> str:
        """计算数组哈希"""
        shape_str = str(array.shape)
        dtype_str = str(array.dtype)
        
        if array.size > 1000:
            data_sample = array.flatten()[:100]
        else:
            data_sample = array
        
        data_str = hashlib.md5(data_sample.tobytes()).hexdigest()
        return f"{shape_str}_{dtype_str}_{data_str}"
    
    def _calculate_size(self, value: Any) -> int:
        """计算值的大小（字节）"""
        if isinstance(value, torch.Tensor):
            return value.numel() * value.element_size()
        elif isinstance(value, np.ndarray):
            return value.nbytes
        elif isinstance(value, (list, tuple)):
            return sum(self._calculate_size(item) for item in value)
        elif isinstance(value, dict):
            return sum(self._calculate_size(v) for v in value.values())
        else:
            # 对于其他类型，估算大小
            return len(str(value).encode())
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            entry = self._cache[key]
            
            # 检查是否过期
            if not entry.is_valid:
                del self._cache[key]
                self._misses += 1
                return None
            
            # 更新访问信息
            entry.last_accessed_time = time.time()
            entry.access_count += 1
            
            # 移动到末尾（LRU）
            self._cache.move_to_end(key)
            
            self._hits += 1
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """设置缓存值"""
        with self._lock:
            # 计算大小
            size_bytes = self._calculate_size(value)
            
            # 如果单个值太大，不缓存
            if size_bytes > self.max_memory_bytes:
                self.logger.warning(f"值太大，不缓存: {size_bytes} bytes")
                return
            
            # 创建缓存条目
            entry = CacheEntry(
                key=key,
                value=value,
                created_time=time.time(),
                last_accessed_time=time.time(),
                size_bytes=size_bytes
            )
            
            # 如果键已存在，先删除旧条目
            if key in self._cache:
                old_entry = self._cache[key]
                self._current_memory_usage -= old_entry.size_bytes
                del self._cache[key]
            
            # 检查是否需要清理空间
            self._evict_if_needed(size_bytes)
            
            # 添加新条目
            self._cache[key] = entry
            self._current_memory_usage += size_bytes
    
    def _evict_if_needed(self, required_size: int) -> None:
        """如果需要，清理缓存空间"""
        # 检查内存限制
        while (self._current_memory_usage + required_size > self.max_memory_bytes or 
               len(self._cache) >= self.max_size):
            
            if not self._cache:
                break
            
            # 移除最久未使用的条目
            oldest_key, oldest_entry = self._cache.popitem(last=False)
            self._current_memory_usage -= oldest_entry.size_bytes
            self._evictions += 1
            
            self.logger.debug(f"清理缓存条目: {oldest_key}")
    
    def invalidate(self, key: str) -> bool:
        """使缓存条目失效"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._current_memory_usage = 0
            self.logger.info("缓存已清空")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'memory_usage_bytes': self._current_memory_usage,
                'max_memory_bytes': self.max_memory_bytes,
                'memory_usage_percent': (self._current_memory_usage / self.max_memory_bytes) * 100,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'evictions': self._evictions
            }
    
    def cleanup_expired(self, ttl: float = 3600) -> int:
        """清理过期条目"""
        with self._lock:
            current_time = time.time()
            expired_keys = []
            
            for key, entry in self._cache.items():
                if current_time - entry.last_accessed_time > ttl:
                    expired_keys.append(key)
            
            for key in expired_keys:
                entry = self._cache[key]
                self._current_memory_usage -= entry.size_bytes
                del self._cache[key]
            
            if expired_keys:
                self.logger.info(f"清理了 {len(expired_keys)} 个过期缓存条目")
            
            return len(expired_keys)


class FunctionCache:
    """函数缓存装饰器"""
    
    def __init__(self, cache: Optional[SmartCache] = None, ttl: Optional[float] = None):
        self.cache = cache or SmartCache()
        self.ttl = ttl
        self.logger = get_logger()
    
    def __call__(self, func: Callable) -> Callable:
        """装饰器实现"""
        def wrapper(*args, **kwargs):
            # 生成缓存键
            key = self.cache._generate_key(func.__name__, *args, **kwargs)
            
            # 尝试从缓存获取
            result = self.cache.get(key)
            if result is not None:
                self.logger.debug(f"缓存命中: {func.__name__}")
                return result
            
            # 执行函数
            self.logger.debug(f"缓存未命中: {func.__name__}")
            result = func(*args, **kwargs)
            
            # 存储到缓存
            self.cache.set(key, result, self.ttl)
            
            return result
        
        return wrapper


class TensorCache:
    """张量专用缓存"""
    
    def __init__(self, max_size: int = 100):
        self.cache = SmartCache(max_size, max_memory_mb=500)  # 500MB
        self.logger = get_logger()
    
    def get_tensor(self, key: str) -> Optional[torch.Tensor]:
        """获取张量"""
        return self.cache.get(key)
    
    def set_tensor(self, key: str, tensor: torch.Tensor) -> None:
        """设置张量"""
        # 克隆张量以避免引用问题
        tensor_clone = tensor.clone().detach()
        self.cache.set(key, tensor_clone)
    
    def get_or_compute(self, key: str, compute_func: Callable[[], torch.Tensor]) -> torch.Tensor:
        """获取或计算张量"""
        result = self.get_tensor(key)
        if result is not None:
            return result
        
        result = compute_func()
        self.set_tensor(key, result)
        return result


class LookupTableCache:
    """查找表专用缓存"""
    
    def __init__(self):
        # self.cache = SmartCache(max_size=50, max_memory_mb=200)  # 200MB
        self.logger = get_logger()
        # 检测localhost:8000是否可用
        try:
            httpx.get("http://localhost:8000")
        except:
            self.logger.warning("查找表缓存服务未启动，请先启动服务")
    
    def get_table(self, config_hash: str) -> Optional[Any]:
        """获取查找表"""
        try:
            response = httpx.get(f"http://localhost:8000/?hash_key={config_hash}")
            rep = pickle.loads(response.content)
            self.logger.info(f"获取查找表: {config_hash} ({len(response.content)}bytes)")
            return rep
        except:
            return None
        # return self.cache.get(f"lookup_table_{config_hash}")
    
    def set_table(self, config_hash: str, table: Any) -> None:
        """设置查找表"""
        try:
            response = httpx.post(f"http://localhost:8000/?hash_key={config_hash}", data=pickle.dumps(table, protocol=5), headers={"Content-Type": "application/octet-stream"})
            self.logger.info(f"设置查找表: {config_hash} {response.text}")
        except:
            self.logger.warning("查找表缓存服务未启动，请先启动服务")
            raise


# 全局缓存实例
_global_cache: Optional[SmartCache] = None
_tensor_cache: Optional[TensorCache] = None
_lookup_table_cache: Optional[LookupTableCache] = None


def get_global_cache() -> SmartCache:
    """获取全局缓存"""
    global _global_cache
    if _global_cache is None:
        _global_cache = SmartCache()
    return _global_cache


def get_tensor_cache() -> TensorCache:
    """获取张量缓存"""
    global _tensor_cache
    if _tensor_cache is None:
        _tensor_cache = TensorCache()
    return _tensor_cache


def get_lookup_table_cache() -> LookupTableCache:
    """获取查找表缓存"""
    global _lookup_table_cache
    if _lookup_table_cache is None:
        _lookup_table_cache = LookupTableCache()
    return _lookup_table_cache


def cached_function(ttl: Optional[float] = None):
    """函数缓存装饰器"""
    cache = get_global_cache()
    return FunctionCache(cache, ttl)


def clear_all_caches():
    """清空所有缓存"""
    get_global_cache().clear()
    get_tensor_cache().cache.clear()
    get_lookup_table_cache().cache.clear()
