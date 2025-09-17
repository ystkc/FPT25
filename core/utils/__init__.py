"""
工具模块
包含数据类型管理、内存池和智能缓存系统
"""

from .data_type_manager import (
    DataType,
    DataTypeManager,
    get_data_type_manager,
    ensure_dtype,
    create_tensor,
    safe_bf16_operation
)

from .memory_pool import (
    MemoryBlock,
    MemoryPool,
    TensorMemoryManager,
    MemoryContext,
    get_memory_manager,
    create_tensor_with_pool,
    release_tensor_to_pool,
    memory_context
)

from .smart_cache import (
    CacheEntry,
    SmartCache,
    FunctionCache,
    TensorCache,
    LookupTableCache,
    get_global_cache,
    get_tensor_cache,
    get_lookup_table_cache,
    cached_function,
    clear_all_caches
)

__all__ = [
    # 数据类型管理
    'DataType', 'DataTypeManager', 'get_data_type_manager',
    'ensure_dtype', 'create_tensor', 'safe_bf16_operation',
    
    # 内存池
    'MemoryBlock', 'MemoryPool', 'TensorMemoryManager', 'MemoryContext',
    'get_memory_manager', 'create_tensor_with_pool', 'release_tensor_to_pool',
    'memory_context',
    
    # 智能缓存
    'CacheEntry', 'SmartCache', 'FunctionCache', 'TensorCache',
    'LookupTableCache', 'get_global_cache', 'get_tensor_cache',
    'get_lookup_table_cache', 'cached_function', 'clear_all_caches'
]
