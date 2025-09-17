"""
并行处理模块
提供多线程和多进程并行计算支持
"""

import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Callable, Optional, Union, Tuple
from dataclasses import dataclass
from queue import Queue, Empty
import time
import torch
import numpy as np

from core.base.constants import PARALLEL_PROCESSING
from core.base.exceptions import OptimizationError
from core.base.logs import get_logger


@dataclass
class ParallelConfig:
    """并行处理配置"""
    max_workers: int = PARALLEL_PROCESSING['max_workers']
    chunk_size: int = PARALLEL_PROCESSING['chunk_size']
    enable_vectorization: bool = PARALLEL_PROCESSING['enable_vectorization']
    use_multiprocessing: bool = False
    timeout: Optional[float] = None


class ParallelProcessor:
    """并行处理器"""
    
    def __init__(self, config: ParallelConfig):
        self.config = config
        self.logger = get_logger()
        self._thread_pool: Optional[ThreadPoolExecutor] = None
        self._process_pool: Optional[ProcessPoolExecutor] = None
    
    def __enter__(self):
        """上下文管理器入口"""
        self._initialize_pools()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self._cleanup_pools()
    
    def _initialize_pools(self):
        """初始化线程池和进程池"""
        if self.config.use_multiprocessing:
            self._process_pool = ProcessPoolExecutor(max_workers=self.config.max_workers)
        else:
            self._thread_pool = ThreadPoolExecutor(max_workers=self.config.max_workers)
    
    def _cleanup_pools(self):
        """清理线程池和进程池"""
        if self._thread_pool:
            self._thread_pool.shutdown(wait=True)
            self._thread_pool = None
        
        if self._process_pool:
            self._process_pool.shutdown(wait=True)
            self._process_pool = None
    
    def parallel_map(self, func: Callable, data: List[Any], 
                    chunk_size: Optional[int] = None) -> List[Any]:
        """并行映射操作"""
        if chunk_size is None:
            chunk_size = self.config.chunk_size
        
        # 将数据分块
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        
        # 选择执行器
        executor = self._process_pool if self.config.use_multiprocessing else self._thread_pool
        
        if executor is None:
            raise OptimizationError("并行处理器未初始化")
        
        # 提交任务
        future_to_chunk = {
            executor.submit(func, chunk): chunk for chunk in chunks
        }
        
        # 收集结果
        results = []
        for future in as_completed(future_to_chunk, timeout=self.config.timeout):
            try:
                chunk_result = future.result()
                if isinstance(chunk_result, list):
                    results.extend(chunk_result)
                else:
                    results.append(chunk_result)
            except Exception as e:
                self.logger.error(f"并行处理错误: {e}")
                raise OptimizationError(f"并行处理失败: {e}")
        
        return results
    
    def parallel_apply(self, func: Callable, data: List[Any], 
                      **kwargs) -> List[Any]:
        """并行应用函数"""
        executor = self._process_pool if self.config.use_multiprocessing else self._thread_pool
        
        if executor is None:
            raise OptimizationError("并行处理器未初始化")
        
        # 提交任务
        future_to_item = {
            executor.submit(func, item, **kwargs): item for item in data
        }
        
        # 收集结果
        results = []
        for future in as_completed(future_to_item, timeout=self.config.timeout):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                self.logger.error(f"并行应用错误: {e}")
                raise OptimizationError(f"并行应用失败: {e}")
        
        return results


class TensorParallelProcessor:
    """张量并行处理器"""
    
    def __init__(self, config: ParallelConfig):
        self.config = config
        self.logger = get_logger()
    
    def parallel_tensor_operation(self, operation: Callable, 
                                 tensors: List[torch.Tensor],
                                 dim: int = 0) -> torch.Tensor:
        """并行张量操作"""
        if not self.config.enable_vectorization:
            return self._sequential_tensor_operation(operation, tensors, dim)
        
        # 使用 PyTorch 的并行操作
        if len(tensors) == 1:
            return operation(tensors[0])
        
        # 将张量堆叠
        stacked_tensor = torch.stack(tensors, dim=dim)
        
        # 应用操作
        result = operation(stacked_tensor)
        
        return result
    
    def _sequential_tensor_operation(self, operation: Callable,
                                   tensors: List[torch.Tensor],
                                   dim: int = 0) -> torch.Tensor:
        """顺序张量操作"""
        results = []
        for tensor in tensors:
            result = operation(tensor)
            results.append(result)
        
        return torch.stack(results, dim=dim)
    
    def parallel_activation_functions(self, 
                                    activation_funcs: List[Callable],
                                    input_tensor: torch.Tensor) -> List[torch.Tensor]:
        """并行激活函数计算"""
        if not self.config.enable_vectorization:
            return self._sequential_activation_functions(activation_funcs, input_tensor)
        
        # 使用批处理
        batch_size = self.config.chunk_size
        results = []
        
        for i in range(0, len(activation_funcs), batch_size):
            batch_funcs = activation_funcs[i:i + batch_size]
            batch_results = self._process_activation_batch(batch_funcs, input_tensor)
            results.extend(batch_results)
        
        return results
    
    def _sequential_activation_functions(self, 
                                       activation_funcs: List[Callable],
                                       input_tensor: torch.Tensor) -> List[torch.Tensor]:
        """顺序激活函数计算"""
        results = []
        for func in activation_funcs:
            result = func(input_tensor)
            results.append(result)
        
        return results
    
    def _process_activation_batch(self, 
                                 activation_funcs: List[Callable],
                                 input_tensor: torch.Tensor) -> List[torch.Tensor]:
        """处理激活函数批次"""
        results = []
        for func in activation_funcs:
            result = func(input_tensor)
            results.append(result)
        
        return results


class LookupTableParallelProcessor:
    """查找表并行处理器"""
    
    def __init__(self, config: ParallelConfig):
        self.config = config
        self.logger = get_logger()
    
    def parallel_lookup(self, lookup_func: Callable, 
                       input_data: List[Any]) -> List[Any]:
        """并行查找表查询"""
        if len(input_data) <= self.config.chunk_size:
            return [lookup_func(x) for x in input_data]
        
        # 分块处理
        chunks = [input_data[i:i + self.config.chunk_size] 
                 for i in range(0, len(input_data), self.config.chunk_size)]
        
        results = []
        for chunk in chunks:
            chunk_results = [lookup_func(x) for x in chunk]
            results.extend(chunk_results)
        
        return results
    
    def parallel_table_generation(self, 
                                 table_configs: List[Dict[str, Any]]) -> List[Any]:
        """并行查找表生成"""
        from ..algorithms.lookup_table import LookupTable, LookupTableConfig
        
        results = []
        for config_dict in table_configs:
            config = LookupTableConfig(**config_dict)
            table = LookupTable(config)
            results.append(table)
        
        return results


class BatchProcessor:
    """批处理器"""
    
    def __init__(self, batch_size: int = PARALLEL_PROCESSING['chunk_size']):
        self.batch_size = batch_size
        self.logger = get_logger()
    
    def process_batches(self, data: List[Any], 
                       processor: Callable) -> List[Any]:
        """批处理数据"""
        results = []
        
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            batch_result = processor(batch)
            
            if isinstance(batch_result, list):
                results.extend(batch_result)
            else:
                results.append(batch_result)
            
            self.logger.debug(f"处理批次 {i // self.batch_size + 1}/{(len(data) - 1) // self.batch_size + 1}")
        
        return results
    
    def process_tensor_batches(self, tensor: torch.Tensor, 
                              processor: Callable,
                              dim: int = 0) -> torch.Tensor:
        """批处理张量"""
        if tensor.size(dim) <= self.batch_size:
            return processor(tensor)
        
        # 分割张量
        tensor_chunks = torch.chunk(tensor, 
                                   chunks=(tensor.size(dim) + self.batch_size - 1) // self.batch_size,
                                   dim=dim)
        
        # 处理每个块
        processed_chunks = []
        for i, chunk in enumerate(tensor_chunks):
            processed_chunk = processor(chunk)
            processed_chunks.append(processed_chunk)
            self.logger.debug(f"处理张量批次 {i + 1}/{len(tensor_chunks)}")
        
        # 合并结果
        return torch.cat(processed_chunks, dim=dim)


class PipelineProcessor:
    """流水线处理器"""
    
    def __init__(self, stages: List[Callable]):
        self.stages = stages
        self.logger = get_logger()
    
    def process_pipeline(self, data: Any) -> Any:
        """处理流水线"""
        current_data = data
        
        for i, stage in enumerate(self.stages):
            self.logger.debug(f"执行流水线阶段 {i + 1}/{len(self.stages)}")
            current_data = stage(current_data)
        
        return current_data
    
    def parallel_pipeline(self, data_list: List[Any]) -> List[Any]:
        """并行流水线处理"""
        results = []
        
        for i, data in enumerate(data_list):
            self.logger.debug(f"处理流水线 {i + 1}/{len(data_list)}")
            result = self.process_pipeline(data)
            results.append(result)
        
        return results


class WorkerPool:
    """工作池"""
    
    def __init__(self, num_workers: int = PARALLEL_PROCESSING['max_workers']):
        self.num_workers = num_workers
        self.workers: List[threading.Thread] = []
        self.task_queue: Queue = Queue()
        self.result_queue: Queue = Queue()
        self.running = False
        self.logger = get_logger()
    
    def start(self):
        """启动工作池"""
        self.running = True
        
        for i in range(self.num_workers):
            worker = threading.Thread(target=self._worker, args=(i,))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
        
        self.logger.info(f"启动工作池，工作线程数: {self.num_workers}")
    
    def stop(self):
        """停止工作池"""
        self.running = False
        
        # 等待所有任务完成
        self.task_queue.join()
        
        # 等待所有工作线程结束
        for worker in self.workers:
            worker.join()
        
        self.workers.clear()
        self.logger.info("工作池已停止")
    
    def submit_task(self, func: Callable, *args, **kwargs):
        """提交任务"""
        self.task_queue.put((func, args, kwargs))
    
    def get_result(self, timeout: Optional[float] = None) -> Any:
        """获取结果"""
        try:
            return self.result_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def _worker(self, worker_id: int):
        """工作线程"""
        while self.running:
            try:
                func, args, kwargs = self.task_queue.get(timeout=1)
                result = func(*args, **kwargs)
                self.result_queue.put(result)
                self.task_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"工作线程 {worker_id} 错误: {e}")
                self.task_queue.task_done()


# 全局并行处理器
_parallel_processor: Optional[ParallelProcessor] = None
_tensor_processor: Optional[TensorParallelProcessor] = None
_batch_processor: Optional[BatchProcessor] = None


def get_parallel_processor(config: Optional[ParallelConfig] = None) -> ParallelProcessor:
    """获取全局并行处理器"""
    global _parallel_processor
    if _parallel_processor is None:
        if config is None:
            config = ParallelConfig()
        _parallel_processor = ParallelProcessor(config)
    return _parallel_processor


def get_tensor_processor(config: Optional[ParallelConfig] = None) -> TensorParallelProcessor:
    """获取张量并行处理器"""
    global _tensor_processor
    if _tensor_processor is None:
        if config is None:
            config = ParallelConfig()
        _tensor_processor = TensorParallelProcessor(config)
    return _tensor_processor


def get_batch_processor(batch_size: Optional[int] = None) -> BatchProcessor:
    """获取批处理器"""
    global _batch_processor
    if _batch_processor is None:
        if batch_size is None:
            batch_size = PARALLEL_PROCESSING['chunk_size']
        _batch_processor = BatchProcessor(batch_size)
    return _batch_processor


def parallel_map(func: Callable, data: List[Any], 
                config: Optional[ParallelConfig] = None) -> List[Any]:
    """并行映射便捷函数"""
    processor = get_parallel_processor(config)
    return processor.parallel_map(func, data)


def parallel_apply(func: Callable, data: List[Any], 
                  config: Optional[ParallelConfig] = None, **kwargs) -> List[Any]:
    """并行应用便捷函数"""
    processor = get_parallel_processor(config)
    return processor.parallel_apply(func, data, **kwargs)
