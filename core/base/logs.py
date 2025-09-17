"""
日志配置模块
统一管理项目中的日志记录功能
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

from .constants import LOG_LEVELS, DEFAULT_LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT, OUTPUT_DIRS


class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""
    
    # 颜色代码
    COLORS = {
        'DEBUG': '\033[36m',      # 青色
        'INFO': '\033[32m',       # 绿色
        'WARNING': '\033[33m',    # 黄色
        'ERROR': '\033[31m',      # 红色
        'CRITICAL': '\033[35m',   # 紫色
        'RESET': '\033[0m'        # 重置
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录"""
        # 获取原始格式
        original_format = super().format(record)
        
        # 添加颜色
        if record.levelname in self.COLORS:
            color = self.COLORS[record.levelname]
            reset = self.COLORS['RESET']
            return f"{color}{original_format}{reset}"
        
        return original_format


class FPT25Logger:
    """FPT25项目专用日志器"""
    
    def __init__(self, name: str = "FPT25", level: str = DEFAULT_LOG_LEVEL):
        self.name = name
        self.level = level
        self.logger = logging.getLogger(name)
        self._setup_logger()
    
    def _setup_logger(self) -> None:
        """设置日志器"""
        # 清除现有的处理器
        self.logger.handlers.clear()
        
        # 设置日志级别
        self.logger.setLevel(getattr(logging, self.level.upper()))
        
        # 防止重复日志
        self.logger.propagate = False
        
        # 添加控制台处理器
        self._add_console_handler()
        
        # 添加文件处理器
        self._add_file_handler()
    
    def _add_console_handler(self) -> None:
        """添加控制台处理器"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, self.level.upper()))
        
        # 使用彩色格式化器
        formatter = ColoredFormatter(LOG_FORMAT, LOG_DATE_FORMAT)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
    
    def _add_file_handler(self) -> None:
        """添加文件处理器"""
        # 确保日志目录存在
        log_dir = Path(OUTPUT_DIRS['logs'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建日志文件路径
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = log_dir / f"fpt25_{timestamp}.log"
        
        # 创建文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # 文件记录所有级别
        
        # 使用标准格式化器
        formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
    
    def debug(self, message: str, **kwargs) -> None:
        """调试日志"""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """信息日志"""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """警告日志"""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """错误日志"""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """严重错误日志"""
        self.logger.critical(message, **kwargs)
    
    def exception(self, message: str, **kwargs) -> None:
        """异常日志（包含堆栈跟踪）"""
        self.logger.exception(message, **kwargs)
    
    def set_level(self, level: str) -> None:
        """设置日志级别"""
        if level.upper() not in LOG_LEVELS:
            raise ValueError(f"无效的日志级别: {level}, 支持: {LOG_LEVELS}")
        
        self.level = level.upper()
        self.logger.setLevel(getattr(logging, self.level))
        
        # 更新控制台处理器级别
        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handler.setLevel(getattr(logging, self.level))
    
    def add_handler(self, handler: logging.Handler) -> None:
        """添加自定义处理器"""
        self.logger.addHandler(handler)
    
    def remove_handler(self, handler: logging.Handler) -> None:
        """移除处理器"""
        self.logger.removeHandler(handler)


class PerformanceLogger:
    """性能专用日志器"""
    
    def __init__(self, logger: FPT25Logger):
        self.logger = logger
    
    def log_activation_function(self, function_name: str, input_shape: tuple, 
                              execution_time: float, accuracy: float) -> None:
        """记录激活函数执行信息"""
        message = (f"激活函数执行: {function_name}, "
                  f"输入形状: {input_shape}, "
                  f"执行时间: {execution_time:.6f}s, "
                  f"精度: {accuracy:.6f}")
        self.logger.info(message)
    
    def log_lookup_table_generation(self, table_type: str, point_count: int, 
                                  generation_time: float) -> None:
        """记录查找表生成信息"""
        message = (f"查找表生成: {table_type}, "
                  f"点数: {point_count}, "
                  f"生成时间: {generation_time:.6f}s")
        self.logger.info(message)
    
    def log_optimization(self, optimization_type: str, before_value: float, 
                        after_value: float, improvement: float) -> None:
        """记录优化信息"""
        message = (f"优化完成: {optimization_type}, "
                  f"优化前: {before_value:.6f}, "
                  f"优化后: {after_value:.6f}, "
                  f"改进: {improvement:.2%}")
        self.logger.info(message)
    
    def log_benchmark(self, test_name: str, result: Dict[str, Any]) -> None:
        """记录基准测试信息"""
        message = f"基准测试: {test_name}, 结果: {result}"
        self.logger.info(message)
    
    def log_memory_usage(self, memory_usage: int, peak_memory: int) -> None:
        """记录内存使用信息"""
        message = (f"内存使用: 当前 {memory_usage / 1024 / 1024:.2f}MB, "
                  f"峰值 {peak_memory / 1024 / 1024:.2f}MB")
        self.logger.debug(message)


class AccuracyLogger:
    """精度专用日志器"""
    
    def __init__(self, logger: FPT25Logger):
        self.logger = logger
    
    def log_accuracy_test(self, function_name: str, error: float, 
                         threshold: float, passed: bool) -> None:
        """记录精度测试信息"""
        status = "通过" if passed else "失败"
        message = (f"精度测试 {status}: {function_name}, "
                  f"误差: {error:.2e}, "
                  f"阈值: {threshold:.2e}")
        
        if passed:
            self.logger.info(message)
        else:
            self.logger.error(message)
    
    def log_accuracy_comparison(self, function_name: str, 
                              fpga_error: float, reference_error: float) -> None:
        """记录精度对比信息"""
        message = (f"精度对比: {function_name}, "
                  f"FPGA误差: {fpga_error:.2e}, "
                  f"参考误差: {reference_error:.2e}")
        self.logger.info(message)


# 全局日志器实例
_global_logger: Optional[FPT25Logger] = None
_performance_logger: Optional[PerformanceLogger] = None
_accuracy_logger: Optional[AccuracyLogger] = None


def get_logger(name: str = "FPT25", level: str = DEFAULT_LOG_LEVEL) -> FPT25Logger:
    """获取日志器实例"""
    global _global_logger
    if _global_logger is None:
        _global_logger = FPT25Logger(name, level)
    return _global_logger


def get_performance_logger() -> PerformanceLogger:
    """获取性能日志器实例"""
    global _performance_logger
    if _performance_logger is None:
        _performance_logger = PerformanceLogger(get_logger())
    return _performance_logger


def get_accuracy_logger() -> AccuracyLogger:
    """获取精度日志器实例"""
    global _accuracy_logger
    if _accuracy_logger is None:
        _accuracy_logger = AccuracyLogger(get_logger())
    return _accuracy_logger


def setup_logging(level: str = DEFAULT_LOG_LEVEL, 
                 log_file: Optional[str] = None) -> FPT25Logger:
    """设置项目日志"""
    logger = get_logger(level=level)
    
    if log_file:
        # 添加自定义文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
        file_handler.setFormatter(formatter)
        logger.add_handler(file_handler)
    
    return logger


def log_function_call(func_name: str, args: tuple = (), kwargs: dict = None):
    """函数调用装饰器"""
    def decorator(func):
        def wrapper(*fargs, **fkwargs):
            logger = get_logger()
            logger.debug(f"调用函数: {func_name}, 参数: {fargs}, 关键字参数: {fkwargs or {}}")
            try:
                result = func(*fargs, **fkwargs)
                logger.debug(f"函数 {func_name} 执行成功")
                return result
            except Exception as e:
                logger.error(f"函数 {func_name} 执行失败: {str(e)}")
                raise
        return wrapper
    return decorator


def log_execution_time(func_name: str):
    """执行时间记录装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            logger = get_logger()
            
            start_time = time.time()
            logger.debug(f"开始执行: {func_name}")
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(f"执行完成: {func_name}, 耗时: {execution_time:.6f}s")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"执行失败: {func_name}, 耗时: {execution_time:.6f}s, 错误: {str(e)}")
                raise
        return wrapper
    return decorator
