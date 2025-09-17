"""
日志配置模块
统一管理项目中的日志记录功能
"""

import logging
import os
import sys
import tarfile
import gzip
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
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
        # 初始化时检查并压缩旧日志文件
        self._compress_old_logs()
        # 记录当前日志文件路径，用于定期检查
        self._current_log_file = None
    
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
        
        # 记录当前日志文件路径
        self._current_log_file = log_file
        
        # 创建文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # 文件记录所有级别
        
        # 使用标准格式化器
        formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
    
    def debug(self, message: str, **kwargs) -> None:
        """调试日志"""
        self._check_and_compress_if_needed()
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """信息日志"""
        self._check_and_compress_if_needed()
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """警告日志"""
        self._check_and_compress_if_needed()
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """错误日志"""
        self._check_and_compress_if_needed()
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """严重错误日志"""
        self._check_and_compress_if_needed()
        self.logger.critical(message, **kwargs)
    
    def exception(self, message: str, **kwargs) -> None:
        """异常日志（包含堆栈跟踪）"""
        self._check_and_compress_if_needed()
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
    
    def _check_and_compress_if_needed(self) -> None:
        """检查当前日志文件大小，如果超过阈值则压缩"""
        try:
            if not self._current_log_file or not self._current_log_file.exists():
                return
            
            # 检查文件大小（100KB = 102400 bytes）
            file_size = self._current_log_file.stat().st_size
            if file_size > 100 * 1024:  # 100KB
                self.logger.info(f"当前日志文件 {self._current_log_file.name} 大小超过 100KB ({file_size/1024:.1f}KB)，开始压缩...")
                
                # 先关闭文件处理器，释放文件句柄
                self._close_file_handlers()
                
                # 压缩当前文件
                self._compress_single_log(self._current_log_file)
                
                # 重新创建文件处理器（因为原文件被压缩了）
                self._recreate_file_handler()
                
        except Exception as e:
            # 避免在压缩检查时出错影响正常日志记录
            self.logger.debug(f"压缩检查时出错: {str(e)}")
    
    def _close_file_handlers(self) -> None:
        """关闭所有文件处理器"""
        try:
            for handler in self.logger.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    handler.close()
                    self.logger.removeHandler(handler)
        except Exception as e:
            self.logger.debug(f"关闭文件处理器时出错: {str(e)}")
    
    def _recreate_file_handler(self) -> None:
        """重新创建文件处理器"""
        try:
            # 移除现有的文件处理器
            for handler in self.logger.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    self.logger.removeHandler(handler)
            
            # 重新添加文件处理器
            self._add_file_handler()
            
        except Exception as e:
            self.logger.error(f"重新创建文件处理器失败: {str(e)}")
    
    def _compress_old_logs(self) -> None:
        """压缩旧日志文件"""
        try:
            log_dir = Path(OUTPUT_DIRS['logs'])
            if not log_dir.exists():
                return
            
            # 获取所有日志文件
            log_files = list(log_dir.glob("fpt25_*.log"))
            
            # 按修改时间排序，保留最新的一个
            log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # 如果只有一个或没有日志文件，不需要压缩
            if len(log_files) <= 1:
                return
            
            # 保留最新的日志文件，压缩其他的
            files_to_compress = log_files[1:]  # 跳过最新的文件
            
            for log_file in files_to_compress:
                # 检查文件大小，只压缩大于100KB的文件
                if log_file.stat().st_size > 100 * 1024:  # 100KB
                    self._compress_single_log(log_file)
                else:
                    # 小文件直接删除
                    log_file.unlink()
                    self.logger.debug(f"删除小日志文件: {log_file.name}")
        
        except Exception as e:
            self.logger.error(f"压缩旧日志文件时出错: {str(e)}")
    
    def _compress_single_log(self, log_file: Path) -> None:
        """压缩单个日志文件"""
        try:
            # 创建压缩文件名
            compressed_file = log_file.with_suffix('.log.tar.gz')
            
            # 使用 gzip 压缩
            with open(log_file, 'rb') as f_in:
                with gzip.open(compressed_file, 'wb') as f_out:
                    f_out.writelines(f_in)
            
            # 删除原始文件
            log_file.unlink()
            
            # 记录压缩信息
            original_size = log_file.stat().st_size if log_file.exists() else 0
            compressed_size = compressed_file.stat().st_size
            compression_ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
            
            self.logger.info(f"压缩日志文件: {log_file.name} -> {compressed_file.name}, "
                           f"压缩率: {compression_ratio:.1f}%")
        
        except Exception as e:
            self.logger.error(f"压缩日志文件 {log_file.name} 失败: {str(e)}")
    
    def cleanup_old_compressed_logs(self, days_to_keep: int = 30) -> None:
        """清理过期的压缩日志文件"""
        try:
            log_dir = Path(OUTPUT_DIRS['logs'])
            if not log_dir.exists():
                return
            
            # 计算过期时间
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # 查找所有压缩的日志文件
            compressed_files = list(log_dir.glob("fpt25_*.log.tar.gz"))
            
            for compressed_file in compressed_files:
                # 检查文件修改时间
                file_time = datetime.fromtimestamp(compressed_file.stat().st_mtime)
                if file_time < cutoff_date:
                    compressed_file.unlink()
                    self.logger.info(f"删除过期压缩日志文件: {compressed_file.name}")
        
        except Exception as e:
            self.logger.error(f"清理过期压缩日志文件时出错: {str(e)}")
    
    def get_log_file_info(self) -> Dict[str, Any]:
        """获取日志文件信息"""
        try:
            log_dir = Path(OUTPUT_DIRS['logs'])
            if not log_dir.exists():
                return {"total_files": 0, "total_size": 0, "files": []}
            
            # 获取所有日志相关文件
            log_files = list(log_dir.glob("fpt25_*"))
            
            files_info = []
            total_size = 0
            
            for log_file in log_files:
                file_info = {
                    "name": log_file.name,
                    "size": log_file.stat().st_size,
                    "size_mb": log_file.stat().st_size / (1024 * 1024),
                    "modified": datetime.fromtimestamp(log_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                    "is_compressed": log_file.suffix == '.gz'
                }
                files_info.append(file_info)
                total_size += file_info["size"]
            
            # 按修改时间排序
            files_info.sort(key=lambda x: x["modified"], reverse=True)
            
            return {
                "total_files": len(files_info),
                "total_size": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "files": files_info
            }
        
        except Exception as e:
            self.logger.error(f"获取日志文件信息时出错: {str(e)}")
            return {"total_files": 0, "total_size": 0, "files": []}


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
    else:
        # 确保当前日志文件路径正确设置
        if _global_logger._current_log_file is None:
            log_dir = Path(OUTPUT_DIRS['logs'])
            timestamp = datetime.now().strftime("%Y%m%d")
            _global_logger._current_log_file = log_dir / f"fpt25_{timestamp}.log"
        
        # 每次获取日志器时都检查是否需要压缩
        _global_logger._check_and_compress_if_needed()
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


def compress_log_files() -> None:
    """手动压缩日志文件"""
    logger = get_logger()
    logger.info("开始手动压缩日志文件...")
    
    # 获取日志器实例并执行压缩
    fpt25_logger = FPT25Logger()
    fpt25_logger._compress_old_logs()
    
    # 显示压缩后的文件信息
    info = fpt25_logger.get_log_file_info()
    logger.info(f"日志文件压缩完成，当前共有 {info['total_files']} 个文件，"
               f"总大小: {info['total_size_mb']:.2f} MB")


def cleanup_old_logs(days_to_keep: int = 30) -> None:
    """清理过期的压缩日志文件"""
    logger = get_logger()
    logger.info(f"开始清理 {days_to_keep} 天前的压缩日志文件...")
    
    fpt25_logger = FPT25Logger()
    fpt25_logger.cleanup_old_compressed_logs(days_to_keep)
    
    logger.info("过期日志文件清理完成")


def show_log_info() -> None:
    """显示日志文件信息"""
    logger = get_logger()
    fpt25_logger = FPT25Logger()
    info = fpt25_logger.get_log_file_info()
    
    logger.info("=== 日志文件信息 ===")
    logger.info(f"总文件数: {info['total_files']}")
    logger.info(f"总大小: {info['total_size_mb']:.2f} MB")
    logger.info("文件列表:")
    
    for file_info in info['files']:
        status = "压缩" if file_info['is_compressed'] else "正常"
        logger.info(f"  - {file_info['name']} ({file_info['size_mb']:.2f} MB) [{status}] - {file_info['modified']}")


def decompress_log_file(compressed_file: str) -> None:
    """解压缩指定的日志文件"""
    try:
        import gzip
        from pathlib import Path
        
        compressed_path = Path(compressed_file)
        if not compressed_path.exists():
            print(f"文件不存在: {compressed_file}")
            return
        
        # 创建解压后的文件名
        decompressed_file = compressed_path.with_suffix('')  # 移除 .gz 后缀
        
        # 解压缩
        with gzip.open(compressed_path, 'rb') as f_in:
            with open(decompressed_file, 'wb') as f_out:
                f_out.write(f_in.read())
        
        print(f"解压缩完成: {compressed_file} -> {decompressed_file}")
        
    except Exception as e:
        print(f"解压缩失败: {str(e)}")
