"""
异常定义模块
定义项目中使用的所有自定义异常类
"""

from typing import Optional, Any


class FPT25BaseException(Exception):
    """FPT25项目基础异常类"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details
    
    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class ConfigurationError(FPT25BaseException):
    """配置相关异常"""
    pass


class ValidationError(FPT25BaseException):
    """数据验证异常"""
    pass


class TensorError(FPT25BaseException):
    """张量操作异常"""
    pass


class LookupTableError(FPT25BaseException):
    """查找表相关异常"""
    pass


class ActivationFunctionError(FPT25BaseException):
    """激活函数相关异常"""
    pass


class HardwareError(FPT25BaseException):
    """硬件相关异常"""
    pass


class OptimizationError(FPT25BaseException):
    """优化相关异常"""
    pass


class EvaluationError(FPT25BaseException):
    """评估相关异常"""
    pass


class FileOperationError(FPT25BaseException):
    """文件操作异常"""
    pass


class MemoryError(FPT25BaseException):
    """内存相关异常"""
    pass


class NumericalError(FPT25BaseException):
    """数值计算异常"""
    pass


class AccuracyError(FPT25BaseException):
    """精度相关异常"""
    pass


class PerformanceError(FPT25BaseException):
    """性能相关异常"""
    pass


# 具体的异常类定义

class InvalidTensorShapeError(TensorError):
    """无效张量形状异常"""
    
    def __init__(self, actual_shape: tuple, expected_shape: tuple):
        message = f"张量形状不匹配: 实际 {actual_shape}, 期望 {expected_shape}"
        super().__init__(message, "INVALID_TENSOR_SHAPE", {
            'actual_shape': actual_shape,
            'expected_shape': expected_shape
        })


class UnsupportedDataTypeError(TensorError):
    """不支持的数据类型异常"""
    
    def __init__(self, dtype: str, supported_types: list):
        message = f"不支持的数据类型: {dtype}, 支持的类型: {supported_types}"
        super().__init__(message, "UNSUPPORTED_DATA_TYPE", {
            'dtype': dtype,
            'supported_types': supported_types
        })


class InvalidPointCountError(LookupTableError):
    """无效查找表点数异常"""
    
    def __init__(self, point_count: int, min_count: int, max_count: int):
        message = f"查找表点数无效: {point_count}, 应在 {min_count}-{max_count} 之间"
        super().__init__(message, "INVALID_POINT_COUNT", {
            'point_count': point_count,
            'min_count': min_count,
            'max_count': max_count
        })


class InvalidInterpolationMethodError(LookupTableError):
    """无效插值方法异常"""
    
    def __init__(self, method: str, supported_methods: list):
        message = f"无效插值方法: {method}, 支持的方法: {supported_methods}"
        super().__init__(message, "INVALID_INTERPOLATION_METHOD", {
            'method': method,
            'supported_methods': supported_methods
        })


class NumericalOverflowError(NumericalError):
    """数值溢出异常"""
    
    def __init__(self, value: float, max_value: float):
        message = f"数值溢出: {value} > {max_value}"
        super().__init__(message, "NUMERICAL_OVERFLOW", {
            'value': value,
            'max_value': max_value
        })


class NumericalUnderflowError(NumericalError):
    """数值下溢异常"""
    
    def __init__(self, value: float, min_value: float):
        message = f"数值下溢: {value} < {min_value}"
        super().__init__(message, "NUMERICAL_UNDERFLOW", {
            'value': value,
            'min_value': min_value
        })


class AccuracyThresholdExceededError(AccuracyError):
    """精度阈值超出异常"""
    
    def __init__(self, actual_error: float, threshold: float):
        message = f"精度误差超出阈值: {actual_error} > {threshold}"
        super().__init__(message, "ACCURACY_THRESHOLD_EXCEEDED", {
            'actual_error': actual_error,
            'threshold': threshold
        })


class MemoryAllocationError(MemoryError):
    """内存分配失败异常"""
    
    def __init__(self, requested_size: int, available_size: int):
        message = f"内存分配失败: 请求 {requested_size} bytes, 可用 {available_size} bytes"
        super().__init__(message, "MEMORY_ALLOCATION_FAILED", {
            'requested_size': requested_size,
            'available_size': available_size
        })


class FileNotFoundError(FileOperationError):
    """文件不存在异常"""
    
    def __init__(self, file_path: str):
        message = f"文件不存在: {file_path}"
        super().__init__(message, "FILE_NOT_FOUND", {'file_path': file_path})


class ConfigParseError(ConfigurationError):
    """配置文件解析异常"""
    
    def __init__(self, config_file: str, parse_error: str):
        message = f"配置文件解析失败: {config_file}, 错误: {parse_error}"
        super().__init__(message, "CONFIG_PARSE_ERROR", {
            'config_file': config_file,
            'parse_error': parse_error
        })


class ActivationFunctionNotImplementedError(ActivationFunctionError):
    """激活函数未实现异常"""
    
    def __init__(self, function_name: str):
        message = f"激活函数未实现: {function_name}"
        super().__init__(message, "ACTIVATION_FUNCTION_NOT_IMPLEMENTED", {
            'function_name': function_name
        })


class HardwareNotSupportedError(HardwareError):
    """硬件不支持异常"""
    
    def __init__(self, feature: str):
        message = f"硬件不支持该功能: {feature}"
        super().__init__(message, "HARDWARE_NOT_SUPPORTED", {'feature': feature})


class OptimizationTimeoutError(OptimizationError):
    """优化超时异常"""
    
    def __init__(self, timeout_seconds: int):
        message = f"优化超时: {timeout_seconds} 秒"
        super().__init__(message, "OPTIMIZATION_TIMEOUT", {
            'timeout_seconds': timeout_seconds
        })


class BenchmarkError(PerformanceError):
    """基准测试异常"""
    
    def __init__(self, test_name: str, error_message: str):
        message = f"基准测试失败: {test_name}, 错误: {error_message}"
        super().__init__(message, "BENCHMARK_ERROR", {
            'test_name': test_name,
            'error_message': error_message
        })


# 异常处理工具函数

def handle_exception(exception: Exception, context: str = "") -> str:
    """
    统一异常处理函数
    
    Args:
        exception: 异常对象
        context: 异常上下文信息
    
    Returns:
        格式化的错误消息
    """
    if isinstance(exception, FPT25BaseException):
        error_msg = str(exception)
        if context:
            error_msg = f"[{context}] {error_msg}"
        return error_msg
    else:
        error_msg = f"未处理的异常: {type(exception).__name__}: {str(exception)}"
        if context:
            error_msg = f"[{context}] {error_msg}"
        return error_msg


def validate_tensor_shape(shape: tuple, expected_shape: tuple) -> None:
    """
    验证张量形状
    
    Args:
        shape: 实际形状
        expected_shape: 期望形状
    
    Raises:
        InvalidTensorShapeError: 形状不匹配时抛出
    """
    if shape != expected_shape:
        raise InvalidTensorShapeError(shape, expected_shape)


def validate_dtype(dtype: str, supported_types: list) -> None:
    """
    验证数据类型
    
    Args:
        dtype: 数据类型
        supported_types: 支持的类型列表
    
    Raises:
        UnsupportedDataTypeError: 类型不支持时抛出
    """
    if dtype not in supported_types:
        raise UnsupportedDataTypeError(dtype, supported_types)


def validate_point_count(point_count: int, min_count: int, max_count: int) -> None:
    """
    验证查找表点数
    
    Args:
        point_count: 点数
        min_count: 最小点数
        max_count: 最大点数
    
    Raises:
        InvalidPointCountError: 点数无效时抛出
    """
    if not (min_count <= point_count <= max_count):
        raise InvalidPointCountError(point_count, min_count, max_count)
