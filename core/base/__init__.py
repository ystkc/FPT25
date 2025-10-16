"""
基础模块
包含常量定义、异常处理和日志系统
"""

from .constants import (
    TENSOR_SHAPE,
    BATCH_SIZE,
    SUPPORTED_DTYPES,
    DEFAULT_DTYPE,
    EPSILON_STAR,
    EPSILON_MAX,
    EPSILON_TINY,
    TABLE_THRESH,
    BIT_LEN_RANGE,
    DEFAULT_BIT_LEN,
    INTERPOLATION_METHODS,
    DEFAULT_INTERPOLATION,
    SAMPLING_STRATEGIES,
    DEFAULT_SAMPLING_STRATEGY,
    ACTIVATION_FUNCTION_WEIGHTS,
    ACTIVATION_FUNCTIONS,
    FIXED_POINT_FORMATS,
    DEFAULT_FIXED_POINT_FORMAT,
    FIXED_POINT_PRECISION,
    MEMORY_OPTIMIZATION,
    PARALLEL_PROCESSING,
    OUTPUT_DIRS,
    FILE_EXTENSIONS,
    MATH_CONSTANTS,
    NUMERICAL_STABILITY,
    TEST_CONFIG,
    BENCHMARK_CONFIG,
    LOG_LEVELS,
    DEFAULT_LOG_LEVEL,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    ERROR_MESSAGES,
    SUCCESS_MESSAGES,
    VERSION_INFO
)

from .exceptions import (
    FPT25BaseException,
    ConfigurationError,
    ValidationError,
    TensorError,
    LookupTableError,
    ActivationFunctionError,
    HardwareError,
    OptimizationError,
    EvaluationError,
    FileOperationError,
    MemoryError,
    NumericalError,
    AccuracyError,
    PerformanceError,
    InvalidTensorShapeError,
    UnsupportedDataTypeError,
    InvalidPointCountError,
    InvalidInterpolationMethodError,
    NumericalOverflowError,
    NumericalUnderflowError,
    AccuracyThresholdExceededError,
    MemoryAllocationError,
    FileNotFoundError,
    ConfigParseError,
    ActivationFunctionNotImplementedError,
    HardwareNotSupportedError,
    OptimizationTimeoutError,
    BenchmarkError,
    handle_exception,
    validate_tensor_shape,
    validate_dtype,
    validate_bitlen
)

from .logs import (
    FPT25Logger,
    PerformanceLogger,
    AccuracyLogger,
    get_logger,
    get_performance_logger,
    get_accuracy_logger,
    setup_logging,
    log_function_call,
    log_execution_time
)

__all__ = [
    # 常量
    'TENSOR_SHAPE', 'BATCH_SIZE', 'SUPPORTED_DTYPES', 'DEFAULT_DTYPE',
    'EPSILON_STAR', 'EPSILON_MAX', 'EPSILON_TINY', 'TABLE_THRESH',
    'BIT_LEN_RANGE', 'DEFAULT_BIT_LEN', 'INTERPOLATION_METHODS',
    'DEFAULT_INTERPOLATION', 'SAMPLING_STRATEGIES', 'DEFAULT_SAMPLING_STRATEGY',
    'ACTIVATION_FUNCTION_WEIGHTS', 'ACTIVATION_FUNCTIONS', 'FIXED_POINT_FORMATS',
    'DEFAULT_FIXED_POINT_FORMAT', 'FIXED_POINT_PRECISION', 'MEMORY_OPTIMIZATION',
    'PARALLEL_PROCESSING', 'OUTPUT_DIRS', 'FILE_EXTENSIONS', 'MATH_CONSTANTS',
    'NUMERICAL_STABILITY', 'TEST_CONFIG', 'BENCHMARK_CONFIG', 'LOG_LEVELS',
    'DEFAULT_LOG_LEVEL', 'LOG_FORMAT', 'LOG_DATE_FORMAT', 'ERROR_MESSAGES',
    'SUCCESS_MESSAGES', 'VERSION_INFO',
    
    # 异常
    'FPT25BaseException', 'ConfigurationError', 'ValidationError', 'TensorError',
    'LookupTableError', 'ActivationFunctionError', 'HardwareError',
    'OptimizationError', 'EvaluationError', 'FileOperationError', 'MemoryError',
    'NumericalError', 'AccuracyError', 'PerformanceError', 'InvalidTensorShapeError',
    'UnsupportedDataTypeError', 'InvalidPointCountError', 'InvalidInterpolationMethodError',
    'NumericalOverflowError', 'NumericalUnderflowError', 'AccuracyThresholdExceededError',
    'MemoryAllocationError', 'FileNotFoundError', 'ConfigParseError',
    'ActivationFunctionNotImplementedError', 'HardwareNotSupportedError',
    'OptimizationTimeoutError', 'BenchmarkError', 'handle_exception',
    'validate_tensor_shape', 'validate_dtype', 'validate_bitlen',
    
    # 日志
    'FPT25Logger', 'PerformanceLogger', 'AccuracyLogger', 'get_logger',
    'get_performance_logger', 'get_accuracy_logger', 'setup_logging',
    'log_function_call', 'log_execution_time'
]
