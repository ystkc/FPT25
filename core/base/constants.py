"""
常量定义模块
统一管理项目中的所有常量，包括竞赛要求的参数和配置项
"""

import math
import torch
from typing import Tuple, List, Dict, Any

# =============================================================================
# 竞赛核心参数
# =============================================================================

# 张量形状参数
TENSOR_SHAPE: Tuple[int, int] = (64, 768)  # (N, D) 竞赛要求的张量形状
BATCH_SIZE: int = 16  # 批处理大小

# 数据类型
SUPPORTED_DTYPES: List[str] = ['float32', 'bfloat16']
DEFAULT_DTYPE: str = 'bfloat16'
DEFAULT_UNSIGNED_TYPE: str = 'uint16'
DEFAULT_DTYPE_LEN: int = 16

# 数据映射(将str映射到dtype)
DATA_TYPE_MAP: Dict[str, torch.dtype] = {
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
    'uint32': torch.uint32,
    'uint16': torch.uint16,
}

# 精度评估参数
EPSILON_STAR: float = 1e-3  # 无损精度阈值
EPSILON_MAX: float = 1e-1   # 最大允许误差
EPSILON_TINY: float = 1e-12  # 防止除零的小常数

# =============================================================================
# 查找表参数
# =============================================================================

# 查找表阈值
TABLE_THRESH: float = 0.5 * math.log(2)  # 查找表阈值

# 查找表点数范围 - 增加点数以提高精度
BIT_LEN_RANGE: Tuple[int, int, int] = (10, 32, 1)  # (min, max, step)
DEFAULT_BIT_LEN: int = 12

# 插值方法
INTERPOLATION_METHODS: List[str] = ['direct', 'linear', 'quadratic']
DEFAULT_INTERPOLATION: str = 'quadratic'  # 默认使用二次插值以提高精度

# 采样策略
SAMPLING_STRATEGIES: List[str] = ['uniform', 'adaptive', 'logarithmic', 'quadratic']
DEFAULT_SAMPLING_STRATEGY: str = 'adaptive'  # 默认使用自适应采样以提高精度

# =============================================================================
# 激活函数权重（竞赛评分权重）
# =============================================================================

ACTIVATION_FUNCTION_WEIGHTS: Dict[str, int] = {
    'softmax': 15,      # 核心注意力机制，权重最高
    'layer_norm': 12,   # 残差连接后使用
    'rms_norm': 10,     # 趋势性替代方案
    'silu': 10,         # 主流激活函数
    'gelu': 10,         # 广泛使用
    'add': 7,           # 高频操作（残差连接）
    'multiply': 6       # 缩放、门控等
}

# 激活函数列表
ACTIVATION_FUNCTIONS: List[str] = list(ACTIVATION_FUNCTION_WEIGHTS.keys())

# =============================================================================
# 硬件相关参数
# =============================================================================

# 定点数格式
FIXED_POINT_FORMATS: List[str] = ['Q8_8', 'Q16_16', 'Q32_32', 'Q8_24']
DEFAULT_FIXED_POINT_FORMAT: str = 'Q16_16'

# 定点数精度
FIXED_POINT_PRECISION: Dict[str, Tuple[int, int]] = {
    'Q8_8': (8, 8),
    'Q16_16': (16, 16),
    'Q32_32': (32, 32),
    'Q8_24': (8, 24)
}

# =============================================================================
# 性能优化参数
# =============================================================================

# 内存优化
MEMORY_OPTIMIZATION: Dict[str, Any] = {
    'enable_tensor_views': True,
    'enable_memory_pooling': True,
    'max_memory_usage': 1024 * 1024 * 1024,  # 1GB
    'cache_size': 1000
}

# 并行处理
PARALLEL_PROCESSING: Dict[str, Any] = {
    'max_workers': 4,
    'chunk_size': 1000,
    'enable_vectorization': True
}

# =============================================================================
# 文件路径常量
# =============================================================================

# 输出目录结构
OUTPUT_DIRS: Dict[str, str] = {
    'results': 'results',
    'charts': 'charts',
    'reports': 'reports',
    'data': 'data',
    'logs': 'logs'
}

# 文件扩展名
FILE_EXTENSIONS: Dict[str, str] = {
    'tensor': '.pt',
    'json': '.json',
    'excel': '.xlsx',
    'log': '.log',
    'config': '.json'
}

# =============================================================================
# 数学常量
# =============================================================================

# 数学常数
MATH_CONSTANTS: Dict[str, float] = {
    'PI': math.pi,
    'E': math.e,
    'LN2': math.log(2),
    'SQRT2': math.sqrt(2),
    'SQRT2PI': math.sqrt(2 * math.pi)
}

# 数值稳定性参数
NUMERICAL_STABILITY: Dict[str, float] = {
    'min_value': 1e-8,
    'max_value': 1e8,
    'log_min': -20.0,
    'log_max': 20.0
}

# =============================================================================
# 测试参数
# =============================================================================

# 测试配置
TEST_CONFIG: Dict[str, Any] = {
    'test_tensor_shapes': [(32, 128), (64, 256), (64, 768)],
    'test_batch_sizes': [1, 4, 8, 16],
    'test_dtypes': ['float32', 'bfloat16'],
    'test_bit_lens': [800, 900, 1000],
    'test_interpolations': ['direct', 'linear', 'quadratic']
}

# 基准测试参数
BENCHMARK_CONFIG: Dict[str, Any] = {
    'warmup_runs': 5,
    'measurement_runs': 20,
    'timeout_seconds': 300,
    'memory_limit_mb': 2048
}

# =============================================================================
# 日志配置
# =============================================================================

# 日志级别
LOG_LEVELS: List[str] = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
DEFAULT_LOG_LEVEL: str = 'INFO'

# 日志格式
LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT: str = '%Y-%m-%d %H:%M:%S'

# =============================================================================
# 错误消息
# =============================================================================

ERROR_MESSAGES: Dict[str, str] = {
    'invalid_tensor_shape': '张量形状必须为 (64, 768)',
    'invalid_dtype': '不支持的数据类型，支持的类型: {supported_types}',
    'invalid_bit_len': '查找表点数必须在 {min_count} 到 {max_count} 之间',
    'invalid_interpolation': '不支持的插值方法: {method}',
    'tensor_dimension_mismatch': '张量维度不匹配',
    'numerical_overflow': '数值溢出',
    'numerical_underflow': '数值下溢',
    'invalid_input_range': '输入值超出有效范围',
    'memory_allocation_failed': '内存分配失败',
    'file_not_found': '文件不存在: {file_path}',
    'config_parse_error': '配置文件解析错误: {error}'
}

# =============================================================================
# 成功消息
# =============================================================================

SUCCESS_MESSAGES: Dict[str, str] = {
    'activation_computed': '激活函数计算完成',
    'lookup_table_generated': '查找表生成完成',
    'accuracy_test_passed': '精度测试通过',
    'benchmark_completed': '基准测试完成',
    'optimization_finished': '优化完成',
    'file_saved': '文件保存成功: {file_path}',
    'config_loaded': '配置加载成功'
}

# =============================================================================
# 版本信息
# =============================================================================

VERSION_INFO: Dict[str, str] = {
    'version': '1.0.0',
    'author': 'FPT25 Team',
    'description': 'FPGA 激活函数硬件加速项目',
    'contest': 'FPT25 Design Competition',
    'track': 'Large-Scale AI Model Activation Function FPGA Acceleration'
}
