"""
配置管理模块
提供项目配置的默认值和配置管理功能
"""

import json
import os
import traceback
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict, field
from pathlib import Path

import torch

from core.base.constants import (
    DATA_TYPE_MAP, DEFAULT_DTYPE_LEN, DEFAULT_UNSIGNED_TYPE, TENSOR_SHAPE, BATCH_SIZE, DEFAULT_DTYPE, DEFAULT_COMPUTE_DTYPE, DEFAULT_BIT_LEN,
    DEFAULT_INTERPOLATION, DEFAULT_SAMPLING_STRATEGY, ACTIVATION_FUNCTIONS,
    FIXED_POINT_FORMATS, DEFAULT_FIXED_POINT_FORMAT, OUTPUT_DIRS
)
from core.base.exceptions import ConfigParseError, FileNotFoundError
from core.base.logs import get_logger


@dataclass
class ActivationConfig:
    """激活函数配置基类"""
    dtype_str: str = DEFAULT_DTYPE
    compute_dtype_str: str = DEFAULT_COMPUTE_DTYPE
    use_lookup_table: bool = True
    lookup_table_bitlen: int = 16  # 增加查找表大小

    interpolation_method: str = 'quadratic'  # 使用二次插值
    
    use_fixed_point: bool = False
    fixed_point_format: str = 'Q16_16'

    def __post_init__(self):
        """初始化后处理，将json配置的字符串转换为dtype"""
        self.dtype = DATA_TYPE_MAP[self.dtype_str]
        self.compute_dtype = DATA_TYPE_MAP[self.compute_dtype_str]

@dataclass
class LookupTableConfig:
    """查找表配置"""
    bit_len: int = DEFAULT_BIT_LEN # 查找表位长度，默认12位
    sample_count: int = 1024  # （仅随机采样有效）采样点数，默认1024个
    random_seed: int = 0 # （仅随机采样有效）
    dtype_str: str = DEFAULT_DTYPE  # 查找表数据类型，默认bfloat16
    unsigned_type_str: str = DEFAULT_UNSIGNED_TYPE  # 无符号数据类型，和dtype位宽相同
    dtype_len: int = DEFAULT_DTYPE_LEN  # 数据类型位长，默认16位
    interpolation_method: str = DEFAULT_INTERPOLATION  # 默认使用二次插值
    sampling_strategy: str = DEFAULT_SAMPLING_STRATEGY  # 默认使用自适应采样
    use_advanced_lookup: bool = False
    x_struct_range_int: list = field(default_factory=lambda: [(0x0000, 0x7f7f), (0x8000, 0xff7f)]) # 使用int初始化，在后初始化中转换为dtype类型
    table_name: str = ''
    use_cache: bool = False # 计算完成后写入文件，并在遇到相同配置时直接使用文件
    use_parallel_lookup: bool = True # 并行查找表查询操作（暂时不确定能否加速还是拖慢计算速度）

    def __post_init__(self):
        '''将x_struct_range转换为dtype类型'''
        self.dtype = DATA_TYPE_MAP[self.dtype_str]
        self.unsigned_type = DATA_TYPE_MAP[self.unsigned_type_str]
        self.x_struct_range = torch.tensor(self.x_struct_range_int, dtype=self.unsigned_type)
        self.unsigned_mask = (1 << self.dtype_len) - 1  # 无符号掩码
        self.zero_len = self.dtype_len - self.bit_len  # 零位长度


@dataclass
class SoftmaxConfig(ActivationConfig):
    """Softmax 配置"""

@dataclass
class LayerNormConfig(ActivationConfig):
    """LayerNorm 配置"""
    eps: float = 1e-5
    use_learnable_params: bool = True
    gamma_init: float = 1.0
    beta_init: float = 0.0
@dataclass
class RMSNormConfig(ActivationConfig):
    """RMSNorm 配置"""
    eps: float = 1e-5
    use_learnable_params: bool = True
    gamma_init: float = 1.0

@dataclass
class SiLUConfig(ActivationConfig):
    """SiLU 配置"""
    use_lookup_table: bool = True
    lookup_bit_len: int = 800
    interpolation_method: str = 'linear'

@dataclass
class GELUConfig(ActivationConfig):
    """GELU 配置"""
    use_approximation: bool = True
    approximation_type: str = 'none'  # 'tanh' or 'erf' or 'none'

@dataclass
class AddConfig(ActivationConfig):
    """Add 配置"""
    pass

@dataclass
class MultiplyConfig(ActivationConfig):
    """Multiply 配置"""
    pass




@dataclass
class HardwareConfig:
    """硬件配置"""
    use_fixed_point: bool = True
    fixed_point_format: str = DEFAULT_FIXED_POINT_FORMAT
    enable_parallel_processing: bool = True
    max_workers: int = 4


@dataclass
class OptimizationConfig:
    """优化配置"""
    enable_memory_pooling: bool = True
    enable_vectorization: bool = True
    chunk_size: int = 1000
    memory_limit_mb: int = 1024


@dataclass
class TestConfig:
    """测试配置"""
    tensor_shape: tuple = TENSOR_SHAPE
    batch_size: int = BATCH_SIZE
    dtype: str = DEFAULT_DTYPE
    num_runs: int = 100
    warmup_runs: int = 10


@dataclass
class OutputConfig:
    """输出配置"""
    output_dir: str = "results"
    save_tensors: bool = True
    save_reports: bool = True
    save_charts: bool = True
    generate_excel: bool = True


DEFAULT_SOFTMAX_CONFIG = SoftmaxConfig()
DEFAULT_LAYER_NORM_CONFIG = LayerNormConfig()
DEFAULT_RMS_NORM_CONFIG = RMSNormConfig()
DEFAULT_SILU_CONFIG = SiLUConfig()
DEFAULT_GELU_CONFIG = GELUConfig()
DEFAULT_ADD_CONFIG = AddConfig()
DEFAULT_MULTIPLY_CONFIG = MultiplyConfig()

@dataclass
class ActivationFunctions:
    """激活函数配置"""
    softmax: SoftmaxConfig = None
    layer_norm: LayerNormConfig = None
    rms_norm: RMSNormConfig = None
    silu: SiLUConfig = None
    gelu: GELUConfig = None
    add: AddConfig = None
    multiply: MultiplyConfig = None


@dataclass
class ProjectConfig:
    """项目主配置"""
    # 基本配置
    project_name: str = "FPT25 Activation Functions"
    version: str = "1.0.0"
    description: str = "FPGA 激活函数硬件加速项目"
    
    # 子配置
    lookup_table: LookupTableConfig = None
    hardware: HardwareConfig = None
    optimization: OptimizationConfig = None
    test: TestConfig = None
    output: OutputConfig = None
    
    # 激活函数配置
    activation_functions: ActivationConfig = None
    
    def __post_init__(self):
        """初始化后处理"""
        if self.lookup_table is None:
            self.lookup_table = LookupTableConfig()
        if self.hardware is None:
            self.hardware = HardwareConfig()
        if self.optimization is None:
            self.optimization = OptimizationConfig()
        if self.test is None:
            self.test = TestConfig()
        if self.output is None:
            self.output = OutputConfig()
        if self.activation_functions is None:
            self.activation_functions = ActivationFunctions()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProjectConfig':
        """从字典创建配置"""
        # 处理嵌套配置
        if 'lookup_table' in data and isinstance(data['lookup_table'], dict):
            data['lookup_table'] = LookupTableConfig(**data['lookup_table'])
        
        if 'hardware' in data and isinstance(data['hardware'], dict):
            data['hardware'] = HardwareConfig(**data['hardware'])
        
        if 'optimization' in data and isinstance(data['optimization'], dict):
            data['optimization'] = OptimizationConfig(**data['optimization'])
        
        if 'test' in data and isinstance(data['test'], dict):
            data['test'] = TestConfig(**data['test'])
        
        if 'output' in data and isinstance(data['output'], dict):
            data['output'] = OutputConfig(**data['output'])

        if 'activation_functions' in data and isinstance(data['activation_functions'], dict):
             afs = ActivationFunctions(**data['activation_functions'])
             if afs.softmax is None:
                 afs.softmax = DEFAULT_SOFTMAX_CONFIG
             if afs.layer_norm is None:
                 afs.layer_norm = DEFAULT_LAYER_NORM_CONFIG
             if afs.rms_norm is None:
                 afs.rms_norm = DEFAULT_RMS_NORM_CONFIG
             if afs.silu is None:
                 afs.silu = DEFAULT_SILU_CONFIG
             if afs.gelu is None:
                 afs.gelu = DEFAULT_GELU_CONFIG
             if afs.add is None:
                 afs.add = DEFAULT_ADD_CONFIG
             if afs.multiply is None:
                 afs.multiply = DEFAULT_MULTIPLY_CONFIG
             data['activation_functions'] = afs
        
        return cls(**data)


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.logger = get_logger()
        self.config_file = config_file or "config.json"
        self.config: Optional[ProjectConfig] = None
        self._load_config()
    
    def _load_config(self) -> None:
        """加载配置"""
        try:
            if os.path.exists(self.config_file):
                self.config = self._load_from_file(self.config_file)
                self.logger.info(f"从文件加载配置: {self.config_file}")
            else:
                self.config = ProjectConfig()
                self._save_to_file(self.config_file, self.config)
                self.logger.info(f"创建默认配置文件: {self.config_file}")
        except Exception as e:
            self.logger.error(f"配置加载失败: {e}")
            self.config = ProjectConfig()
    
    def _load_from_file(self, filepath: str) -> ProjectConfig:
        """从文件加载配置"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return ProjectConfig.from_dict(data)
        except json.JSONDecodeError as e:
            raise ConfigParseError(filepath, f"JSON 解析错误: {e}")
        except Exception as e:
            raise ConfigParseError(filepath, f"文件读取错误: {e}")
    
    def _save_to_file(self, filepath: str, config: ProjectConfig) -> None:
        """保存配置到文件"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise ConfigParseError(filepath, f"文件保存错误: {e}")
    
    def get_config(self) -> ProjectConfig:
        """获取当前配置"""
        if self.config is None:
            self.config = ProjectConfig()
        return self.config
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """更新配置"""
        if self.config is None:
            self.config = ProjectConfig()
        
        # 递归更新配置
        self._update_nested_dict(self.config.to_dict(), updates)
        self.config = ProjectConfig.from_dict(self.config.to_dict())
        
        self.logger.info("配置已更新")
    
    def _update_nested_dict(self, base_dict: Dict[str, Any], 
                           updates: Dict[str, Any]) -> None:
        """递归更新嵌套字典"""
        for key, value in updates.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._update_nested_dict(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def save_config(self, filepath: Optional[str] = None) -> None:
        """保存配置"""
        save_path = filepath or self.config_file
        self._save_to_file(save_path, self.get_config())
        self.logger.info(f"配置已保存到: {save_path}")
    
    def reset_to_default(self) -> None:
        """重置为默认配置"""
        self.config = ProjectConfig()
        self.logger.info("配置已重置为默认值")
    
    def validate_config(self) -> List[str]:
        """验证配置"""
        errors = []
        config = self.get_config()
        
        # 验证查找表配置
        if config.lookup_table.bit_len < 100 or config.lookup_table.bit_len > 2000:
            errors.append("查找表点数应在 100-2000 之间")
        
        if config.lookup_table.interpolation_method not in ['direct', 'linear', 'quadratic']:
            errors.append("插值方法必须是 direct、linear 或 quadratic")
        
        # 验证硬件配置
        if config.hardware.fixed_point_format not in FIXED_POINT_FORMATS:
            errors.append(f"定点数格式必须是 {FIXED_POINT_FORMATS} 之一")
        
        # 验证测试配置
        if config.test.batch_size < 1:
            errors.append("批处理大小必须大于 0")
        
        if config.test.dtype not in ['float32', 'bfloat16']:
            errors.append("数据类型必须是 float32 或 bfloat16")
        
        return errors
    
    def get_activation_config(self, function_name: str) -> Dict[str, Any]:
        """获取激活函数配置"""
        config = self.get_config()
        return config.activation_functions.get(function_name, {})
    
    def update_activation_config(self, function_name: str, 
                               updates: Dict[str, Any]) -> None:
        """更新激活函数配置"""
        config = self.get_config()
        if function_name not in config.activation_functions:
            config.activation_functions[function_name] = {}
        
        config.activation_functions[function_name].update(updates)
        self.logger.info(f"更新激活函数配置: {function_name}")
    
    def export_config(self, filepath: str) -> None:
        """导出配置"""
        self._save_to_file(filepath, self.get_config())
        self.logger.info(f"配置已导出到: {filepath}")
    
    def import_config(self, filepath: str) -> None:
        """导入配置"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(filepath)
        
        self.config = self._load_from_file(filepath)
        self.logger.info(f"配置已从 {filepath} 导入")


# 全局配置管理器
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_file: Optional[str] = None) -> ConfigManager:
    """获取全局配置管理器"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_file)
    return _config_manager


def get_config() -> ProjectConfig:
    """获取当前配置便捷函数"""
    manager = get_config_manager()
    return manager.get_config()


def load_config(filepath: str) -> ProjectConfig:
    """加载配置文件便捷函数"""
    manager = ConfigManager(filepath)
    return manager.get_config()


def save_config(config: ProjectConfig, filepath: str) -> None:
    """保存配置便捷函数"""
    manager = ConfigManager()
    manager.config = config
    manager.save_config(filepath)
