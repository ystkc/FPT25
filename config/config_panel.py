"""
配置面板模块
提供交互式配置管理界面
"""

import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass

from .config import ProjectConfig, ConfigManager, get_config_manager
from core.base.logs import get_logger


@dataclass
class ConfigOption:
    """配置选项"""
    name: str
    description: str
    value_type: type
    default_value: Any
    choices: Optional[List[Any]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    validation_func: Optional[Callable[[Any], bool]] = None


class ConfigPanel:
    """配置面板"""
    
    def __init__(self):
        self.logger = get_logger()
        self.config_manager = get_config_manager()
        self.config_options = self._initialize_config_options()
    
    def _initialize_config_options(self) -> Dict[str, List[ConfigOption]]:
        """初始化配置选项"""
        return {
            'lookup_table': [
                ConfigOption(
                    name='point_count',
                    description='查找表点数',
                    value_type=int,
                    default_value=800,
                    min_value=100,
                    max_value=2000
                ),
                ConfigOption(
                    name='interpolation_method',
                    description='插值方法',
                    value_type=str,
                    default_value='linear',
                    choices=['direct', 'linear', 'quadratic']
                ),
                ConfigOption(
                    name='sampling_strategy',
                    description='采样策略',
                    value_type=str,
                    default_value='uniform',
                    choices=['uniform', 'adaptive', 'logarithmic', 'quadratic']
                ),
                ConfigOption(
                    name='use_advanced_lookup',
                    description='使用高级查找表',
                    value_type=bool,
                    default_value=False
                )
            ],
            'hardware': [
                ConfigOption(
                    name='use_fixed_point',
                    description='使用定点数',
                    value_type=bool,
                    default_value=True
                ),
                ConfigOption(
                    name='fixed_point_format',
                    description='定点数格式',
                    value_type=str,
                    default_value='Q16_16',
                    choices=['Q8_8', 'Q16_16', 'Q32_32', 'Q8_24']
                ),
                ConfigOption(
                    name='enable_parallel_processing',
                    description='启用并行处理',
                    value_type=bool,
                    default_value=True
                ),
                ConfigOption(
                    name='max_workers',
                    description='最大工作线程数',
                    value_type=int,
                    default_value=4,
                    min_value=1,
                    max_value=16
                )
            ],
            'optimization': [
                ConfigOption(
                    name='enable_memory_pooling',
                    description='启用内存池',
                    value_type=bool,
                    default_value=True
                ),
                ConfigOption(
                    name='enable_vectorization',
                    description='启用向量化',
                    value_type=bool,
                    default_value=True
                ),
                ConfigOption(
                    name='chunk_size',
                    description='批处理大小',
                    value_type=int,
                    default_value=1000,
                    min_value=100,
                    max_value=10000
                ),
                ConfigOption(
                    name='memory_limit_mb',
                    description='内存限制 (MB)',
                    value_type=int,
                    default_value=1024,
                    min_value=256,
                    max_value=8192
                )
            ],
            'test': [
                ConfigOption(
                    name='tensor_shape',
                    description='张量形状',
                    value_type=tuple,
                    default_value=(64, 768)
                ),
                ConfigOption(
                    name='batch_size',
                    description='批处理大小',
                    value_type=int,
                    default_value=16,
                    min_value=1,
                    max_value=64
                ),
                ConfigOption(
                    name='dtype',
                    description='数据类型',
                    value_type=str,
                    default_value='bfloat16',
                    choices=['float32', 'bfloat16']
                ),
                ConfigOption(
                    name='num_runs',
                    description='测试运行次数',
                    value_type=int,
                    default_value=100,
                    min_value=10,
                    max_value=1000
                )
            ],
            'output': [
                ConfigOption(
                    name='output_dir',
                    description='输出目录',
                    value_type=str,
                    default_value='results'
                ),
                ConfigOption(
                    name='save_tensors',
                    description='保存张量',
                    value_type=bool,
                    default_value=True
                ),
                ConfigOption(
                    name='save_reports',
                    description='保存报告',
                    value_type=bool,
                    default_value=True
                ),
                ConfigOption(
                    name='save_charts',
                    description='保存图表',
                    value_type=bool,
                    default_value=True
                ),
                ConfigOption(
                    name='generate_excel',
                    description='生成 Excel 报告',
                    value_type=bool,
                    default_value=True
                )
            ]
        }
    
    def get_config_options(self, section: str) -> List[ConfigOption]:
        """获取配置选项"""
        return self.config_options.get(section, [])
    
    def get_all_sections(self) -> List[str]:
        """获取所有配置节"""
        return list(self.config_options.keys())
    
    def validate_option_value(self, option: ConfigOption, value: Any) -> bool:
        """验证选项值"""
        # 类型检查
        if not isinstance(value, option.value_type):
            return False
        
        # 选择检查
        if option.choices and value not in option.choices:
            return False
        
        # 范围检查
        if option.min_value is not None and value < option.min_value:
            return False
        
        if option.max_value is not None and value > option.max_value:
            return False
        
        # 自定义验证
        if option.validation_func and not option.validation_func(value):
            return False
        
        return True
    
    def set_option_value(self, section: str, option_name: str, value: Any) -> bool:
        """设置选项值"""
        options = self.get_config_options(section)
        option = next((opt for opt in options if opt.name == option_name), None)
        
        if option is None:
            self.logger.error(f"未找到选项: {section}.{option_name}")
            return False
        
        if not self.validate_option_value(option, value):
            self.logger.error(f"选项值无效: {section}.{option_name} = {value}")
            return False
        
        # 更新配置
        config = self.config_manager.get_config()
        if section == 'lookup_table':
            setattr(config.lookup_table, option_name, value)
        elif section == 'hardware':
            setattr(config.hardware, option_name, value)
        elif section == 'optimization':
            setattr(config.optimization, option_name, value)
        elif section == 'test':
            setattr(config.test, option_name, value)
        elif section == 'output':
            setattr(config.output, option_name, value)
        else:
            self.logger.error(f"未知的配置节: {section}")
            return False
        
        self.logger.info(f"设置配置: {section}.{option_name} = {value}")
        return True
    
    def get_option_value(self, section: str, option_name: str) -> Any:
        """获取选项值"""
        config = self.config_manager.get_config()
        
        if section == 'lookup_table':
            return getattr(config.lookup_table, option_name)
        elif section == 'hardware':
            return getattr(config.hardware, option_name)
        elif section == 'optimization':
            return getattr(config.optimization, option_name)
        elif section == 'test':
            return getattr(config.test, option_name)
        elif section == 'output':
            return getattr(config.output, option_name)
        else:
            self.logger.error(f"未知的配置节: {section}")
            return None
    
    def reset_section_to_default(self, section: str) -> bool:
        """重置节为默认值"""
        if section not in self.config_options:
            self.logger.error(f"未知的配置节: {section}")
            return False
        
        config = self.config_manager.get_config()
        
        if section == 'lookup_table':
            config.lookup_table = type(config.lookup_table)()
        elif section == 'hardware':
            config.hardware = type(config.hardware)()
        elif section == 'optimization':
            config.optimization = type(config.optimization)()
        elif section == 'test':
            config.test = type(config.test)()
        elif section == 'output':
            config.output = type(config.output)()
        
        self.logger.info(f"重置配置节为默认值: {section}")
        return True
    
    def export_section_config(self, section: str, filepath: str) -> bool:
        """导出节配置"""
        if section not in self.config_options:
            self.logger.error(f"未知的配置节: {section}")
            return False
        
        try:
            config = self.config_manager.get_config()
            section_config = {}
            
            if section == 'lookup_table':
                section_config = config.lookup_table.to_dict() if hasattr(config.lookup_table, 'to_dict') else config.lookup_table.__dict__
            elif section == 'hardware':
                section_config = config.hardware.to_dict() if hasattr(config.hardware, 'to_dict') else config.hardware.__dict__
            elif section == 'optimization':
                section_config = config.optimization.to_dict() if hasattr(config.optimization, 'to_dict') else config.optimization.__dict__
            elif section == 'test':
                section_config = config.test.to_dict() if hasattr(config.test, 'to_dict') else config.test.__dict__
            elif section == 'output':
                section_config = config.output.to_dict() if hasattr(config.output, 'to_dict') else config.output.__dict__
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(section_config, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"导出配置节: {section} -> {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"导出配置失败: {e}")
            return False
    
    def import_section_config(self, section: str, filepath: str) -> bool:
        """导入节配置"""
        if section not in self.config_options:
            self.logger.error(f"未知的配置节: {section}")
            return False
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                section_config = json.load(f)
            
            config = self.config_manager.get_config()
            
            if section == 'lookup_table':
                for key, value in section_config.items():
                    setattr(config.lookup_table, key, value)
            elif section == 'hardware':
                for key, value in section_config.items():
                    setattr(config.hardware, key, value)
            elif section == 'optimization':
                for key, value in section_config.items():
                    setattr(config.optimization, key, value)
            elif section == 'test':
                for key, value in section_config.items():
                    setattr(config.test, key, value)
            elif section == 'output':
                for key, value in section_config.items():
                    setattr(config.output, key, value)
            
            self.logger.info(f"导入配置节: {filepath} -> {section}")
            return True
            
        except Exception as e:
            self.logger.error(f"导入配置失败: {e}")
            return False
    
    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        config = self.config_manager.get_config()
        
        return {
            'project_name': config.project_name,
            'version': config.version,
            'description': config.description,
            'sections': {
                'lookup_table': config.lookup_table.__dict__,
                'hardware': config.hardware.__dict__,
                'optimization': config.optimization.__dict__,
                'test': config.test.__dict__,
                'output': config.output.__dict__
            },
            'activation_functions': list(config.activation_functions.keys())
        }


# 全局配置面板
_config_panel: Optional[ConfigPanel] = None


def get_config_panel() -> ConfigPanel:
    """获取全局配置面板"""
    global _config_panel
    if _config_panel is None:
        _config_panel = ConfigPanel()
    return _config_panel


def configure_option(section: str, option_name: str, value: Any) -> bool:
    """配置选项便捷函数"""
    panel = get_config_panel()
    return panel.set_option_value(section, option_name, value)


def get_option_value(section: str, option_name: str) -> Any:
    """获取选项值便捷函数"""
    panel = get_config_panel()
    return panel.get_option_value(section, option_name)
