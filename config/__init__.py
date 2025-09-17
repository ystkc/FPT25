"""
配置管理模块
提供项目配置管理功能
"""

from .config import (
    LookupTableConfig,
    HardwareConfig,
    OptimizationConfig,
    TestConfig,
    OutputConfig,
    ProjectConfig,
    ConfigManager,
    get_config_manager,
    get_config,
    load_config,
    save_config
)

from .config_panel import (
    ConfigOption,
    ConfigPanel,
    get_config_panel,
    configure_option,
    get_option_value
)

__all__ = [
    # 配置类
    'LookupTableConfig', 'HardwareConfig', 'OptimizationConfig',
    'TestConfig', 'OutputConfig', 'ProjectConfig',
    
    # 配置管理器
    'ConfigManager', 'get_config_manager', 'get_config', 'load_config', 'save_config',
    
    # 配置面板
    'ConfigOption', 'ConfigPanel', 'get_config_panel', 'configure_option', 'get_option_value'
]
