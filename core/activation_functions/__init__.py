"""
激活函数模块
提供所有激活函数的实现和管理
"""

from .activation_manager import (
    BaseActivationFunction,
    ActivationConfig,
    ActivationFunctionManager,
    get_activation_manager,
    create_activation_function,
    benchmark_activation_functions
)

from .base_activation import (
    BaseActivationFunction,
    ActivationScanner
)

from .softmax_activation import (
    SoftmaxConfig,
    SoftmaxActivation,
    create_softmax,
    softmax_forward,
    softmax_benchmark
)

from .layer_norm_activation import (
    LayerNormConfig,
    LayerNormActivation
)

from .rms_norm_activation import (
    RMSNormConfig,
    RMSNormActivation
)

from .silu_activation import (
    SiLUConfig,
    SiLUActivation
)

from .gelu_activation import (
    GELUConfig,
    GELUActivation
)

from .add_activation import (
    AddConfig,
    AddActivation
)

from .multiply_activation import (
    MultiplyConfig,
    MultiplyActivation
)

__all__ = [
    # 基础类
    'BaseActivationFunction', 'ActivationConfig', 'ActivationFunctionManager',
    'get_activation_manager', 'create_activation_function', 'benchmark_activation_functions',
    
    'BaseActivationFunction', 'ActivationScanner',
    
    # Softmax
    'SoftmaxConfig', 'SoftmaxActivation',
    'create_softmax', 'softmax_forward', 'softmax_benchmark',
    
    # LayerNorm
    'LayerNormConfig', 'LayerNormActivation',
    
    # RMSNorm
    'RMSNormConfig', 'RMSNormActivation',
    
    # SiLU
    'SiLUConfig', 'SiLUActivation',
    
    # GELU
    'GELUConfig', 'GELUActivation',
    
    # Add
    'AddConfig', 'AddActivation',
    
    # Multiply
    'MultiplyConfig', 'MultiplyActivation'
]
