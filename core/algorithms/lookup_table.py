"""
查找表实现模块
提供高效的查找表算法，支持均匀采样和非均匀采样
"""

import math
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass

from core.base.constants import (
    TABLE_THRESH, POINT_COUNT_RANGE, INTERPOLATION_METHODS,
    SAMPLING_STRATEGIES, NUMERICAL_STABILITY, EPSILON_TINY
)
from core.base.exceptions import (
    InvalidPointCountError, InvalidInterpolationMethodError,
    validate_point_count, validate_dtype
)
from core.algorithms.math_utils import MathUtils
from core.utils.smart_cache import get_lookup_table_cache


@dataclass
class LookupTableConfig:
    """查找表配置"""
    point_count: int = 2000  # 增加采样点数量
    interpolation_method: str = 'quadratic'  # 默认使用二次插值
    sampling_strategy: str = 'adaptive'  # 默认使用自适应采样
    x_min: float = -20.0  # 扩大范围以覆盖Softmax的实际输入范围
    x_max: float = 0.0    # 调整上界，因为Softmax中减去最大值后通常为负值
    function_type: str = 'exp'
    precision: int = 16


class SamplingStrategy(ABC):
    """采样策略抽象基类"""
    
    @abstractmethod
    def generate_points(self, n_points: int, x_min: float, x_max: float) -> np.ndarray:
        """生成采样点"""
        pass


class UniformSampling(SamplingStrategy):
    """均匀采样策略"""
    
    def generate_points(self, n_points: int, x_min: float, x_max: float) -> np.ndarray:
        """生成均匀分布的采样点"""
        return np.linspace(x_min, x_max, n_points)


class AdaptiveSampling(SamplingStrategy):
    """自适应采样策略（在函数变化剧烈区域密集采样）"""
    
    def __init__(self, function: Callable[[float], float], 
                 sensitivity: float = 0.1):
        self.function = function
        self.sensitivity = sensitivity
    
    def generate_points(self, n_points: int, x_min: float, x_max: float) -> np.ndarray:
        """生成自适应采样点 - 针对指数函数优化"""
        # 首先生成密集的候选点
        candidate_points = np.linspace(x_min, x_max, n_points * 30)  # 增加候选点密度
        
        # 计算函数值
        y_values = np.array([self.function(x) for x in candidate_points])
        
        # 计算对数梯度（更适合指数函数）
        log_y_values = np.log(np.maximum(y_values, 1e-12))
        gradients = np.abs(np.gradient(log_y_values))
        
        # 在负值区域大幅增加权重（指数函数变化最剧烈的区域）
        negative_mask = candidate_points < 0
        gradients[negative_mask] *= 5.0  # 在负值区域增加5倍权重
        
        # 在接近0的区域也增加权重
        near_zero_mask = np.abs(candidate_points) < 1.0
        gradients[near_zero_mask] *= 2.0
        
        # 平滑权重以避免过度集中
        try:
            from scipy.ndimage import gaussian_filter1d
            gradients = gaussian_filter1d(gradients, sigma=1.5)
        except ImportError:
            # 如果没有scipy，使用简单的移动平均
            window_size = min(3, len(gradients) // 20)
            if window_size > 1:
                gradients = np.convolve(gradients, np.ones(window_size)/window_size, mode='same')
        
        # 根据梯度选择采样点
        weights = gradients / (np.sum(gradients) + 1e-12)
        cumulative_weights = np.cumsum(weights)
        
        # 生成目标累积概率
        target_probs = np.linspace(0, 1, n_points)
        
        # 找到对应的采样点
        sampled_indices = np.searchsorted(cumulative_weights, target_probs)
        sampled_indices = np.clip(sampled_indices, 0, len(candidate_points) - 1)
        
        # 确保边界点被包含
        sampled_indices[0] = 0
        sampled_indices[-1] = len(candidate_points) - 1
        
        # 去重并排序
        unique_indices = np.unique(sampled_indices)
        if len(unique_indices) < n_points:
            # 如果去重后点数不够，补充均匀分布的点
            additional_points = np.linspace(x_min, x_max, n_points - len(unique_indices))
            all_points = np.concatenate([candidate_points[unique_indices], additional_points])
            all_points = np.unique(all_points)
            return all_points[:n_points]
        
        return candidate_points[unique_indices]


class LogarithmicSampling(SamplingStrategy):
    """对数采样策略"""
    
    def generate_points(self, n_points: int, x_min: float, x_max: float) -> np.ndarray:
        """生成对数分布的采样点"""
        # 确保输入为正数
        if x_min <= 0:
            x_min = EPSILON_TINY
        if x_max <= 0:
            x_max = EPSILON_TINY
        
        # 对数空间均匀采样
        log_points = np.linspace(math.log(x_min), math.log(x_max), n_points)
        return np.exp(log_points)


class QuadraticSampling(SamplingStrategy):
    """二次采样策略（在中心区域密集采样）"""
    
    def generate_points(self, n_points: int, x_min: float, x_max: float) -> np.ndarray:
        """生成二次分布的采样点"""
        # 使用二次函数生成权重
        center = (x_min + x_max) / 2
        width = x_max - x_min
        
        # 生成均匀分布的点
        uniform_points = np.linspace(0, 1, n_points)
        
        # 应用二次变换
        quadratic_points = uniform_points ** 2
        
        # 映射到目标区间
        return x_min + quadratic_points * width


class InterpolationMethod(ABC):
    """插值方法抽象基类"""
    
    @abstractmethod
    def interpolate(self, x: float, x_points: np.ndarray, 
                   y_points: np.ndarray) -> float:
        """插值计算"""
        pass


class DirectLookup(InterpolationMethod):
    """直接查表方法"""
    
    def interpolate(self, x: float, x_points: np.ndarray, 
                   y_points: np.ndarray) -> float:
        """直接查表"""
        # 找到最近的索引
        idx = np.argmin(np.abs(x_points - x))
        return float(y_points[idx])


class LinearInterpolation(InterpolationMethod):
    """线性插值方法"""
    
    def interpolate(self, x: float, x_points: np.ndarray, 
                   y_points: np.ndarray) -> float:
        """线性插值"""
        # 处理边界情况
        if x <= x_points[0]:
            return float(y_points[0])
        if x >= x_points[-1]:
            return float(y_points[-1])
        
        # 找到插值区间
        idx = np.searchsorted(x_points, x)
        if idx == 0:
            return float(y_points[0])
        
        # 线性插值
        x0, x1 = x_points[idx - 1], x_points[idx]
        y0, y1 = y_points[idx - 1], y_points[idx]
        
        # 计算插值权重
        weight = (x - x0) / (x1 - x0)
        
        return float(y0 + weight * (y1 - y0))


class QuadraticInterpolation(InterpolationMethod):
    """二次插值方法"""
    
    def interpolate(self, x: float, x_points: np.ndarray, 
                   y_points: np.ndarray) -> float:
        """二次插值 - 硬件优化版本"""
        n = len(x_points)
        
        # 处理边界情况
        if x <= x_points[0]:
            return float(y_points[0])
        if x >= x_points[-1]:
            return float(y_points[-1])
        
        # 找到插值区间
        idx = np.searchsorted(x_points, x)
        
        # 选择三个点进行二次插值
        if idx <= 1:
            # 使用前三个点
            x0, x1, x2 = x_points[0], x_points[1], x_points[2]
            y0, y1, y2 = y_points[0], y_points[1], y_points[2]
        elif idx >= n - 1:
            # 使用后三个点
            x0, x1, x2 = x_points[-3], x_points[-2], x_points[-1]
            y0, y1, y2 = y_points[-3], y_points[-2], y_points[-1]
        else:
            # 使用当前点及其相邻点
            x0, x1, x2 = x_points[idx - 1], x_points[idx], x_points[idx + 1]
            y0, y1, y2 = y_points[idx - 1], y_points[idx], y_points[idx + 1]
        
        # 硬件优化的二次插值公式
        # 使用牛顿差分形式，减少除法运算
        dx01 = x1 - x0
        dx02 = x2 - x0
        dx12 = x2 - x1
        
        # 避免除零错误
        if abs(dx01) < 1e-12 or abs(dx02) < 1e-12 or abs(dx12) < 1e-12:
            # 回退到线性插值
            return float(y0 + (y1 - y0) * (x - x0) / dx01)
        
        # 牛顿差分二次插值
        # 计算一阶差分
        f01 = (y1 - y0) / dx01
        f12 = (y2 - y1) / dx12
        
        # 计算二阶差分
        f012 = (f12 - f01) / dx02
        
        # 二次插值
        result = y0 + f01 * (x - x0) + f012 * (x - x0) * (x - x1)
        
        # 数值稳定性检查
        if not np.isfinite(result):
            # 回退到线性插值
            return float(y0 + f01 * (x - x0))
        
        return float(result)


class LookupTable:
    """查找表类"""
    
    def __init__(self, config: LookupTableConfig):
        self.config = config
        self.x_points: Optional[np.ndarray] = None
        self.y_points: Optional[np.ndarray] = None
        self.function: Optional[Callable[[float], float]] = None
        
        # 缓存优化
        self._x_points_tensor_cache: Optional[torch.Tensor] = None
        self._y_points_tensor_cache: Optional[torch.Tensor] = None
        self._cached_dtype: Optional[torch.dtype] = None
        self._cached_device: Optional[torch.device] = None
        
        # 智能缓存
        self.cache = get_lookup_table_cache()
        self._config_hash = self._generate_config_hash()
        
        # 验证配置
        self._validate_config()
        
        # 初始化采样策略和插值方法
        self._init_sampling_strategy()
        self._init_interpolation_method()
    
    def _validate_config(self) -> None:
        """验证配置参数"""
        validate_point_count(
            self.config.point_count,
            POINT_COUNT_RANGE[0],
            POINT_COUNT_RANGE[1]
        )
        
        if self.config.interpolation_method not in INTERPOLATION_METHODS:
            raise InvalidInterpolationMethodError(
                self.config.interpolation_method,
                INTERPOLATION_METHODS
            )
        
        if self.config.sampling_strategy not in SAMPLING_STRATEGIES:
            raise ValueError(f"不支持的采样策略: {self.config.sampling_strategy}")
    
    def _generate_config_hash(self) -> str:
        """生成配置哈希"""
        import hashlib
        config_str = f"{self.config.point_count}_{self.config.interpolation_method}_{self.config.sampling_strategy}_{self.config.x_min}_{self.config.x_max}_{self.config.function_type}_{self.config.precision}"
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _init_sampling_strategy(self) -> None:
        """初始化采样策略"""
        if self.config.sampling_strategy == 'uniform':
            self.sampling_strategy = UniformSampling()
        elif self.config.sampling_strategy == 'adaptive':
            # 需要先设置函数
            self.sampling_strategy = None
        elif self.config.sampling_strategy == 'logarithmic':
            self.sampling_strategy = LogarithmicSampling()
        elif self.config.sampling_strategy == 'quadratic':
            self.sampling_strategy = QuadraticSampling()
        else:
            raise ValueError(f"不支持的采样策略: {self.config.sampling_strategy}")
    
    def _init_interpolation_method(self) -> None:
        """初始化插值方法"""
        if self.config.interpolation_method == 'direct':
            self.interpolation_method = DirectLookup()
        elif self.config.interpolation_method == 'linear':
            self.interpolation_method = LinearInterpolation()
        elif self.config.interpolation_method == 'quadratic':
            self.interpolation_method = QuadraticInterpolation()
        else:
            raise InvalidInterpolationMethodError(
                self.config.interpolation_method,
                INTERPOLATION_METHODS
            )
    
    def generate_table(self, function: Callable[[float], float]) -> None:
        """生成查找表"""
        self.function = function
        
        # 尝试从缓存获取
        cached_table = self.cache.get_table(self._config_hash)
        if cached_table is not None:
            self.x_points = cached_table['x_points']
            self.y_points = cached_table['y_points']
            return
        
        # 如果是自适应采样，需要先设置函数
        if self.config.sampling_strategy == 'adaptive':
            self.sampling_strategy = AdaptiveSampling(function)
        
        # 生成采样点
        self.x_points = self.sampling_strategy.generate_points(
            self.config.point_count,
            self.config.x_min,
            self.config.x_max
        )
        
        # 计算函数值
        self.y_points = np.array([function(x) for x in self.x_points])
        
        # 检查数值稳定性
        self._check_stability()
        
        # 缓存结果
        table_data = {
            'x_points': self.x_points,
            'y_points': self.y_points
        }
        self.cache.set_table(self._config_hash, table_data)
    
    def _check_stability(self) -> None:
        """检查数值稳定性"""
        if np.any(np.isnan(self.y_points)):
            raise ValueError("查找表包含 NaN 值")
        
        if np.any(np.isinf(self.y_points)):
            raise ValueError("查找表包含无穷大值")
    
    def lookup(self, x: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        """查找表查询"""
        if self.x_points is None or self.y_points is None:
            raise ValueError("查找表未初始化，请先调用 generate_table()")
        
        if isinstance(x, torch.Tensor):
            return self._lookup_tensor(x)
        else:
            return self._lookup_scalar(x)
    
    def _lookup_scalar(self, x: float) -> float:
        """标量查找 - 针对Softmax优化"""
        # 针对Softmax的边界处理
        if x <= self.x_points[0]:
            # 对于超出下界的情况，使用指数衰减近似
            if x < self.x_points[0] - 2.0:  # 如果超出太多，直接返回接近0的值
                return 1e-10
            else:
                # 使用线性外推，但限制在合理范围内
                x0, x1 = self.x_points[0], self.x_points[1]
                y0, y1 = self.y_points[0], self.y_points[1]
                slope = (y1 - y0) / (x1 - x0)
                extrapolated = y0 + slope * (x - x0)
                return max(1e-10, min(1.0, extrapolated))  # 限制在[1e-10, 1.0]范围内
        
        if x >= self.x_points[-1]:
            # 对于超出上界的情况，使用线性外推
            if x > self.x_points[-1] + 2.0:  # 如果超出太多，直接返回1.0
                return 1.0
            else:
                # 使用线性外推
                x0, x1 = self.x_points[-2], self.x_points[-1]
                y0, y1 = self.y_points[-2], self.y_points[-1]
                slope = (y1 - y0) / (x1 - x0)
                extrapolated = y1 + slope * (x - x1)
                return max(1e-10, min(1.0, extrapolated))  # 限制在[1e-10, 1.0]范围内
        
        return self.interpolation_method.interpolate(x, self.x_points, self.y_points)
    
    def _lookup_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """张量查找 - 向量化实现"""
        if self.config.interpolation_method == 'linear':
            # 使用向量化的线性插值
            return self._vectorized_linear_interpolation(x)
        elif self.config.interpolation_method == 'direct':
            # 使用向量化的直接查找
            return self._vectorized_direct_lookup(x)
        elif self.config.interpolation_method == 'quadratic':
            # 使用向量化的二次插值
            return self._vectorized_quadratic_interpolation(x)
        else:
            # 回退到逐元素处理
            return self._fallback_elementwise_lookup(x)
    
    def _vectorized_linear_interpolation(self, x: torch.Tensor) -> torch.Tensor:
        """向量化线性插值"""
        # 使用缓存的张量
        x_points_tensor, y_points_tensor = self._get_cached_tensors(x.dtype, x.device)
        
        # 使用torch.searchsorted进行高效查找
        indices = torch.searchsorted(x_points_tensor, x)
        indices = torch.clamp(indices, 1, len(self.x_points) - 1)
        
        # 获取插值点
        x0 = x_points_tensor[indices - 1]
        x1 = x_points_tensor[indices]
        y0 = y_points_tensor[indices - 1]
        y1 = y_points_tensor[indices]
        
        # 线性插值
        weight = (x - x0) / (x1 - x0 + 1e-12)  # 添加小常数防止除零
        result = y0 + weight * (y1 - y0)
        
        return result
    
    def _vectorized_direct_lookup(self, x: torch.Tensor) -> torch.Tensor:
        """向量化直接查找"""
        # 使用缓存的张量
        x_points_tensor, y_points_tensor = self._get_cached_tensors(x.dtype, x.device)
        
        # 找到最近的索引
        distances = torch.abs(x.unsqueeze(-1) - x_points_tensor.unsqueeze(0))
        indices = torch.argmin(distances, dim=-1)
        
        # 直接查找
        result = y_points_tensor[indices]
        
        return result
    
    def _vectorized_quadratic_interpolation(self, x: torch.Tensor) -> torch.Tensor:
        """向量化二次插值"""
        # 使用缓存的张量
        x_points_tensor, y_points_tensor = self._get_cached_tensors(x.dtype, x.device)
        
        # 找到插值区间
        indices = torch.searchsorted(x_points_tensor, x)
        indices = torch.clamp(indices, 1, len(self.x_points) - 2)
        
        # 获取三个插值点
        x0 = x_points_tensor[indices - 1]
        x1 = x_points_tensor[indices]
        x2 = x_points_tensor[indices + 1]
        y0 = y_points_tensor[indices - 1]
        y1 = y_points_tensor[indices]
        y2 = y_points_tensor[indices + 1]
        
        # 二次插值
        L0 = ((x - x1) * (x - x2)) / ((x0 - x1) * (x0 - x2) + 1e-12)
        L1 = ((x - x0) * (x - x2)) / ((x1 - x0) * (x1 - x2) + 1e-12)
        L2 = ((x - x0) * (x - x1)) / ((x2 - x0) * (x2 - x1) + 1e-12)
        
        result = y0 * L0 + y1 * L1 + y2 * L2
        
        return result
    
    def _fallback_elementwise_lookup(self, x: torch.Tensor) -> torch.Tensor:
        """回退的逐元素查找"""
        result = torch.zeros_like(x)
        flat_x = x.view(-1)
        flat_result = result.view(-1)
        
        for i in range(x.numel()):
            flat_result[i] = self._lookup_scalar(flat_x[i].item())
        
        return result
    
    def _get_cached_tensors(self, dtype: torch.dtype, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取缓存的张量"""
        if (self._x_points_tensor_cache is not None and 
            self._y_points_tensor_cache is not None and
            self._cached_dtype == dtype and 
            self._cached_device == device):
            return self._x_points_tensor_cache, self._y_points_tensor_cache
        
        # 创建新的张量并缓存
        self._x_points_tensor_cache = torch.tensor(self.x_points, dtype=dtype, device=device)
        self._y_points_tensor_cache = torch.tensor(self.y_points, dtype=dtype, device=device)
        self._cached_dtype = dtype
        self._cached_device = device
        
        return self._x_points_tensor_cache, self._y_points_tensor_cache
    
    def _clear_cache(self):
        """清除缓存"""
        self._x_points_tensor_cache = None
        self._y_points_tensor_cache = None
        self._cached_dtype = None
        self._cached_device = None
    
    def get_table_info(self) -> Dict[str, any]:
        """获取查找表信息"""
        return {
            'point_count': self.config.point_count,
            'interpolation_method': self.config.interpolation_method,
            'sampling_strategy': self.config.sampling_strategy,
            'x_range': (self.config.x_min, self.config.x_max),
            'function_type': self.config.function_type,
            'x_points': self.x_points.tolist() if self.x_points is not None else None,
            'y_points': self.y_points.tolist() if self.y_points is not None else None
        }
    
    def save_table(self, filepath: str) -> None:
        """保存查找表到文件"""
        import json
        
        table_data = {
            'config': {
                'point_count': self.config.point_count,
                'interpolation_method': self.config.interpolation_method,
                'sampling_strategy': self.config.sampling_strategy,
                'x_min': self.config.x_min,
                'x_max': self.config.x_max,
                'function_type': self.config.function_type,
                'precision': self.config.precision
            },
            'x_points': self.x_points.tolist() if self.x_points is not None else None,
            'y_points': self.y_points.tolist() if self.y_points is not None else None
        }
        
        with open(filepath, 'w') as f:
            json.dump(table_data, f, indent=2)
    
    def load_table(self, filepath: str) -> None:
        """从文件加载查找表"""
        import json
        
        with open(filepath, 'r') as f:
            table_data = json.load(f)
        
        # 恢复配置
        config_data = table_data['config']
        self.config = LookupTableConfig(**config_data)
        
        # 恢复数据
        self.x_points = np.array(table_data['x_points']) if table_data['x_points'] else None
        self.y_points = np.array(table_data['y_points']) if table_data['y_points'] else None
        
        # 重新初始化
        self._init_sampling_strategy()
        self._init_interpolation_method()


class ExpLookupTable(LookupTable):
    """指数函数查找表"""
    
    def __init__(self, point_count: int = 2000, 
                 interpolation_method: str = 'quadratic',
                 sampling_strategy: str = 'adaptive',
                 x_min: float = -20.0,  # 扩大范围以覆盖Softmax的实际输入范围
                 x_max: float = 0.0):   # 调整上界，因为Softmax中减去最大值后通常为负值
        config = LookupTableConfig(
            point_count=point_count,
            interpolation_method=interpolation_method,
            sampling_strategy=sampling_strategy,
            x_min=x_min,
            x_max=x_max,
            function_type='exp'
        )
        super().__init__(config)
        
        # 生成指数函数查找表
        self.generate_table(MathUtils.safe_exp)
    
    def lookup_exp(self, x: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        """查找指数函数值"""
        return self.lookup(x)


class SigmoidLookupTable(LookupTable):
    """Sigmoid 函数查找表"""
    
    def __init__(self, point_count: int = 800,
                 interpolation_method: str = 'linear',
                 sampling_strategy: str = 'uniform',
                 x_min: float = -10.0,
                 x_max: float = 10.0):
        config = LookupTableConfig(
            point_count=point_count,
            interpolation_method=interpolation_method,
            sampling_strategy=sampling_strategy,
            x_min=x_min,
            x_max=x_max,
            function_type='sigmoid'
        )
        super().__init__(config)
        
        # 生成 Sigmoid 函数查找表
        self.generate_table(MathUtils.sigmoid_stable)
    
    def lookup_sigmoid(self, x: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        """查找 Sigmoid 函数值"""
        return self.lookup(x)


class LookupTableManager:
    """查找表管理器"""
    
    def __init__(self):
        self.tables: Dict[str, LookupTable] = {}
    
    def create_table(self, name: str, table_type: str, **kwargs) -> LookupTable:
        """创建查找表"""
        if table_type == 'exp':
            table = ExpLookupTable(**kwargs)
        elif table_type == 'sigmoid':
            table = SigmoidLookupTable(**kwargs)
        else:
            raise ValueError(f"不支持的查找表类型: {table_type}")
        
        self.tables[name] = table
        return table
    
    def get_table(self, name: str) -> LookupTable:
        """获取查找表"""
        if name not in self.tables:
            raise KeyError(f"查找表不存在: {name}")
        return self.tables[name]
    
    def remove_table(self, name: str) -> None:
        """移除查找表"""
        if name in self.tables:
            del self.tables[name]
    
    def list_tables(self) -> List[str]:
        """列出所有查找表"""
        return list(self.tables.keys())
    
    def clear_tables(self) -> None:
        """清空所有查找表"""
        self.tables.clear()


# 全局查找表管理器
_table_manager = LookupTableManager()


def get_table_manager() -> LookupTableManager:
    """获取全局查找表管理器"""
    return _table_manager


def create_exp_table(name: str = "default_exp", **kwargs) -> ExpLookupTable:
    """创建指数函数查找表"""
    return _table_manager.create_table(name, 'exp', **kwargs)


def create_sigmoid_table(name: str = "default_sigmoid", **kwargs) -> SigmoidLookupTable:
    """创建 Sigmoid 函数查找表"""
    return _table_manager.create_table(name, 'sigmoid', **kwargs)
