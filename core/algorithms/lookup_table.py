"""
查找表实现模块
提供高效的查找表算法，支持均匀采样和非均匀采样
"""

import math
import hashlib
import pickle
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass

from core.base.constants import (
    DEFAULT_X_STRUCT_MAX_INT, DEFAULT_X_STRUCT_MIN_INT, TABLE_THRESH, BIT_LEN_RANGE, INTERPOLATION_METHODS,
    SAMPLING_STRATEGIES, NUMERICAL_STABILITY, EPSILON_TINY,
    DEFAULT_BIT_LEN, DEFAULT_DTYPE, DEFAULT_DTYPE_LEN, DEFAULT_UNSIGNED_TYPE, DEFAULT_INTERPOLATION,
    DEFAULT_SAMPLING_STRATEGY, DEFAULT_FIXED_POINT_FORMAT, DATA_TYPE_MAP
)
from core.base.exceptions import (
    InvalidPointCountError, InvalidInterpolationMethodError,
    validate_bitlen, validate_dtype
)
from core.utils.smart_cache import get_lookup_table_cache
from config.config import LookupTableConfig





class SamplingStrategy(ABC):
    """采样策略抽象基类"""
    
    @abstractmethod
    def generate_points(self, n_points: int, x_struct_min: torch.Tensor, x_struct_max: torch.Tensor) -> np.ndarray:
        """生成采样点"""
        pass


class UniformSampling(SamplingStrategy):
    """均匀采样策略"""
    
    def generate_points(self, n_points: int, x_struct_min: torch.Tensor, x_struct_max: torch.Tensor) -> np.ndarray:
        """生成均匀分布的采样点"""
        return np.linspace(x_struct_min, x_struct_max, n_points)


class AdaptiveSampling(SamplingStrategy):
    """自适应采样策略（在函数变化剧烈区域密集采样）"""
    
    def __init__(self, function: Callable[[float], float], 
                 sensitivity: float = 0.1):
        self.function = function
        self.sensitivity = sensitivity
    
    def generate_points(self, n_points: int, x_struct_min: torch.Tensor, x_struct_max: torch.Tensor) -> np.ndarray:
        """生成自适应采样点 - 针对指数函数优化"""
        # 首先生成密集的候选点
        candidate_points = np.linspace(x_struct_min, x_struct_max, n_points * 30)  # 增加候选点密度
        
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
            additional_points = np.linspace(x_struct_min, x_struct_max, n_points - len(unique_indices))
            all_points = np.concatenate([candidate_points[unique_indices], additional_points])
            all_points = np.unique(all_points)
            return all_points[:n_points]
        
        return candidate_points[unique_indices]


class LogarithmicSampling(SamplingStrategy):
    """对数采样策略"""
    
    def generate_points(self, n_points: int, x_struct_min: torch.Tensor, x_struct_max: torch.Tensor) -> np.ndarray:
        """生成对数分布的采样点"""
        # 确保输入为正数
        if x_struct_min <= 0:
            x_struct_min = EPSILON_TINY
        if x_struct_max <= 0:
            x_struct_max = EPSILON_TINY
        
        # 对数空间均匀采样
        log_points = np.linspace(math.log(x_struct_min), math.log(x_struct_max), n_points)
        return np.exp(log_points)


class QuadraticSampling(SamplingStrategy):
    """二次采样策略（在中心区域密集采样）"""
    
    def generate_points(self, n_points: int, x_struct_min: torch.Tensor, x_struct_max: torch.Tensor) -> np.ndarray:
        """生成二次分布的采样点"""
        # 使用二次函数生成权重
        center = (x_struct_min + x_struct_max) / 2
        width = x_struct_max - x_struct_min
        
        # 生成均匀分布的点
        uniform_points = np.linspace(0, 1, n_points)
        
        # 应用二次变换
        quadratic_points = uniform_points ** 2
        
        # 映射到目标区间
        return x_struct_min + quadratic_points * width

class BinarySampling(SamplingStrategy):
    """二进制采样策略（按照view的方法扫描每个浮点数或定点数）"""

    def validate_range(self, x_struct_min: torch.tensor, x_struct_max: torch.tensor):
        """验证输入范围"""
        if x_struct_min.dtype not in [torch.float32, torch.bfloat16]:
            raise ValueError(f"不支持的x_struct_min类型: {x_struct_min.dtype}")
        if x_struct_max.dtype not in [torch.float32, torch.bfloat16]:
            raise ValueError(f"不支持的x_struct_max类型: {x_struct_max.dtype}")
        if x_struct_min.dtype != x_struct_max.dtype:
            raise ValueError(f"x_struct_min和x_struct_max类型不一致: {x_struct_min.dtype}!= {x_struct_max.dtype}")

    def generate_points(self, bit_len: int, x_struct_min: torch.tensor, x_struct_max: torch.tensor) -> np.ndarray:
        """生成二进制分布的采样点(支持float32、bfloat16)"""

        # 验证输入范围
        self.validate_range(x_struct_min, x_struct_max)

        # 计算浮点数位数
        x_start = 0
        x_end = 0
        total_len = 0
        if x_struct_min.dtype == torch.float32:
            total_len = 32
            transit_type = torch.uint32
            zero_len = total_len - bit_len
            x_start = x_struct_min.view(transit_type).item() >> zero_len
            x_end = x_struct_max.view(transit_type).item() >> zero_len
        else:
            total_len = 16
            transit_type = torch.uint16
            zero_len = total_len - bit_len
            x_start = x_struct_min.view(transit_type).item() >> zero_len
            x_end = x_struct_max.view(transit_type).item() >> zero_len

        binary_points = []
        for i in range(x_start, x_end):
            binary_points.append(torch.tensor(i, dtype=x_struct_min.uint16).view(x_struct_min.dtype))
            

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
        
        # 验证配置
        self._validate_config()
        
        # 初始化采样策略和插值方法
        self._init_sampling_strategy()
        self._init_interpolation_method()

        self.sample = 9
    
    def _validate_config(self) -> None:
        """验证配置参数"""
        validate_bitlen(
            self.config.bit_len,
            BIT_LEN_RANGE[0],
            BIT_LEN_RANGE[1]
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
        static_config = self.config.__dict__.copy()
        del static_config["x_struct_min"]
        del static_config["x_struct_max"]
        return hashlib.md5(pickle.dumps(static_config, protocol=5)).hexdigest()
    
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
        elif self.config.sampling_strategy == 'binary':
            self.sampling_strategy = BinarySampling()
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
        if self.config.use_cache:
            cached_table = self.cache.get_table(self._generate_config_hash())
            if cached_table is not None:
                self.x_points = cached_table['x_points']
                self.y_points = cached_table['y_points']
                return
        
        # 如果是自适应采样，需要先设置函数
        if self.config.sampling_strategy == 'adaptive':
            self.sampling_strategy = AdaptiveSampling(function)

        self.x_points = []
        self.y_points = []
        
        # 使用位操作生成查找表

        # def logical_right_shift(x_struct: torch.Tensor, shift: int, unsigned_type=torch.uint16):
        #     '''x_struct是unsigned_type类型，返回值unsigned_type类型'''
        #     return (x_struct.int() >> shift).to(unsigned_type)

        # def logical_left_shift(x_struct: torch.Tensor, shift: int, unsigned_type=torch.uint16):
        #     '''x_struct是unsigned_type类型，返回值unsigned_type类型'''
        #     return (x_struct.int() << shift).to(unsigned_type)

        startX = self.config.x_struct_min_int # python内置int，不会出现负数影响位运算
        endX = self.config.x_struct_max_int
        step = 1 << (self.config.zero_len)
        

        for x in range(startX, endX, step):
            x_target_type = torch.tensor(x & self.config.unsigned_mask, dtype=self.config.unsigned_type).view(self.config.dtype) # 从LSB截断为unsigned_type，再转换为dtype
            y_target_type = self.function(x_target_type)
            self.x_points.append(x_target_type.item())
            self.y_points.append(y_target_type.item())

        self.x_points = np.array(self.x_points)
        self.y_points = np.array(self.y_points)
        
        # 缓存结果
        table_data = {
            'x_points': self.x_points,
            'y_points': self.y_points
        }
        if self.config.use_cache:
            self.cache.set_table(self._generate_config_hash(), table_data)
    
    def lookup(self, x: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        """查找表查询，分类float还是tensor，如果是float则转换为tensor，如果是tensor则进行递归展开"""
        if self.x_points is None or self.y_points is None:
            raise ValueError("查找表未初始化，请先调用 generate_table()")
        
        if isinstance(x, torch.Tensor):
            init_dtype = x.dtype
            return torch.tensor(self._lookup_tensor(x.to(self.config.dtype)), dtype=init_dtype)
        else:
            return self._lookup_scalar(torch.tensor(x, dtype=self.config.dtype)).item()

    def _lookup_tensor(self, x: torch.Tensor) -> list:
        """张量查找，返回为多维数组，在最外层再tensor化"""
        if len(x.shape) == 0:
            return self._lookup_scalar(x)
        return [self._lookup_tensor(x_i) for x_i in x]
    
    def _lookup_scalar(self, x: torch.Tensor) -> float:
        """标量查找，tensor中只有1个元素"""
        x_struct = x.view(self.config.unsigned_type).int() >> (self.config.zero_len)
        result = self.y_points[x_struct]  # 截断位数直接算出地址，无需查表
        # if self.sample > 0:
        #     self.sample -= 1
        #     print(self.config.unsigned_type, self.config.zero_len, self.config.dtype)
        #     print("sample: ", x.item(), 'struct=', x_struct.item(), 'lookup=', result, 'ans=', self.function(x).item())
        return result
    
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
        print("Fallback to elementwise lookup")
        result = torch.zeros_like(x)
        flat_x = x.view(-1)
        flat_result = result.view(-1)
        
        for i in range(x.numel()):
            flat_result[i] = self._lookup_scalar(flat_x[i])
        
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
        return self.config
    
    def save_table(self, filepath: str) -> None:
        """保存查找表到文件"""
        import json
        
        table_data = {
            'config': self.config.__dict__,
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

# 快捷函数

def create_exp_table(table_name: str = "default_exp",
                     bit_len: int = DEFAULT_BIT_LEN,
                     interpolation_method = DEFAULT_INTERPOLATION,
                     sampling_strategy: str = DEFAULT_SAMPLING_STRATEGY,
                     x_struct_min_int: int = DEFAULT_X_STRUCT_MIN_INT,
                     x_struct_max_int: int = DEFAULT_X_STRUCT_MAX_INT,
                     ) -> LookupTable:
    """创建指数函数查找表"""

    config = LookupTableConfig(
        bit_len=bit_len,
        interpolation_method=interpolation_method,
        sampling_strategy=sampling_strategy,
        x_struct_min_int=x_struct_min_int,
        x_struct_max_int=x_struct_max_int,
        table_name=table_name
    )
    tbl = LookupTable(config)
    tbl.generate_table(torch.exp)
    return tbl

def create_sigmoid_table(table_name: str = "default_sigmoid",
                         bit_len: int = DEFAULT_BIT_LEN,
                         interpolation_method = DEFAULT_INTERPOLATION,
                         sampling_strategy: str = DEFAULT_SAMPLING_STRATEGY,
                         x_struct_min_int: int = DEFAULT_X_STRUCT_MIN_INT,
                         x_struct_max_int: int = DEFAULT_X_STRUCT_MAX_INT,
                         ) -> LookupTable:
    """创建 Sigmoid 函数查找表"""
    config = LookupTableConfig(
        bit_len=bit_len,
        interpolation_method=interpolation_method,
        sampling_strategy=sampling_strategy,
        x_struct_min_int=x_struct_min_int,
        x_struct_max_int=x_struct_max_int,
        table_name=table_name
    )
    tbl = LookupTable(config)
    tbl.generate_table(torch.sigmoid)
    return tbl
