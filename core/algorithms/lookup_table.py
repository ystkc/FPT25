"""
查找表实现模块
提供高效的查找表算法，支持均匀采样和非均匀采样
"""


import math
import hashlib
import pickle
import random
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass

from core.base.constants import (
    TABLE_THRESH, BIT_LEN_RANGE, INTERPOLATION_METHODS,
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
    
    def __init__(self, config: LookupTableConfig):
        self.config = config
        self.sample = 3
        self.x_points: Optional[torch.Tensor] = None
    
    @abstractmethod
    def generate_points(self) -> torch.Tensor:
        """根据提供的采样区间、方法和点数生成采样点"""

    @abstractmethod
    def set_cache(self, x_points: torch.Tensor):
        """手动设置缓存，在使用缓存而不是生成点时使用"""

    @abstractmethod
    def parallel_find_point(self, value: torch.Tensor) -> torch.Tensor:
        """（并行）根据提供的采样点和值查找对应的索引（value为二维张量）"""

    @abstractmethod
    def find_point(self, value: torch.Tensor) -> torch.Tensor:
        """根据提供的采样点和值查找对应的索引"""

class BinarySampling(SamplingStrategy):
    """二进制采样策略"""

    def generate_points(self) -> torch.Tensor:
        """生成二进制采样点"""
        step = 1 << (self.config.zero_len)

        seq = []
        for startX, endX in self.config.x_struct_range_int:
            seq += list(range(startX, endX+1, step))

        x_ints = [x & self.config.unsigned_mask for x in seq]
        
        x_points = torch.tensor(x_ints, dtype=self.config.unsigned_type).view(self.config.dtype)
        # 二进制采样寻址无需x，所以不存储x_points
        return x_points
    
    def set_cache(self, _: torch.Tensor):
        # 二进制采样寻址无需x，所以不存储x_points
        pass
    
    def find_point(self, value: torch.Tensor) -> torch.Tensor:
        """二进制直接右移就是查找索引"""
        # if self.sample > 0:
        #     self.sample -= 1
        #     print("sample:", self.config.unsigned_type, self.config.zero_len, value.shape, value[:10], value.view(self.config.unsigned_type).shape)
        return value.view(self.config.unsigned_type).int() >> (self.config.zero_len)

    def parallel_find_point(self, value: torch.Tensor) -> torch.Tensor:
        """并行查找二进制采样点"""
        # if len(value.shape) > 1:
        #     return torch.stack([self.parallel_find_point(v) for v in value], dim=0)
        return self.find_point(value)

class RandomSampling(SamplingStrategy):
    """定点数随机采样策略"""

    def generate_points(self) -> torch.Tensor:
        """生成随机采样点"""
        random.seed(self.config.random_seed)
        # print("random sample", self.config.sample_count, "range", self.config.x_struct_range_int)

        (PosStartX, PosEndX), (NegStartX, NegEndX) = self.config.x_struct_range_int

        # 由于正数是int越大浮点数越大，负数是int越大浮点数越小，所以要分两部分排序
        x_sample = sorted(random.sample(range(NegStartX, NegEndX+1), self.config.sample_count >> 1), reverse=True) + sorted(random.sample(range(PosStartX, PosEndX+1), self.config.sample_count >> 1))

        self.x_points = torch.tensor(x_sample, dtype=self.config.unsigned_type).view(self.config.dtype)
        return self.x_points
    
    def set_cache(self, x_points: torch.Tensor):
        """手动设置缓存"""
        self.x_points = x_points
    
    def find_point(self, value: torch.Tensor) -> torch.Tensor:
        """找到第一个大于value的值并返回索引"""
        idx = torch.searchsorted(self.x_points, value)
        return idx

    def parallel_find_point(self, value: torch.Tensor) -> torch.Tensor:
        """并行查找随机采样点"""
        idxs = torch.searchsorted(self.x_points, value)
        return idxs




class UniformSampling(SamplingStrategy):
    """均匀采样策略"""
        
SAMPLING_STRATEGIES_MAP = {
    'binary': BinarySampling,
    'random': RandomSampling
}

class InterpolationMethod(ABC):
    """插值方法抽象基类"""
    
    @abstractmethod
    def __init__(self, x_points: torch.Tensor, y_points: torch.Tensor):
        """准备插值，例如计算系数等，加快插值速度"""

    @abstractmethod
    def interpolate(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """根据索引和值进行插值"""

    @abstractmethod
    def parallel_interpolate(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """并行插值"""


class DirectLookup(InterpolationMethod):
    """直接查表方法"""

    def __init__(self, x_points: torch.Tensor, y_points: torch.Tensor):
        """准备直接查表"""
        self.y_points = y_points
        self.table_len = len(y_points)
    
    def interpolate(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """直接查表"""
        return self.y_points[idx]

    def parallel_interpolate(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """并行直接查表"""
        def index_select(ts: torch.Tensor, dm: int, iindex: torch.Tensor):
            if len(iindex.shape) > 1:
                return torch.stack([index_select(ts, dm, i) for i in iindex], dim=0)
            # 将iindex中所有负数值偏移table_len
            # final_idx = torch.where(iindex < 0, 
            #                     self.table_len + iindex, 
            #                     iindex)
            # print(final_idx[:10], final_idx[-10:])
            return torch.index_select(ts, 0, 
                    torch.where(iindex < 0, 
                                self.table_len + iindex, 
                                iindex))
        return index_select(self.y_points, 0, idx)

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
    
INTERPOLATION_METHODS_MAP = {
    'direct': DirectLookup,
    'linear': LinearInterpolation,
    'quadratic': QuadraticInterpolation
}


class LookupTable:
    """查找表类"""
    
    def __init__(self, config: LookupTableConfig):
        self.config = config
        self.x_points: Optional[torch.Tensor] = None
        self.y_points: Optional[torch.Tensor] = None
        self.table_len: Optional[int] = None
        self.function: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
        
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
        self.sampling_strategy: SamplingStrategy = SAMPLING_STRATEGIES_MAP[self.config.sampling_strategy](self.config)
        self.interpolation_method: InterpolationMethod = None

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
        del static_config["x_struct_range"]
        del static_config["interpolation_method"]
        del static_config["table_name"]
        del static_config["use_parallel_lookup"]
        return hashlib.md5(pickle.dumps(static_config, protocol=5)).hexdigest()
    
    def generate_table(self, function: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """生成查找表"""
        self.function = function
        
        # 尝试从缓存获取
        if self.config.use_cache:
            cached_table = self.cache.get_table(self._generate_config_hash())
            if cached_table is not None:
                self.x_points = cached_table['x_points']
                self.table_len = len(self.x_points)
                self.sampling_strategy.set_cache(self.x_points)
                self.y_points = cached_table['y_points']
                self.interpolation_method = INTERPOLATION_METHODS_MAP[self.config.interpolation_method](self.x_points, self.y_points)
                return
        
        # 使用位操作生成查找表

        # def logical_right_shift(x_struct: torch.Tensor, shift: int, unsigned_type=torch.uint16):
        #     '''x_struct是unsigned_type类型，返回值unsigned_type类型'''
        #     return (x_struct.int() >> shift).to(unsigned_type)

        # def logical_left_shift(x_struct: torch.Tensor, shift: int, unsigned_type=torch.uint16):
        #     '''x_struct是unsigned_type类型，返回值unsigned_type类型'''
        #     return (x_struct.int() << shift).to(unsigned_type)

        x_points_tensor: torch.Tensor = self.sampling_strategy.generate_points()
        self.y_points = self.function(x_points_tensor)#.numpy()
        self.x_points = x_points_tensor#.numpy()
        self.table_len = len(self.x_points)
        self.interpolation_method = INTERPOLATION_METHODS_MAP[self.config.interpolation_method](self.x_points, self.y_points)

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
        
        if self.config.use_parallel_lookup:
            idx = self.sampling_strategy.parallel_find_point(x.to(self.config.dtype))
            return self.interpolation_method.parallel_interpolate(x, idx)
        
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
    
    def _lookup_scalar(self, x: torch.Tensor) -> torch.Tensor:
        """标量查找，tensor中只有1个元素"""
        result = self.interpolation_method.interpolate(self.sampling_strategy.find_point(x))  # 截断位数直接算出地址，无需遍历
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
                     x_struct_range_int: list = [(0x0000, 0x7f7f), (0x8000, 0xff7f)],
                     ) -> LookupTable:
    """创建指数函数查找表"""

    config = LookupTableConfig(
        bit_len=bit_len,
        interpolation_method=interpolation_method,
        sampling_strategy=sampling_strategy,
        x_struct_range_int=x_struct_range_int,
        table_name=table_name
    )
    tbl = LookupTable(config)
    tbl.generate_table(torch.exp)
    return tbl

def create_sigmoid_table(table_name: str = "default_sigmoid",
                         bit_len: int = DEFAULT_BIT_LEN,
                         interpolation_method = DEFAULT_INTERPOLATION,
                         sampling_strategy: str = DEFAULT_SAMPLING_STRATEGY,
                         x_struct_range_int: list = [(0x0000, 0x7f7f), (0x8000, 0xff7f)],
                         ) -> LookupTable:
    """创建 Sigmoid 函数查找表"""
    config = LookupTableConfig(
        bit_len=bit_len,
        interpolation_method=interpolation_method,
        sampling_strategy=sampling_strategy,
        x_struct_range_int=x_struct_range_int,
        table_name=table_name
    )
    tbl = LookupTable(config)
    tbl.generate_table(torch.sigmoid)
    return tbl
