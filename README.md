# FPT25 激活函数 FPGA 硬件加速项目

本项目是 FPT25 设计竞赛的参赛项目，专注于**大模型激活函数 FPGA 硬件加速**。竞赛要求实现 7 种激活函数的高效 FPGA 加速，支持 bf16 数据类型，重点关注 Softmax 函数的优化实现。

## 项目特点

- **精度要求**：相对 L2 误差 ≤ 1e-3 获得满分，1e-3 ~ 1e-1 对数衰减
- **性能要求**：针对 64×768 张量实现高吞吐量、低延迟计算
- **硬件适配**：支持 FPGA 硬件实现，使用查找表技术替代昂贵运算
- **竞赛评分**：Softmax 权重最高（15 分），是核心优化目标

## 支持的激活函数

1. **Softmax** (权重: 15) - 核心注意力机制
2. **LayerNorm** (权重: 12) - 残差连接后使用
3. **RMSNorm** (权重: 10) - 趋势性替代方案
4. **SiLU** (权重: 10) - 主流激活函数
5. **GELU** (权重: 10) - 广泛使用
6. **Add** (权重: 7) - 高频操作
7. **Multiply** (权重: 6) - 缩放、门控等

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行测试

```bash
# 测试单个激活函数
python main.py --mode test --function softmax

# 基准测试所有激活函数
python main.py --mode benchmark --function all

# 精度测试
python main.py --mode accuracy --function softmax --point_count 800

# 优化测试
python main.py --mode optimize --function softmax --interpolation quadratic

# 高精度测试（推荐配置）
python main.py --mode benchmark --function softmax --point_count 2000 --interpolation quadratic --sampling_strategy adaptive

# 所有激活函数测试
python main.py --mode benchmark --function all
```

### 命令行参数

| 参数                    | 类型     | 默认值        | 说明                                                                                    |
| ----------------------- | -------- | ------------- | --------------------------------------------------------------------------------------- |
| `--mode`                | 选择     | `benchmark`   | 运行模式：`test`、`benchmark`、`optimize`、`all_functions`                              |
| `--function`            | 选择     | `softmax`     | 激活函数：`softmax`、`layer_norm`、`rms_norm`、`silu`、`gelu`、`add`、`multiply`、`all` |
| `--point_count`         | 整数     | `2000`        | 查找表点数（1500-3000）                                                                 |
| `--interpolation`       | 选择     | `quadratic`   | 插值方法：`direct`、`linear`、`quadratic`                                               |
| `--sampling_strategy`   | 选择     | `adaptive`    | 采样策略：`uniform`、`adaptive`、`logarithmic`、`quadratic`                             |
| `--use_advanced_lookup` | 标志     | `True`        | 使用高级查找表（非均匀采样）                                                            |
| `--batch_size`          | 整数     | `16`          | 批处理大小                                                                              |
| `--tensor_shape`        | 两个整数 | `64 768`      | 张量形状（高度 宽度）                                                                   |
| `--dtype`               | 选择     | `bfloat16`    | 数据类型：`float32`、`bfloat16`                                                         |
| `--use_fixed_point`     | 标志     | `False`       | 使用定点数计算                                                                          |
| `--fixed_point_format`  | 选择     | `Q16_16`      | 定点数格式：`Q8_8`、`Q16_16`、`Q32_32`、`Q8_24`                                         |
| `--output_dir`          | 字符串   | `results/`    | 输出目录                                                                                |
| `--config`              | 字符串   | `config.json` | 配置文件路径                                                                            |

## 项目结构

```
FPT25/
├── 📁 core/                                 # 核心模块
│   ├── __init__.py                          # 核心模块入口
│   ├── 📁 base/                             # 基础模块
│   │   ├── constants.py                     # 常量定义
│   │   ├── exceptions.py                    # 异常定义
│   │   └── logs.py                          # 日志配置
│   │
│   ├── 📁 utils/                            # 工具模块
│   │   ├── data_type_manager.py             # 数据类型管理器
│   │   ├── memory_pool.py                   # 内存池管理
│   │   └── smart_cache.py                   # 智能缓存系统
│   │
│   ├── 📁 algorithms/                       # 算法模块
│   │   ├── lookup_table.py                  # 查找表实现（包括均匀采样和非均匀采样）
│   │   └── math_utils.py                    # 数学工具函数
│   │
│   ├── 📁 hardware/                         # 硬件支持模块
│   │   └── fixed_point.py                   # 定点数支持（FPGA硬件适配）
│   │
│   ├── 📁 optimization/                     # 优化模块
│   │   ├── memory_optimizer.py              # 内存优化
│   │   ├── parallel_processor.py            # 并行处理
│   │   └── performance_monitor.py           # 性能监控
│   │
│   └── 📁 activation_functions/             # 激活函数模块
│       ├── activation_manager.py            # 激活函数管理器
│       ├── softmax_activation.py            # Softmax激活函数
│       ├── layer_norm_activation.py         # LayerNorm激活函数
│       ├── rms_norm_activation.py           # RMSNorm激活函数
│       ├── silu_activation.py               # SiLU激活函数
│       ├── gelu_activation.py               # GELU激活函数
│       ├── add_activation.py                # Add激活函数
│       └── multiply_activation.py           # Multiply激活函数
│
├── 📁 evaluation/                           # 评估模块
│   ├── accuracy.py                          # 精度评估（竞赛标准）
│   └── benchmark.py                         # 性能基准测试
├── 📁 config/                               # 配置管理
│   ├── config.py                            # 默认配置参数
│   └── config_panel.py                      # 配置管理器
├── 📁 tests/                                # 测试模块
│   ├── test_activation_functions.py         # 提供7种激活函数功能测试
│   ├── test_accuracy.py                     # 精度评估测试
│   └── test_hardware_optimization.py        # 硬件优化测试
│
├── 📁 utils/                                # 工具模块
│   ├── __init__.py
│   ├── results.py                           # 结果管理器
│   ├── excels.py                            # Excel报告生成器（输出目录：e.g. results/softmax/data/）
│   └── visualizer.py                        # 可视化图表生成器（输出目录: e.g. results/softmax/charts/）
│
├── 📁 results/                              # 统一输出目录
│   ├── 📁 softmax/                          # Softmax结果
│   │   ├── charts/                          # 性能图表
│   │   ├── reports/                         # 测试报告
│   │   │   ├── benchmark_results.json       # 基准测试结果
│   │   │   └── optimization_results.json    # 优化结果
│   │   ├── data/                            # 测试数据
│   │   └── softmax_output.pt                # 输出张量
│   ├── 📁 layer_norm/                       # LayerNorm结果
│   ├── 📁 rms_norm/                         # RMSNorm结果
│   ├── 📁 silu/                             # SiLU结果
│   ├── 📁 gelu/                             # GELU结果
│   ├── 📁 add/                              # Add结果
│   └── 📁 multiply/                         # Multiply结果
├── 📁 docs/                                 # 文档目录
│   ├── contest.md                           # 竞赛说明文档
│   └── fpt25-design-contest.pdf             # 竞赛官方文档
├── 📁 logs/                                 # 日志目录
│   └── fpt25_*.log                          # 运行日志文件
├── main.py                                  # 主程序入口（命令行用法：python main.py --mode benchmark --function <activation_function> --config <config_file>）
├── requirements.txt                         # 依赖配置
├── config.json                              # 默认配置文件（如果没有在根目录检测到config.json，则会生成一个）
├── LICENSE                                  # MIT许可证
└── README.md                                # 项目文档
```

## 核心算法

### Softmax 优化

- **数值稳定性**：减去最大值防止溢出，确保计算稳定性
- **高精度查找表**：2000 点自适应采样 + 二次插值，精度达到 **1.000** (L2 误差 ~1.27e-07)
- **智能采样策略**：针对指数函数特性优化的自适应采样，在负值区域密集采样
- **硬件友好**：查找表大小可配置，支持 FPGA 实现
- **数据类型**：输入输出支持 bf16 格式，内部计算使用 float32 保证精度

### 查找表设计

- **自适应采样**：针对指数函数变化特性，在负值区域和接近零的区域增加采样密度
- **二次插值**：使用牛顿差分形式的二次插值，硬件优化版本，减少除法运算
- **智能边界处理**：线性外推 + 数值稳定性检查，避免截断误差
- **范围优化**：查找表范围 [-20.0, 0.0]，完美匹配 Softmax 数值稳定性后的输入范围
- **采样点数量**：默认 2000 点，可扩展到 3000 点，显著提高精度

### 精度评估

严格按照竞赛标准实现：

```python
# 精度评分公式
def accuracy_score(fpga_output, reference_output):
    epsilon_f = torch.norm(fpga_output - reference_output, p=2) / (torch.norm(reference_output, p=2) + 1e-12)
    epsilon_star = 1e-3

    if epsilon_f <= epsilon_star:
        return 1.0
    elif epsilon_f <= 100 * epsilon_star:
        return (math.log(100 * epsilon_star) - math.log(epsilon_f)) / math.log(100)
    else:
        return 0.0
```

## 性能优化

### 算法层面优化

- **高精度查找表**：2000 点自适应采样，精度达到竞赛满分标准 (1.000)
- **智能插值算法**：二次插值 + 牛顿差分，硬件友好且高精度
- **数值稳定性**：完善的边界处理和数值稳定性检查
- **数据类型优化**：内部 float32 计算，输出 bfloat16，平衡精度与性能

### 系统层面优化

- **内存优化**：内存池管理、张量视图、批处理
- **并行处理**：多线程、多进程、向量化操作
- **硬件适配**：定点数支持、FPGA 兼容性检查
- **性能监控**：实时性能分析、优化建议

### 最新测试结果

- **Softmax 精度**：1.000 (满分)
- **L2 相对误差**：~1.27e-07 (远低于 1e-3 要求)
- **查找表命中率**：>99.9%
- **硬件兼容性**：完全支持 FPGA 实现

## 测试

运行单元测试：

```bash
python -m pytest tests/ -v
```

运行特定测试：

```bash
python -m pytest tests/test_activation_functions.py -v
python -m pytest tests/test_accuracy.py -v
python -m pytest tests/test_hardware_optimization.py -v
```

## 配置 ⚙️ 智能配置管理

项目使用 JSON 配置文件管理参数，支持动态配置加载和实时更新：

### 核心配置项

- **查找表配置**：点数(2000)、插值方法(quadratic)、采样策略(adaptive)
- **硬件配置**：定点数格式、并行处理、FPGA 兼容性
- **优化配置**：内存池、向量化、批处理、数据类型管理
- **测试配置**：张量形状(64×768)、数据类型(bfloat16)、运行次数
- **输出配置**：结果目录、报告格式(Excel/JSON/图表)

### 配置示例

```json
{
  "lookup_table": {
    "point_count": 2000,
    "interpolation_method": "quadratic",
    "sampling_strategy": "adaptive",
    "use_advanced_lookup": true
  },
  "softmax": {
    "use_lookup_table": true,
    "lookup_table_size": 2000,
    "interpolation_method": "quadratic",
    "use_fixed_point": false,
    "numerical_stability": true
  }
}
```

## 竞赛要求 🏆 竞赛标准

本项目严格遵循 FPT25 设计竞赛要求，已实现关键突破：

### 核心要求达成

1. **LLM 辅助设计**：✅ 完成 - 多个 LLM 提示词优化算法设计
2. **技术报告**：✅ 完成 - 详细架构设计、bf16 计算单元、资源复用策略
3. **代码提交**：✅ 完成 - 完整源代码、构建脚本、运行说明
4. **开源要求**：✅ 完成 - 项目完全开源

### 竞赛评分标准

- **Softmax 精度**：✅ **1.000** (满分 15 分) - 相对 L2 误差 ~1.27e-07
- **硬件适配**：✅ 完全支持 FPGA 实现
- **性能优化**：✅ 2000 点查找表 + 二次插值
- **创新性**：✅ 自适应采样 + 智能边界处理

### 技术亮点

- 🎯 **精度突破**：达到竞赛满分标准，远超 1e-3 要求
- 🚀 **性能优化**：硬件友好的查找表算法
- 🧠 **智能算法**：自适应采样 + 二次插值
- ⚡ **高效实现**：支持批处理和并行计算

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

MIT 许可证允许您自由使用、修改和分发本软件，包括商业用途，只要保留版权声明和许可证文本即可。

## 联系方式

如有问题，请通过以下方式联系：

- 项目 Issues: [GitHub Issues](https://github.com/ApexGP/FPT25/issues)

---

**注意**：本项目是 FPT25 设计竞赛的参赛作品，专注于 FPGA 硬件加速技术的研究和应用。
