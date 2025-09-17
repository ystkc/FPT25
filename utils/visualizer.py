"""
可视化模块
提供测试结果的可视化功能
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List, Any, Optional
from pathlib import Path

from .results import get_result_manager, get_result_analyzer
from core.base.logs import get_logger

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ResultVisualizer:
    """结果可视化器"""
    
    def __init__(self):
        self.logger = get_logger()
        self.result_manager = get_result_manager()
        self.result_analyzer = get_result_analyzer()
    
    def plot_performance_trends(self, function_name: str, output_file: str) -> bool:
        """绘制性能趋势图"""
        try:
            analysis = self.result_analyzer.analyze_performance_trends(function_name)
            
            if 'error' in analysis:
                self.logger.warning(f"无法分析性能趋势: {analysis['error']}")
                return False
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # 执行时间趋势
            ax1.plot(analysis['execution_times'], 'b-', marker='o')
            ax1.set_title(f'{function_name} - 执行时间趋势')
            ax1.set_xlabel('测试次数')
            ax1.set_ylabel('执行时间 (秒)')
            ax1.grid(True)
            
            # 吞吐量趋势
            if analysis['throughputs']:
                ax2.plot(analysis['throughputs'], 'g-', marker='s')
                ax2.set_title(f'{function_name} - 吞吐量趋势')
                ax2.set_xlabel('测试次数')
                ax2.set_ylabel('吞吐量 (ops/s)')
                ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"性能趋势图已保存: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"绘制性能趋势图失败: {e}")
            return False
    
    def plot_benchmark_results(self, result_data: Dict[str, Any], output_file: str) -> bool:
        """绘制基准测试结果图"""
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # 完全覆写文件 - 删除现有文件
            if os.path.exists(output_file):
                os.remove(output_file)
            
            # 处理基准测试结果数据
            if 'results' not in result_data or not isinstance(result_data['results'], list):
                self.logger.warning("没有找到有效的基准测试结果数据")
                return False
            
            results = result_data['results']
            if not results:
                self.logger.warning("基准测试结果数据为空")
                return False
            
            # 提取数据
            shapes = []
            avg_times = []
            throughputs = []
            accuracy_scores = []
            
            for result in results:
                if isinstance(result, dict):
                    shapes.append(str(result.get('shape', '')))
                    avg_times.append(result.get('average_time', 0))
                    throughputs.append(result.get('throughput', 0))
                    accuracy_scores.append(result.get('accuracy_score', 0))
            
            if not shapes:
                self.logger.warning("没有有效的测试结果数据")
                return False
            
            # 创建图表
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'基准测试结果 - {result_data.get("function_name", "Unknown")}', fontsize=16)
            
            # 子图1：执行时间
            ax1.bar(range(len(shapes)), avg_times, color='skyblue', alpha=0.7)
            ax1.set_title('平均执行时间')
            ax1.set_xlabel('输入形状')
            ax1.set_ylabel('时间 (秒)')
            ax1.set_xticks(range(len(shapes)))
            ax1.set_xticklabels(shapes, rotation=45)
            
            # 子图2：吞吐量
            ax2.bar(range(len(shapes)), throughputs, color='lightgreen', alpha=0.7)
            ax2.set_title('吞吐量')
            ax2.set_xlabel('输入形状')
            ax2.set_ylabel('操作/秒')
            ax2.set_xticks(range(len(shapes)))
            ax2.set_xticklabels(shapes, rotation=45)
            
            # 子图3：精度分数
            ax3.bar(range(len(shapes)), accuracy_scores, color='orange', alpha=0.7)
            ax3.set_title('精度分数')
            ax3.set_xlabel('输入形状')
            ax3.set_ylabel('精度分数')
            ax3.set_xticks(range(len(shapes)))
            ax3.set_xticklabels(shapes, rotation=45)
            ax3.set_ylim(0, 1.1)
            
            # 子图4：性能对比（时间 vs 精度）
            scatter = ax4.scatter(avg_times, accuracy_scores, c=throughputs, 
                                 cmap='viridis', s=100, alpha=0.7)
            ax4.set_title('性能对比 (时间 vs 精度)')
            ax4.set_xlabel('平均执行时间 (秒)')
            ax4.set_ylabel('精度分数')
            ax4.grid(True, alpha=0.3)
            
            # 添加颜色条
            cbar = plt.colorbar(scatter, ax=ax4)
            cbar.set_label('吞吐量 (ops/s)')
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图表
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"基准测试图表已生成: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"绘制基准测试结果图失败: {e}")
            return False


def plot_performance_trends(function_name: str, output_file: str) -> bool:
    """绘制性能趋势图便捷函数"""
    visualizer = ResultVisualizer()
    return visualizer.plot_performance_trends(function_name, output_file)


def plot_benchmark_results(result_data: Dict[str, Any], output_file: str) -> bool:
    """绘制基准测试结果图便捷函数"""
    visualizer = ResultVisualizer()
    return visualizer.plot_benchmark_results(result_data, output_file)
