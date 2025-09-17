"""
Excel 报告生成模块
提供 Excel 格式的测试报告生成功能
"""

import pandas as pd
import os
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

from .results import TestResult, ExperimentResult, get_result_manager
from core.base.logs import get_logger


class ExcelReportGenerator:
    """Excel 报告生成器"""
    
    def __init__(self):
        self.logger = get_logger()
        self.result_manager = get_result_manager()
    
    def generate_test_report(self, function_name: str, output_file: str) -> bool:
        """生成测试报告"""
        try:
            results = self.result_manager.list_test_results(function_name)
            
            if not results:
                self.logger.warning(f"没有找到 {function_name} 的测试结果")
                return False
            
            # 创建 DataFrame
            data = []
            for result in results:
                data.append({
                    'Test ID': result.test_id,
                    'Test Type': result.test_type,
                    'Timestamp': result.timestamp,
                    'Input Shape': str(result.input_shape),
                    'Batch Size': result.batch_size,
                    'Data Type': result.dtype,
                    'Success': result.success,
                    'Error Message': result.error_message or '',
                    **result.metrics
                })
            
            df = pd.DataFrame(data)
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # 保存到 Excel
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Test Results', index=False)
            
            self.logger.info(f"Excel 报告已生成: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"生成 Excel 报告失败: {e}")
            return False
    
    def generate_benchmark_report(self, result_data: Dict[str, Any], output_file: str) -> bool:
        """生成基准测试报告"""
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # 创建基准测试数据
            benchmark_data = []
            
            # 处理基准测试结果数据
            if 'results' in result_data and isinstance(result_data['results'], list):
                # 这是从激活函数的benchmark方法返回的数据结构
                for i, result in enumerate(result_data['results']):
                    if isinstance(result, dict):
                        benchmark_data.append({
                            'Test ID': i + 1,
                            'Function Name': result_data.get('function_name', 'Unknown'),
                            'Test Type': 'Benchmark',
                            'Input Shape': str(result.get('shape', '')),
                            'Average Time (s)': result.get('average_time', 0),
                            'Min Time (s)': result.get('min_time', 0),
                            'Max Time (s)': result.get('max_time', 0),
                            'Std Time (s)': result.get('std_time', 0),
                            'Throughput (ops/s)': result.get('throughput', 0),
                            'Accuracy Score': result.get('accuracy_score', 0),
                            'L2 Error': result.get('l2_error', 0)
                        })
            
            # 如果没有results数据，尝试其他格式
            if not benchmark_data:
                # 添加基本测试信息
                benchmark_data.append({
                    'Test ID': 1,
                    'Function Name': result_data.get('function_name', 'Unknown'),
                    'Test Type': 'Benchmark',
                    'Input Shape': str(result_data.get('input_shape', '')),
                    'Average Time (s)': result_data.get('average_time', 0),
                    'Min Time (s)': result_data.get('min_time', 0),
                    'Max Time (s)': result_data.get('max_time', 0),
                    'Std Time (s)': result_data.get('std_time', 0),
                    'Throughput (ops/s)': result_data.get('throughput', 0),
                    'Accuracy Score': result_data.get('accuracy_score', 0),
                    'L2 Error': result_data.get('l2_error', 0)
                })
            
            # 创建 DataFrame - 只包含基准测试结果
            df = pd.DataFrame(benchmark_data)
            
            # 完全覆写文件 - 删除现有文件并创建新文件
            if os.path.exists(output_file):
                try:
                    os.remove(output_file)
                except PermissionError:
                    # 如果文件被占用，等待一下再试
                    import time
                    time.sleep(0.5)
                    try:
                        os.remove(output_file)
                    except PermissionError:
                        # 如果还是被占用，使用临时文件名
                        temp_file = output_file + '.tmp'
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                        output_file = temp_file
            
            # 保存到 Excel - 使用 mode='w' 确保完全覆写
            with pd.ExcelWriter(output_file, engine='openpyxl', mode='w') as writer:
                # 基准测试结果
                df.to_excel(writer, sheet_name='Benchmark Results', index=False)
                
                # 配置信息单独放在一个工作表中
                if 'config' in result_data:
                    config_data = []
                    config = result_data['config']
                    for key, value in config.items():
                        config_data.append({
                            'Configuration Key': key,
                            'Configuration Value': str(value)
                        })
                    
                    config_df = pd.DataFrame(config_data)
                    config_df.to_excel(writer, sheet_name='Configuration', index=False)
            
            self.logger.info(f"基准测试 Excel 报告已生成: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"生成基准测试 Excel 报告失败: {e}")
            return False


def generate_excel_report(function_name: str, output_file: str) -> bool:
    """生成 Excel 报告便捷函数"""
    generator = ExcelReportGenerator()
    return generator.generate_test_report(function_name, output_file)


def generate_benchmark_excel_report(result_data: Dict[str, Any], output_file: str) -> bool:
    """生成基准测试 Excel 报告便捷函数"""
    generator = ExcelReportGenerator()
    return generator.generate_benchmark_report(result_data, output_file)
