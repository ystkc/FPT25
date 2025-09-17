"""
结果管理模块
提供测试结果的管理、存储和检索功能
"""

import json
import os
import pickle
import torch
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from core.base.constants import OUTPUT_DIRS, FILE_EXTENSIONS
from core.base.exceptions import FileOperationError
from core.base.logs import get_logger


@dataclass
class TestResult:
    """测试结果"""
    test_id: str
    function_name: str
    test_type: str  # 'accuracy', 'benchmark', 'optimization'
    timestamp: str
    input_shape: Tuple[int, int]
    batch_size: int
    dtype: str
    config: Dict[str, Any]
    metrics: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class ExperimentResult:
    """实验结果"""
    experiment_id: str
    experiment_name: str
    description: str
    timestamp: str
    config: Dict[str, Any]
    test_results: List[TestResult]
    summary: Dict[str, Any]
    success: bool


class ResultManager:
    """结果管理器"""
    
    def __init__(self, base_dir: str = "results"):
        self.base_dir = Path(base_dir)
        self.logger = get_logger()
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """确保目录存在"""
        # 为每个激活函数创建子目录结构
        activation_functions = ["softmax", "layer_norm", "rms_norm", "silu", "gelu", "add", "multiply"]
        directories = [self.base_dir]
        
        for func in activation_functions:
            directories.extend([
                self.base_dir / func,
                self.base_dir / func / "charts",
                self.base_dir / func / "data", 
                self.base_dir / func / "reports"
            ])
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def save_test_result(self, result: TestResult, 
                        save_tensors: bool = True) -> str:
        """保存测试结果"""
        try:
            # 创建函数专用目录
            function_dir = self.base_dir / result.function_name
            function_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存结果数据
            result_file = function_dir / f"{result.test_id}_{result.test_type}.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(result), f, indent=2, ensure_ascii=False)
            
            # 保存张量（如果需要）
            if save_tensors and 'tensors' in result.metrics:
                tensor_file = function_dir / f"{result.test_id}_tensors.pt"
                torch.save(result.metrics['tensors'], tensor_file)
            
            self.logger.info(f"测试结果已保存: {result_file}")
            return str(result_file)
            
        except Exception as e:
            raise FileOperationError(f"保存测试结果失败: {e}")
    
    def load_test_result(self, filepath: str) -> TestResult:
        """加载测试结果"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 转换元组
            if 'input_shape' in data and isinstance(data['input_shape'], list):
                data['input_shape'] = tuple(data['input_shape'])
            
            return TestResult(**data)
            
        except Exception as e:
            raise FileOperationError(f"加载测试结果失败: {e}")
    
    def save_experiment_result(self, result: ExperimentResult) -> str:
        """保存实验结果"""
        try:
            # 在根目录下创建 experiments 目录
            experiments_dir = self.base_dir / "experiments"
            experiments_dir.mkdir(parents=True, exist_ok=True)
            experiment_dir = experiments_dir / result.experiment_id
            experiment_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存实验数据
            experiment_file = experiment_dir / "experiment.json"
            with open(experiment_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(result), f, indent=2, ensure_ascii=False)
            
            # 保存各个测试结果
            for test_result in result.test_results:
                self.save_test_result(test_result, save_tensors=False)
            
            self.logger.info(f"实验结果已保存: {experiment_file}")
            return str(experiment_file)
            
        except Exception as e:
            raise FileOperationError(f"保存实验结果失败: {e}")
    
    def load_experiment_result(self, filepath: str) -> ExperimentResult:
        """加载实验结果"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 转换测试结果
            test_results = []
            for test_data in data.get('test_results', []):
                if 'input_shape' in test_data and isinstance(test_data['input_shape'], list):
                    test_data['input_shape'] = tuple(test_data['input_shape'])
                test_results.append(TestResult(**test_data))
            
            data['test_results'] = test_results
            return ExperimentResult(**data)
            
        except Exception as e:
            raise FileOperationError(f"加载实验结果失败: {e}")
    
    def list_test_results(self, function_name: Optional[str] = None,
                         test_type: Optional[str] = None) -> List[TestResult]:
        """列出测试结果"""
        results = []
        
        if function_name:
            search_dirs = [self.base_dir / function_name]
        else:
            search_dirs = [self.base_dir / f for f in ['softmax', 'layer_norm', 'rms_norm', 'silu', 'gelu', 'add', 'multiply']]
        
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            
            for file_path in search_dir.glob("*.json"):
                if test_type and test_type not in file_path.name:
                    continue
                
                try:
                    result = self.load_test_result(str(file_path))
                    results.append(result)
                except Exception as e:
                    self.logger.warning(f"加载测试结果失败: {file_path}, 错误: {e}")
        
        return results
    
    def list_experiment_results(self) -> List[ExperimentResult]:
        """列出实验结果"""
        results = []
        experiments_dir = self.base_dir / "experiments"
        
        if not experiments_dir.exists():
            return results
        
        for experiment_dir in experiments_dir.iterdir():
            if experiment_dir.is_dir():
                experiment_file = experiment_dir / "experiment.json"
                if experiment_file.exists():
                    try:
                        result = self.load_experiment_result(str(experiment_file))
                        results.append(result)
                    except Exception as e:
                        self.logger.warning(f"加载实验结果失败: {experiment_file}, 错误: {e}")
        
        return results
    
    def get_result_summary(self, function_name: Optional[str] = None) -> Dict[str, Any]:
        """获取结果摘要"""
        test_results = self.list_test_results(function_name)
        
        if not test_results:
            return {'total_results': 0}
        
        # 按函数分组
        function_groups = {}
        for result in test_results:
            if result.function_name not in function_groups:
                function_groups[result.function_name] = []
            function_groups[result.function_name].append(result)
        
        summary = {
            'total_results': len(test_results),
            'function_summaries': {}
        }
        
        for func_name, results in function_groups.items():
            successful_results = [r for r in results if r.success]
            
            summary['function_summaries'][func_name] = {
                'total_tests': len(results),
                'successful_tests': len(successful_results),
                'success_rate': len(successful_results) / len(results) if results else 0,
                'test_types': list(set(r.test_type for r in results)),
                'latest_test': max(results, key=lambda r: r.timestamp).timestamp if results else None
            }
        
        return summary
    
    def cleanup_old_results(self, days: int = 30) -> int:
        """清理旧结果"""
        cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)
        cleaned_count = 0
        
        for result in self.list_test_results():
            result_date = datetime.fromisoformat(result.timestamp).timestamp()
            if result_date < cutoff_date:
                # 删除结果文件
                function_dir = self.base_dir / result.function_name
                result_file = function_dir / f"{result.test_id}_{result.test_type}.json"
                if result_file.exists():
                    result_file.unlink()
                    cleaned_count += 1
        
        self.logger.info(f"清理了 {cleaned_count} 个旧结果文件")
        return cleaned_count


class ResultAnalyzer:
    """结果分析器"""
    
    def __init__(self, result_manager: ResultManager):
        self.result_manager = result_manager
        self.logger = get_logger()
    
    def analyze_performance_trends(self, function_name: str) -> Dict[str, Any]:
        """分析性能趋势"""
        results = self.result_manager.list_test_results(function_name, 'benchmark')
        
        if not results:
            return {'error': '没有找到基准测试结果'}
        
        # 按时间排序
        results.sort(key=lambda r: r.timestamp)
        
        # 提取性能指标
        timestamps = []
        execution_times = []
        throughputs = []
        
        for result in results:
            if result.success and 'execution_time' in result.metrics:
                timestamps.append(result.timestamp)
                execution_times.append(result.metrics['execution_time'])
                if 'throughput' in result.metrics:
                    throughputs.append(result.metrics['throughput'])
        
        return {
            'function_name': function_name,
            'data_points': len(timestamps),
            'timestamps': timestamps,
            'execution_times': execution_times,
            'throughputs': throughputs,
            'trend_analysis': self._calculate_trends(execution_times)
        }
    
    def _calculate_trends(self, values: List[float]) -> Dict[str, Any]:
        """计算趋势"""
        if len(values) < 2:
            return {'trend': 'insufficient_data'}
        
        # 简单线性趋势分析
        n = len(values)
        x = list(range(n))
        
        # 计算斜率
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        slope = numerator / denominator if denominator != 0 else 0
        
        if slope > 0.01:
            trend = 'increasing'
        elif slope < -0.01:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'slope': slope,
            'min_value': min(values),
            'max_value': max(values),
            'avg_value': sum(values) / len(values)
        }
    
    def compare_configurations(self, function_name: str) -> Dict[str, Any]:
        """比较不同配置的性能"""
        results = self.result_manager.list_test_results(function_name)
        
        if not results:
            return {'error': '没有找到测试结果'}
        
        # 按配置分组
        config_groups = {}
        for result in results:
            config_key = json.dumps(result.config, sort_keys=True)
            if config_key not in config_groups:
                config_groups[config_key] = []
            config_groups[config_key].append(result)
        
        comparison = {
            'function_name': function_name,
            'configurations': len(config_groups),
            'config_analysis': {}
        }
        
        for config_key, config_results in config_groups.items():
            successful_results = [r for r in config_results if r.success]
            
            if successful_results:
                avg_execution_time = sum(r.metrics.get('execution_time', 0) for r in successful_results) / len(successful_results)
                avg_accuracy = sum(r.metrics.get('accuracy_score', 0) for r in successful_results) / len(successful_results)
                
                comparison['config_analysis'][config_key] = {
                    'total_tests': len(config_results),
                    'successful_tests': len(successful_results),
                    'avg_execution_time': avg_execution_time,
                    'avg_accuracy': avg_accuracy,
                    'config': successful_results[0].config
                }
        
        return comparison
    
    def generate_performance_report(self, function_name: str) -> Dict[str, Any]:
        """生成性能报告"""
        results = self.result_manager.list_test_results(function_name)
        
        if not results:
            return {'error': '没有找到测试结果'}
        
        # 分析基准测试结果
        benchmark_results = [r for r in results if r.test_type == 'benchmark' and r.success]
        accuracy_results = [r for r in results if r.test_type == 'accuracy' and r.success]
        
        report = {
            'function_name': function_name,
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_tests': len(results),
                'benchmark_tests': len(benchmark_results),
                'accuracy_tests': len(accuracy_results),
                'success_rate': len([r for r in results if r.success]) / len(results)
            }
        }
        
        # 性能分析
        if benchmark_results:
            execution_times = [r.metrics.get('execution_time', 0) for r in benchmark_results]
            report['performance'] = {
                'avg_execution_time': sum(execution_times) / len(execution_times),
                'min_execution_time': min(execution_times),
                'max_execution_time': max(execution_times),
                'std_execution_time': (sum((x - sum(execution_times)/len(execution_times))**2 for x in execution_times) / len(execution_times))**0.5
            }
        
        # 精度分析
        if accuracy_results:
            accuracy_scores = [r.metrics.get('accuracy_score', 0) for r in accuracy_results]
            report['accuracy'] = {
                'avg_accuracy_score': sum(accuracy_scores) / len(accuracy_scores),
                'min_accuracy_score': min(accuracy_scores),
                'max_accuracy_score': max(accuracy_scores)
            }
        
        return report


# 全局结果管理器
_result_manager: Optional[ResultManager] = None
_result_analyzer: Optional[ResultAnalyzer] = None


def get_result_manager(base_dir: str = "results") -> ResultManager:
    """获取全局结果管理器"""
    global _result_manager
    if _result_manager is None:
        _result_manager = ResultManager(base_dir)
    return _result_manager


def get_result_analyzer() -> ResultAnalyzer:
    """获取结果分析器"""
    global _result_analyzer
    if _result_analyzer is None:
        _result_analyzer = ResultAnalyzer(get_result_manager())
    return _result_analyzer


def save_test_result(result: TestResult, save_tensors: bool = True) -> str:
    """保存测试结果便捷函数"""
    manager = get_result_manager()
    return manager.save_test_result(result, save_tensors)


def load_test_result(filepath: str) -> TestResult:
    """加载测试结果便捷函数"""
    manager = get_result_manager()
    return manager.load_test_result(filepath)


def save_benchmark_json_report(result_data: Dict[str, Any], output_file: str) -> bool:
    """保存基准测试 JSON 报告便捷函数"""
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 完全覆写文件 - 删除现有文件
        if os.path.exists(output_file):
            os.remove(output_file)
        
        # 保存 JSON 报告
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        logger = get_logger()
        logger.info(f"基准测试 JSON 报告已保存: {output_file}")
        return True
        
    except Exception as e:
        logger = get_logger()
        logger.error(f"保存基准测试 JSON 报告失败: {e}")
        return False