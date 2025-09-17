"""
评估模块
提供精度评估和性能基准测试功能
"""

from .accuracy import (
    AccuracyMetrics,
    AccuracyEvaluator,
    ReferenceGenerator,
    AccuracyTestSuite,
    get_accuracy_evaluator,
    get_reference_generator,
    get_test_suite,
    evaluate_accuracy
)

from .benchmark import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkRunner,
    PerformanceAnalyzer,
    BenchmarkReportGenerator,
    get_benchmark_runner,
    get_performance_analyzer,
    get_report_generator,
    run_benchmark
)

__all__ = [
    # 精度评估
    'AccuracyMetrics', 'AccuracyEvaluator', 'ReferenceGenerator',
    'AccuracyTestSuite', 'get_accuracy_evaluator', 'get_reference_generator',
    'get_test_suite', 'evaluate_accuracy',
    
    # 性能基准测试
    'BenchmarkConfig', 'BenchmarkResult', 'BenchmarkRunner',
    'PerformanceAnalyzer', 'BenchmarkReportGenerator',
    'get_benchmark_runner', 'get_performance_analyzer',
    'get_report_generator', 'run_benchmark'
]
