"""
工具模块
提供结果管理、Excel 报告和可视化功能
"""

from .results import (
    TestResult,
    ExperimentResult,
    ResultManager,
    ResultAnalyzer,
    get_result_manager,
    get_result_analyzer,
    save_test_result,
    load_test_result,
    save_benchmark_json_report
)

from .excels import (
    ExcelReportGenerator,
    generate_excel_report,
    generate_benchmark_excel_report
)

from .visualizer import (
    ResultVisualizer,
    plot_performance_trends,
    plot_benchmark_results
)

__all__ = [
    # 结果管理
    'TestResult', 'ExperimentResult', 'ResultManager', 'ResultAnalyzer',
    'get_result_manager', 'get_result_analyzer', 'save_test_result', 'load_test_result',
    'save_benchmark_json_report',
    
    # Excel 报告
    'ExcelReportGenerator', 'generate_excel_report', 'generate_benchmark_excel_report',
    
    # 可视化
    'ResultVisualizer', 'plot_performance_trends', 'plot_benchmark_results'
]
