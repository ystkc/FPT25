"""
FPT25 激活函数 FPGA 硬件加速项目主程序
支持命令行接口，提供测试、基准测试、优化等功能
"""
import traceback
import argparse
import sys
import time
import torch
from typing import Dict, Any
from pathlib import Path

from core.activation_functions.base_activation import ActivationScanner
from core.base.logs import FPT25Logger, setup_logging, get_logger
from core.base.constants import ACTIVATION_FUNCTIONS, TENSOR_SHAPE, BATCH_SIZE
from core.activation_functions import ActivationFunctionManager, get_activation_manager, create_activation_function
from evaluation import get_accuracy_evaluator, get_benchmark_runner, get_test_suite
from config import get_config_manager, get_config
from utils import get_result_manager, generate_excel_report, generate_benchmark_excel_report, plot_performance_trends, plot_benchmark_results, save_benchmark_json_report

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 全局模块缓存
modules = {
    'setup_logging': setup_logging,
    'get_logger': get_logger,
    'ACTIVATION_FUNCTIONS': ACTIVATION_FUNCTIONS,
    'TENSOR_SHAPE': TENSOR_SHAPE,
    'BATCH_SIZE': BATCH_SIZE,
    'get_activation_manager': get_activation_manager,
    'create_activation_function': create_activation_function,
    'get_accuracy_evaluator': get_accuracy_evaluator,
    'get_benchmark_runner': get_benchmark_runner,
    'get_test_suite': get_test_suite,
    'get_config_manager': get_config_manager,
    'get_config': get_config,
    'get_result_manager': get_result_manager,
    'generate_excel_report': generate_excel_report,
    'generate_benchmark_excel_report': generate_benchmark_excel_report,
    'plot_performance_trends': plot_performance_trends,
    'plot_benchmark_results': plot_benchmark_results,
    'save_benchmark_json_report': save_benchmark_json_report
}

class FPT25Main:
    """FPT25 主程序类"""
    
    def __init__(self):
        self.logger: FPT25Logger = modules['get_logger']()
        self.config_manager = modules['get_config_manager']()
        self.config = modules['get_config']()
        self.activation_manager: ActivationFunctionManager = modules['get_activation_manager']()
        self.accuracy_evaluator = modules['get_accuracy_evaluator']()
        self.benchmark_runner = modules['get_benchmark_runner']()
        self.test_suite = modules['get_test_suite']()
        self.result_manager = modules['get_result_manager']()
    
    def run_test(self, function_name: str, **kwargs) -> Dict[str, Any]:
        """运行测试"""
        self.logger.info(f"开始测试激活函数: {function_name}")
        
        try:
            # 创建激活函数
            activation_function = self.activation_manager.create_function(function_name)
            
            # 创建测试数据
            input_tensor = torch.randn(modules['TENSOR_SHAPE'], dtype=getattr(torch, self.config.test.dtype))
            
            # 运行前向传播
            start_time = time.perf_counter()
            output = activation_function.forward(input_tensor)
            end_time = time.perf_counter()
            
            execution_time = end_time - start_time
            
            # 记录结果
            result = {
                'function_name': function_name,
                'input_shape': modules['TENSOR_SHAPE'],
                'output_shape': output.shape,
                'execution_time': execution_time,
                'success': True,
                'output_dtype': str(output.dtype)
            }
            
            self.logger.info(f"测试完成: {function_name}, 执行时间: {execution_time:.6f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"测试失败: {function_name}, 错误: {e}")
            import traceback
            traceback.print_exc()
            return {
                'function_name': function_name,
                'success': False,
                'error': str(e)
            }
    
    def run_benchmark(self, function_name: str, **kwargs) -> Dict[str, Any]:
        """运行基准测试"""
        self.logger.info(f"开始基准测试: {function_name}")
        
        try:
            # 创建激活函数
            activation_function = self.activation_manager.create_function(function_name)
            
            # 运行基准测试
            benchmark_result = activation_function.benchmark()
            
            # 添加success字段
            benchmark_result['success'] = True
            benchmark_result['function_name'] = function_name
            
            # 生成报告
            results = {function_name: benchmark_result}
            self._generate_summary_report(results, 'benchmark')
            
            self.logger.info(f"基准测试完成: {function_name}")
            return benchmark_result
            
        except Exception as e:
            self.logger.error(f"基准测试失败: {function_name}, 错误: {e}")
            return {
                'function_name': function_name,
                'success': False,
                'error': str(e)
            }
    
    def run_accuracy_test(self, function_name: str, **kwargs) -> Dict[str, Any]:
        """运行精度测试"""
        self.logger.info(f"开始精度测试: {function_name}")
        
        try:
            # 创建激活函数
            activation_function = self.activation_manager.create_function(function_name)
            
            # 创建测试数据
            input_tensor = torch.randn(modules['TENSOR_SHAPE'], dtype=getattr(torch, self.config.test.dtype))
            
            # 运行激活函数
            fpga_output = activation_function.forward(input_tensor)
            
            # 生成参考结果
            reference_generator = self.accuracy_evaluator.reference_generator
            if function_name == 'softmax':
                reference_output = reference_generator.generate_softmax_reference(input_tensor)
            elif function_name == 'layer_norm':
                reference_output = reference_generator.generate_layer_norm_reference(input_tensor)
            elif function_name == 'rms_norm':
                reference_output = reference_generator.generate_rms_norm_reference(input_tensor)
            elif function_name == 'silu':
                reference_output = reference_generator.generate_silu_reference(input_tensor)
            elif function_name == 'gelu':
                reference_output = reference_generator.generate_gelu_reference(input_tensor)
            else:
                self.logger.warning(f"未找到 {function_name} 的参考实现")
                return {'function_name': function_name, 'success': False, 'error': 'No reference implementation'}
            
            # 评估精度
            accuracy_result = self.accuracy_evaluator.evaluate_activation_function(
                function_name, fpga_output, reference_output
            )
            
            self.logger.info(f"精度测试完成: {function_name}, 精度评分: {accuracy_result['metrics'].accuracy_score:.4f}")
            return accuracy_result
            
        except Exception as e:
            self.logger.error(f"精度测试失败: {function_name}, 错误: {e}")
            return {
                'function_name': function_name,
                'success': False,
                'error': str(e)
            }
    
    def run_all_functions(self, mode: str = 'test', **kwargs) -> Dict[str, Any]:
        """运行所有激活函数"""
        self.logger.info(f"开始运行所有激活函数: {mode}")
        
        results = {}
        
        for function_name in modules['ACTIVATION_FUNCTIONS']:
            self.logger.info(f"处理激活函数: {function_name}")
            
            if mode == 'test':
                result = self.run_test(function_name, **kwargs)
            elif mode == 'benchmark':
                result = self.run_benchmark(function_name, **kwargs)
            elif mode == 'accuracy':
                result = self.run_accuracy_test(function_name, **kwargs)
            elif mode == 'scan':
                result = self.run_scan(function_name, **kwargs)
            else:
                result = {'function_name': function_name, 'success': False, 'error': f'Unknown mode: {mode}'}
            
            results[function_name] = result
        
        # 生成报告
        if kwargs.get('generate_report', True):
            self._generate_summary_report(results, mode)
        
        return results
    
    def run_scan(self, function_name: str, **kwargs) -> Dict[str, Any]:
        """运行扫描"""
        self.logger.info(f"开始扫描: {function_name}")
        
        try:
            function = self.activation_manager.create_function(function_name)
            scanner = ActivationScanner(function)
            bit_lens = kwargs.get('bit_lens')
            scanner.optimize_lookup_bit_len(bit_lens)

            
        except Exception as e:
            self.logger.error(f"扫描失败: {function_name}, 错误: {e}")
            traceback.print_exc()
            return {
                'function_name': function_name,
                'success': False,
                'error': str(e)
            }
    
    def _generate_summary_report(self, results: Dict[str, Any], mode: str) -> None:
        """生成摘要报告"""
        successful_results = [r for r in results.values() if r.get('success', False)]
        failed_results = [r for r in results.values() if not r.get('success', False)]
        
        self.logger.info(f"=== {mode.upper()} 摘要报告 ===")
        self.logger.info(f"总函数数: {len(results)}")
        self.logger.info(f"成功: {len(successful_results)}")
        self.logger.info(f"失败: {len(failed_results)}")
        
        if failed_results:
            self.logger.warning("失败的函数:")
            for result in failed_results:
                self.logger.warning(f"  - {result['function_name']}: {result.get('error', 'Unknown error')}")
        
        # 生成详细报告文件
        try:
            import json
            import os
            modules = get_modules()
            
            # 生成报告文件
            if mode == 'benchmark':
                for function_name, result in results.items():
                    if result.get('success', False):
                        # 生成 Excel 报告
                        excel_file = f"results/{function_name}/data/benchmark_results.xlsx"
                        modules['generate_benchmark_excel_report'](result, excel_file)
                        
                        # 生成性能图表
                        chart_file = f"results/{function_name}/charts/performance_trends.png"
                        modules['plot_benchmark_results'](result, chart_file)
                        
                        # 生成 JSON 报告
                        json_file = f"results/{function_name}/reports/benchmark_results.json"
                        modules['save_benchmark_json_report'](result, json_file)
                        
                        self.logger.info(f"所有报告文件已生成完成: {function_name}")
            
        except Exception as e:
            self.logger.error(f"生成报告文件失败: {e}")
    
    
    
    def _convert_results_for_json(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """转换结果为JSON可序列化格式"""
        json_results = {}
        for key, result in results.items():
            if isinstance(result, dict):
                json_results[key] = {}
                for k, v in result.items():
                    if isinstance(v, (int, float, str, bool, type(None))):
                        json_results[key][k] = v
                    elif hasattr(v, '__dict__'):
                        # 如果是对象，转换为字典
                        json_results[key][k] = v.__dict__
                    else:
                        json_results[key][k] = str(v)
            else:
                json_results[key] = str(result)
        return json_results


def create_argument_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description='FPT25 激活函数 FPGA 硬件加速项目',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python main.py --mode test --function softmax
  python main.py --mode benchmark --function all
  python main.py --mode accuracy --function softmax --bit_lens [14]
  python main.py --mode scan --function softmax --interpolation linear --bit_lens [14, 15, 16]
        """
    )
    
    # 基本参数
    parser.add_argument('--mode', choices=['test', 'benchmark', 'accuracy', 'scan', 'all_functions'],
                       default='test', help='运行模式')
    parser.add_argument('--function', choices=modules['ACTIVATION_FUNCTIONS'] + ['all'],
                       default='softmax', help='激活函数名称')
    
    # 查找表参数
    parser.add_argument('--bit_lens', type=int, nargs='+', default=[14],
                       help='查找表位宽列表 (10-32)')
    parser.add_argument('--interpolation', choices=['direct', 'linear', 'quadratic'],
                       default='linear', help='插值方法')
    parser.add_argument('--sampling_strategy', choices=['uniform', 'adaptive', 'logarithmic', 'quadratic'],
                       default='uniform', help='采样策略')
    parser.add_argument('--use_advanced_lookup', action='store_true',
                       help='使用高级查找表')
    
    # 测试参数
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批处理大小')
    parser.add_argument('--tensor_shape', nargs=2, type=int, default=[64, 768],
                       help='张量形状 (高度 宽度)')
    parser.add_argument('--dtype', choices=['float32', 'bfloat16'],
                       default='bfloat16', help='数据类型')
    
    # 硬件参数
    parser.add_argument('--use_fixed_point', action='store_true',
                       help='使用定点数计算')
    parser.add_argument('--fixed_point_format', choices=['Q8_8', 'Q16_16', 'Q32_32', 'Q8_24'],
                       default='Q16_16', help='定点数格式')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='results',
                       help='输出目录')
    parser.add_argument('--config', type=str, default='config.json',
                       help='配置文件路径')
    parser.add_argument('--generate_report', action='store_true', default=True,
                       help='生成报告')
    
    # 日志参数
    parser.add_argument('--log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='日志级别')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='详细输出')
    
    return parser


def main():
    """主函数"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # 设置日志
    log_level = 'DEBUG' if args.verbose else args.log_level
    modules['setup_logging'](level=log_level)
    logger = modules['get_logger']()
    
    logger.info("FPT25 激活函数 FPGA 硬件加速项目启动")
    logger.info(f"运行模式: {args.mode}")
    logger.info(f"激活函数: {args.function}")
    
    # 创建主程序实例
    main_app = FPT25Main()
    
    # 准备参数
    kwargs = {
        'bit_lens': args.bit_lens,
        'interpolation': args.interpolation,
        'sampling_strategy': args.sampling_strategy,
        'use_advanced_lookup': args.use_advanced_lookup,
        'batch_size': args.batch_size,
        'tensor_shape': tuple(args.tensor_shape),
        'dtype': args.dtype,
        'use_fixed_point': args.use_fixed_point,
        'fixed_point_format': args.fixed_point_format,
        'output_dir': args.output_dir,
        'generate_report': args.generate_report
    }
    
    try:
        # 根据模式运行
        if args.mode == 'test':
            if args.function == 'all':
                results = main_app.run_all_functions('test', **kwargs)
            else:
                results = main_app.run_test(args.function, **kwargs)
        
        elif args.mode == 'benchmark':
            if args.function == 'all':
                results = main_app.run_all_functions('benchmark', **kwargs)
            else:
                results = main_app.run_benchmark(args.function, **kwargs)
        
        elif args.mode == 'accuracy':
            if args.function == 'all':
                results = main_app.run_all_functions('accuracy', **kwargs)
            else:
                results = main_app.run_accuracy_test(args.function, **kwargs)
        
        elif args.mode == 'scan':
            if args.function == 'all':
                results = {}
                for func_name in modules['ACTIVATION_FUNCTIONS']:
                    results[func_name] = main_app.run_scan(func_name, **kwargs)
            else:
                results = main_app.run_scan(args.function, **kwargs)
        
        elif args.mode == 'all_functions':
            results = main_app.run_all_functions('test', **kwargs)
        
        else:
            logger.error(f"未知的运行模式: {args.mode}")
            return 1
        
        logger.info("程序执行完成")
        return 0
        
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
        return 1
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
