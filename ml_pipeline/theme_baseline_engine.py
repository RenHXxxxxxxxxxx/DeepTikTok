# -- coding: utf-8 --
"""
theme_baseline_engine.py
# *主题基准统计引擎 - 纯统计学计算模块，不依赖机器学习模型*
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 全局配置字典
GLOBAL_CONFIG = {
    # 最小有效样本量阈值
    'MIN_SAMPLE_SIZE': 5,
    # 默认全网平均基准值（当样本不足时使用）
    'DEFAULT_BASELINE': {
        'mean': 10000.0,
        'std': 25000.0,
        'p25': 500.0,
        'p50': 2000.0,
        'p75': 8000.0,
        'max': 100000.0,
        'sample_count': 0,
        'warning': '样本量不足，使用全网平均基准值'
    }
}


def calculate_optimal_publishing_times(video_data_list: List[Dict[str, Any]], top_n: int = 3) -> List[str]:
    """
    # *[重构] 修复审计3.2: 基于加权KDE密度估计的时间推荐引擎*
    """
    hours = []
    weights = []
    for v in video_data_list:
        try:
            h = None
            if v.get('create_time'):
                h = pd.to_datetime(v.get('create_time')).hour
            elif v.get('publish_hour') is not None:
                h = float(v.get('publish_hour'))
            digg = float(v.get('digg_count', 0))
            if h is not None and not pd.isna(h) and digg > 0:
                hours.append(h)
                weights.append(digg)
        except Exception:
            continue
            
    if not hours:
        return ["18:00", "19:00", "20:00"]
        
    grid = np.arange(0, 24)
    density = np.zeros(24)
    bandwidth = 1.5
    
    for h, w in zip(hours, weights):
        # 环形距离计算
        diff = np.minimum((grid - h) % 24, (h - grid) % 24)
        density += w * np.exp(-0.5 * (diff / bandwidth)**2)
        
    top_indices = density.argsort()[::-1]
    
    results = []
    for idx in top_indices:
        # 确保时间窗口至少间隔2小时
        if all(min((idx - r) % 24, (r - idx) % 24) >= 2 for r in results):
            results.append(idx)
        if len(results) >= top_n:
            break
            
    if len(results) < top_n:
        for idx in top_indices:
            if idx not in results:
                results.append(idx)
            if len(results) >= top_n:
                break
                
    return [f"{int(h):02d}:00" for h in results]


class ThemeBaselineCalculator:
    """
    # *主题基准统计计算器*
    # *用于计算视频点赞数的统计基准值*
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        # *初始化计算器*
        
        # *Args:*
        #     *config: 可选的配置覆盖字典*
        """
        self.config = {**GLOBAL_CONFIG, **(config or {})}
        self.min_sample_size = self.config['MIN_SAMPLE_SIZE']
        self.default_baseline = self.config['DEFAULT_BASELINE']

    def _clean_data(self, video_data_list: List[Dict[str, Any]]) -> pd.Series:
        """
        # *数据清洗：过滤掉无效的 digg_count 值*
        
        # *Args:*
        #     *video_data_list: 包含视频数据的字典列表*
        
        # *Returns:*
        #     *清洗后的 digg_count Series*
        """
        try:
            # 提取 digg_count 字段
            digg_counts = []
            for item in video_data_list:
                digg_count = item.get('digg_count')
                # 过滤 None 和负数
                if digg_count is not None and isinstance(digg_count, (int, float)) and digg_count >= 0:
                    digg_counts.append(float(digg_count))
                else:
                    logger.debug(f"# *过滤无效数据: digg_count={digg_count}*")
            
            return pd.Series(digg_counts, dtype=np.float64)
        
        except Exception as e:
            logger.error(f"# *数据清洗过程中发生错误: {e}*")
            return pd.Series([], dtype=np.float64)

    def calculate(self, video_data_list: List[Dict[str, Any]], global_stats: Optional[Dict[str, Any]] = None, smoothing_factor: int = 10) -> Dict[str, Any]:
        """
        # *计算主题基准统计值 (支持贝叶斯平滑)*
        
        # *Args:*
        #     *video_data_list: 包含视频数据的字典列表*
        #     *global_stats: 全局统计数据 (Global Prior), 用于平滑小样本主题*
        #     *smoothing_factor: 平滑因子 K (默认10), 越大越倾向于全局均值*
        
        # *Returns:*
        #     *包含统计值的字典*
        """
        try:
            # Step 1: 数据清洗
            cleaned_data = self._clean_data(video_data_list)
            sample_count = len(cleaned_data)
            
            logger.info(f"# *清洗后有效样本量: {sample_count}*")
            
            # Step 2: 基础统计计算 (Local Stats)
            if sample_count > 0:
                local_stats = {
                    'mean': float(np.mean(cleaned_data)),
                    'std': float(np.std(cleaned_data, ddof=1)) if sample_count > 1 else 0.0,
                    'p25': float(np.percentile(cleaned_data, 25)),
                    'p50': float(np.percentile(cleaned_data, 50)),
                    'p75': float(np.percentile(cleaned_data, 75)),
                    'max': float(np.max(cleaned_data)),
                }
            else:
                local_stats = None

            # Step 3: 贝叶斯平滑逻辑 (Bayesian Smoothing)
            # 公式: Smoothed = (N  Local + K  Global) / (N + K)
            
            if global_stats and isinstance(global_stats, dict):
                # 如果有全局基准，进行平滑
                k = smoothing_factor
                n = sample_count
                
                # 辅助函数: 平滑单个指标
                def smooth_val(key):
                    local_val = local_stats.get(key, 0) if local_stats else 0
                    global_val = global_stats.get(key, 0)
                    
                    if n == 0: return global_val
                    
                    return (n * local_val + k * global_val) / (n + k)

                final_stats = {
                    'mean': smooth_val('mean'),
                    'std': smooth_val('std'), # 标准差平滑仅作参考
                    'p25': smooth_val('p25'),
                    'p50': smooth_val('p50'),
                    'p75': smooth_val('p75'),
                    'max': max(local_stats['max'], global_stats.get('max', 0)) if local_stats else global_stats.get('max', 0),
                    'sample_count': sample_count,
                    'warning': f"已应用贝叶斯平滑 (K={k}, N={n})" if n < 50 else None,
                    'optimal_publishing_times': calculate_optimal_publishing_times(video_data_list)
                }
                
                logger.info(f"# *贝叶斯平滑完成: N={n}, K={k} -> Mean: {final_stats['mean']:.2f}*")
                return final_stats
                
            else:
                # 无全局基准，退回原逻辑
                if sample_count < self.min_sample_size:
                    logger.warning(f"# *样本量不足且无全局基准，返回默认值*")
                    result = self.default_baseline.copy()
                    result['sample_count'] = sample_count
                    result['optimal_publishing_times'] = calculate_optimal_publishing_times(video_data_list)
                    return result
                
                return {
                    **local_stats,
                    'sample_count': sample_count,
                    'warning': None,
                    'optimal_publishing_times': calculate_optimal_publishing_times(video_data_list)
                }
        
        except Exception as e:
            logger.error(f"# *统计计算过程中发生错误: {e}*")
            result = self.default_baseline.copy()
            result['warning'] = f'计算异常: {str(e)}'
            return result


def calculate_theme_stats(video_data_list: List[Dict[str, Any]], 
                          config: Optional[Dict[str, Any]] = None,
                          global_stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    # *便捷函数：计算主题基准统计值 (支持传入全局基准)*
    """
    calculator = ThemeBaselineCalculator(config=config)
    return calculator.calculate(video_data_list, global_stats=global_stats)


if __name__ == "__main__":
    # 测试用例
    
    # 测试用例1: 正常数据
    test_data_normal = [
        {'digg_count': 1000, 'title': 'Video 1'},
        {'digg_count': 2500, 'title': 'Video 2'},
        {'digg_count': 800, 'title': 'Video 3'},
        {'digg_count': 15000, 'title': 'Video 4'},
        {'digg_count': 5000, 'title': 'Video 5'},
        {'digg_count': 3200, 'title': 'Video 6'},
        {'digg_count': 900, 'title': 'Video 7'},
    ]
    
    print("# *测试用例1: 正常数据*")
    result1 = calculate_theme_stats(test_data_normal)
    print(f"结果: {result1}")
    print()
    
    # 测试用例2: 包含无效数据
    test_data_with_invalid = [
        {'digg_count': 1000},
        {'digg_count': None},  # 无效
        {'digg_count': -500},  # 无效
        {'digg_count': 2000},
        {'digg_count': 'abc'},  # 无效
        {'digg_count': 3000},
        {'digg_count': 1500},
        {'digg_count': 4000},
    ]
    
    print("# *测试用例2: 包含无效数据*")
    result2 = calculate_theme_stats(test_data_with_invalid)
    print(f"结果: {result2}")
    print()
    
    # 测试用例3: 样本量不足
    test_data_insufficient = [
        {'digg_count': 1000},
        {'digg_count': 2000},
        {'digg_count': 3000},
    ]
    
    print("# *测试用例3: 样本量不足*")
    result3 = calculate_theme_stats(test_data_insufficient)
    print(f"结果: {result3}")
    print()
    
    # 测试用例4: 使用类接口和自定义配置
    custom_config = {
        'MIN_SAMPLE_SIZE': 3,
        'DEFAULT_BASELINE': {
            'mean': 5000.0,
            'std': 10000.0,
            'p25': 250.0,
            'p50': 1000.0,
            'p75': 4000.0,
            'max': 50000.0,
            'sample_count': 0,
            'warning': '自定义警告：样本不足'
        }
    }
    
    print("# *测试用例4: 使用类接口和自定义配置*")
    calculator = ThemeBaselineCalculator(config=custom_config)
    result4 = calculator.calculate(test_data_insufficient)
    print(f"结果: {result4}")
