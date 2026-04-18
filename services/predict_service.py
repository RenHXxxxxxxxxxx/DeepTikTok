# -- coding: utf-8 --
"""
predict_service.py
预测服务模块 - 基于统一竞技场的高可用架构 (Refactored Version)
"""

import json
import threading
import traceback
import logging
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from django.conf import settings

# 配置日志
logger = logging.getLogger(__name__)

# 全局配置
GLOBAL_CONFIG = {
    'CHINESE_MAPPING': {
        '亮度': 'visual_brightness',
        '饱和度': 'visual_saturation',
        'BPM': 'audio_bpm',
        '剪频': 'cut_frequency',
        '剪辑频率': 'cut_frequency'
    },
    'FALLBACK_SUFFIX': " (Fallback Mode)",
    'MAX_DIGGS_LOG_SCALE': 12.0  # 用于质量分计算的上限基准 (~16万赞)
}

class DiggPredictionService:
    """
    视频点赞预测服务 (高可用 Singleton 实现)
    具备: 热更新(Hot-reload)、静默回滚(Silent Rollback)、特征自适应(Feature Defense)
    """

    _instance = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    logger.info(" [Singleton] 初始化 DiggPredictionService 实例")
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, '_initialized', False):
            return

        # 核心资产状态
        self.model = None
        self.scaler = None
        self.manifest = {}
        self._manifest_mtime = 0.0

        # 路径解析 (适配新的 artifacts 目录结构)
        self.base_dir = Path(__file__).resolve().parent.parent
        self.artifacts_dir = self.base_dir / 'artifacts'
        self.manifest_path = self.artifacts_dir / 'version_manifest.json'

        # 默认基础特征拓扑
        self.base_features = [
            'follower_count_log', 'publish_hour', 'duration_sec',
            'visual_brightness', 'visual_saturation', 'cut_frequency',
            'audio_bpm', 'avg_sentiment', 'theme_encoded', 'visual_impact', 'sensory_pace',
            'sentiment_intensity', 'audio_visual_energy', 'content_density'
        ]
        self.ordered_feature_names = []

        # 首次强制加载
        self._load_assets()
        
        self._initialized = True
        logger.info(" [Singleton] DiggPredictionService 资产装载完毕")

    # ==========================================
    # 核心热重载与资产加载逻辑
    # ==========================================
    def _check_hot_reload(self):
        """轻量级探测：通过比对文件修改时间实现零宕机热重载"""
        try:
            if self.manifest_path.exists():
                current_mtime = self.manifest_path.stat().st_mtime
                if current_mtime > self._manifest_mtime:
                    with self._lock:
                        # 双重检查防止并发穿透
                        if self.manifest_path.stat().st_mtime > self._manifest_mtime:
                            logger.info(" [Hot Reload] 监测到新模型版本发布，执行热更新...")
                            self._load_assets()
        except Exception as e:
            logger.error(f" 热重载检查异常: {e}")

    def _load_assets(self):
        """物理加载模型、伸缩器与拓扑清单"""
        try:
            if not self.manifest_path.exists():
                logger.error(f" 找不到版本清单: {self.manifest_path}")
                self._reset_state()
                return

            with open(self.manifest_path, 'r', encoding='utf-8') as f:
                self.manifest = json.load(f)

            # 适配 train_master_arena 的新键名 'current_version'
            version_id = self.manifest.get('current_version')
            if not version_id:
                logger.error(" 清单中缺失 current_version")
                self._reset_state()
                return

            model_path = self.artifacts_dir / f'model_{version_id}.pkl'
            scaler_path = self.artifacts_dir / f'scaler_{version_id}.pkl'

            if model_path.exists() and scaler_path.exists():
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                
                # 动态提取特征拓扑，优先使用 manifest 显式列序，其次回退到重要性字典键序
                feat_dict = self.manifest.get('feature_importances', {})
                self.ordered_feature_names = self.manifest.get('feature_names_in', list(feat_dict.keys()))
                if not self.ordered_feature_names:
                    self.ordered_feature_names = self.base_features
                    
                self._manifest_mtime = self.manifest_path.stat().st_mtime
                logger.info(f" 成功挂载冠军模型: {version_id} ({self.manifest.get('best_model')})")
            else:
                logger.error(f" 物理文件缺失: 期望 {model_path.name}")
                self._reset_state()

        except Exception as e:
            logger.error(f" 资产加载严重崩溃: {e}\n{traceback.format_exc()}")
            self._reset_state()

    def _reset_state(self):
        """容错保护：清空损坏状态"""
        self.model = None
        self.scaler = None
        self.manifest = {}

    # ==========================================
    # 数据工程管道 (特征抽取)
    # ==========================================
    def _safe_float(self, val, default=0.0):
        try:
            return default if pd.isna(val) or val == '' else float(val)
        except:
            return default

    def _build_features(self, video_data: dict, theme_baseline: dict = None) -> pd.DataFrame:
        """特征合成器：从原始 JSON 提取并构建符合拓扑维度的 DataFrame"""
        # 1. 键名中文化映射
        for cn_key, en_key in GLOBAL_CONFIG['CHINESE_MAPPING'].items():
            if cn_key in video_data:
                video_data[en_key] = video_data.pop(cn_key)

        # 2. 基础物理特征提取与对数平滑
        fol_raw = self._safe_float(video_data.get('follower_count', 10000))
        fol_log = np.log1p(fol_raw)

        hrr = self._safe_float(video_data.get('publish_hour'))
        dur = self._safe_float(video_data.get('duration_sec'))
        sent = self._safe_float(video_data.get('avg_sentiment', 0.5))
        vb = self._safe_float(video_data.get('visual_brightness', 128))
        vs = self._safe_float(video_data.get('visual_saturation', 100))
        cf = self._safe_float(video_data.get('cut_frequency', 0.5))
        bpm = self._safe_float(video_data.get('audio_bpm', 110))

        # 2.1 实时贝叶斯目标编码 (Bayesian Target Encoding)
        theme_count = float(theme_baseline.get('count', 10)) if theme_baseline else 10.0
        theme_mean = float(theme_baseline.get('mean', 5000)) if theme_baseline else 5000.0
        local_mean_log = np.log1p(theme_mean)
        global_mean_log = float(self.manifest.get('bayesian_global_mean', np.log1p(10000.0)))
        weight = 10.0
        theme_enc = (theme_count * local_mean_log + weight * global_mean_log) / (theme_count + weight)

        # 3. 构造占位 DataFrame
        X_final = pd.DataFrame(0.0, index=[0], columns=self.ordered_feature_names)

        # 4. 衍生特征计算与注入
        features_to_inject = {
            'follower_count_log': fol_log, 'publish_hour': hrr, 'duration_sec': dur,
            'avg_sentiment': sent, 'visual_brightness': vb, 'visual_saturation': vs,
            'cut_frequency': cf, 'audio_bpm': bpm, 'theme_encoded': theme_enc,
            'visual_impact': (vb * vs) / 1000.0,
            'sensory_pace': bpm * cf,
            'sentiment_intensity': abs(sent - 0.5) * 2,
            'audio_visual_energy': vb * bpm / 1000.0,
            'content_density': cf / (dur + 1) if dur >= 0 else 0
        }

        for col, val in features_to_inject.items():
            if col in self.ordered_feature_names:
                X_final.at[0, col] = val

        # 5. 分类特征（主题）独热映射
        current_theme = str(video_data.get('theme_label', 'Unknown') or 'Unknown')
        theme_mapped = False
        for col in self.ordered_feature_names:
            if col.startswith('theme_'):
                if current_theme == col.replace('theme_', ''):
                    X_final.at[0, col] = 1.0
                    theme_mapped = True

        if not theme_mapped and 'theme_Unknown' in self.ordered_feature_names:
            X_final.at[0, 'theme_Unknown'] = 1.0

        # 6. 均值插补兜底 (防止未覆盖特征传入0.0导致缩放器漂移)
        injected_keys = set(features_to_inject.keys())
        if self.scaler is not None and hasattr(self.scaler, 'mean_'):
            for idx, col in enumerate(self.ordered_feature_names):
                if idx < len(self.scaler.mean_):
                    if col not in injected_keys and not col.startswith('theme_'):
                        X_final.at[0, col] = self.scaler.mean_[idx]

        return X_final[self.ordered_feature_names] # 强制列序对齐

    # ==========================================
    # 预测与后处理逻辑
    # ==========================================
    def predict_digg_count(self, video_data: dict, theme_baseline: dict = None) -> dict:
        """主入口：执行预测及基准校正"""
        self._check_hot_reload()

        if self.model is None or self.scaler is None:
            return {"predicted_digg": 0, "quality_score": 0.0, "percentile_rank": "服务暂不可用", "baseline_ref": None}

        try:
            # 1. 抽取特征
            X_df = self._build_features(video_data, theme_baseline)
            
            # 2. 缩放
            X_scaled_ndarray = self.scaler.transform(X_df)
            X_scaled_df = pd.DataFrame(X_scaled_ndarray, columns=self.ordered_feature_names)

            # 3. 预测尝试与静默回滚 (The True Silent Rollback)
            is_fallback = False
            try:
                #下面是bug开关
                #raise ValueError("【混沌测试】模拟新模型遭遇未知特征崩溃！")
                raw_pred_log = self.model.predict(X_scaled_df)[0]
            except Exception as e:
                logger.warning(f" 主模型推理崩溃，触发降级回滚: {e}")
                raw_pred_log, is_fallback = self._execute_fallback(X_scaled_df)

            # 4. 指数还原与分数映射
            raw_pred_log = max(0.0, float(raw_pred_log))
            quality_score = min(max((raw_pred_log / GLOBAL_CONFIG['MAX_DIGGS_LOG_SCALE']) * 100.0, 0.0), 100.0)

            # 5. 基准校正
            final_pred, percentile_info = self._apply_baseline_correction(raw_pred_log, quality_score, theme_baseline)

            if is_fallback:
                percentile_info += GLOBAL_CONFIG['FALLBACK_SUFFIX']

            return {
                "predicted_digg": int(final_pred),
                "quality_score": round(quality_score, 1),
                "percentile_rank": percentile_info,
                "baseline_ref": theme_baseline
            }

        except Exception as e:
            logger.error(f" 预测防线彻底击穿，返回安全默认值: {e}\n{traceback.format_exc()}")
            return {"predicted_digg": 0, "quality_score": 0.0, "percentile_rank": "安全兜底模式", "baseline_ref": None}

    def _execute_fallback(self, X_scaled_df: pd.DataFrame):
        """提取的私有方法：处理物理回滚"""
        prev_version = self.manifest.get('previous_version')
        if not prev_version:
            raise RuntimeError("缺失 previous_version 历史指针，无法回滚")

        # 利用新规范直接定位旧模型
        prev_model_path = self.artifacts_dir / f"model_{prev_version}.pkl"
        if not prev_model_path.exists():
            raise RuntimeError(f"旧版本物理文件丢失: {prev_model_path.name}")

        fallback_model = joblib.load(prev_model_path)
        pred_log = fallback_model.predict(X_scaled_df)[0]
        return pred_log, True

    def _apply_baseline_correction(self, raw_pred_log: float, quality_score: float, baseline: dict):
        """提取的私有方法：主题分位数映射"""
        if baseline and 'p25' in baseline and 'p75' in baseline:
            p25 = float(baseline.get('p25', 0))
            p75 = float(baseline.get('p75', 0))
            baseline_range = max(p75 - p25, 1.0)
            final_pred = p25 + (quality_score / 100.0) * baseline_range
            info = f"超越 {int(quality_score)}% 的同类作品"
        else:
            final_pred = np.expm1(raw_pred_log)
            info = "通用绝对模型评分"
        return final_pred, info

    def calculate_radar_scores(self, video_data: dict, prediction_result: dict) -> list:
        """计算五维雷达图评分 (保持原有逻辑)"""
        try:
            if not video_data:
                return [50.0, 50.0, 50.0, 50.0, 50.0]
                
            pred_digg = prediction_result.get('predicted_digg', 0) if isinstance(prediction_result, dict) else int(prediction_result or 0)
            followers = max(float(video_data.get('follower_count', 10000) or 10000), 1.0)
            
            # 1. 传播力
            interaction_rate = pred_digg / followers
            score_prop = min(100.0, ((interaction_rate / 0.03) ** 0.5) * 100.0)
            
            # 2. 共鸣度
            avg_sentiment = self._safe_float(video_data.get('avg_sentiment', 0.5))
            score_res = min(100.0, 50.0 + ((abs(avg_sentiment - 0.5) * 2) ** 1.5) * 50.0)
            
            # 3. 视觉冲击
            bright = self._safe_float(video_data.get('visual_brightness', 128))
            sat = self._safe_float(video_data.get('visual_saturation', 100))
            vis_distance = math.sqrt((bright / 180.0)**2 + (sat / 80.0)**2)
            score_vis = min(100.0, max(0.0, (vis_distance / math.sqrt(2.0)) * 100.0))
            
            # 4. 听觉节奏
            bpm = self._safe_float(video_data.get('audio_bpm', 110))
            norm_bpm = min(1.0, max(0.0, (bpm - 80) / 70.0))
            score_aud = (norm_bpm ** 1.2) * 100.0
            
            # 5. 完播潜力
            duration = self._safe_float(video_data.get('duration_sec', 15))
            cut_freq = self._safe_float(video_data.get('cut_frequency', 0.5))
            score_comp = min(100.0, max(0.0, (1.0 - math.exp(-cut_freq * 1.5)) * 100.0))
            
            return [
                round(score_prop, 1), round(score_res, 1), 
                round(score_comp, 1), round(score_vis, 1), round(score_aud, 1)
            ]
        except Exception as e:
            logger.error(f"雷达计算异常: {e}")
            return [50.0, 50.0, 50.0, 50.0, 50.0]

    def generate_suggestions(self, radar_scores: list) -> list:
        """生成雷达建议"""
        suggestions = []
        try:
            if radar_scores[0] < 70: suggestions.append("预期互动率一般，建议在文案中增加引导性提问。")
            if radar_scores[1] < 70: suggestions.append("情感共鸣度不足，建议通过特写镜头强化情绪表达。")
            if radar_scores[2] < 70: suggestions.append("完播潜力有待提升，建议加快前3秒的视觉节奏。")
            if radar_scores[3] < 70: suggestions.append("画面亮度或色彩饱和度偏低，建议提升调色质感。")
            if radar_scores[4] < 70: suggestions.append("背景音乐节奏感较弱，建议更换卡点更精准的BGM。")
            
            if not suggestions: 
                suggestions.append("当前视频各项指标表现均衡，建议保持并尝试更多创意风格。")
            return suggestions[:3]
        except Exception:
            return ["建议优化视频前3秒的视觉冲击力以提升留存。"]


if __name__ == "__main__":
    # 本地测试桩
    service = DiggPredictionService()
    test_video = {
        'follower_count': 5000,
        'publish_hour': 18,
        'duration_sec': 15,
        'visual_brightness': 120,
        'visual_saturation': 80,
        'cut_frequency': 0.8,
        'audio_bpm': 105,
        'avg_sentiment': 0.6
    }
    test_baseline = {'mean': 2000, 'p25': 500, 'p75': 5000}
    
    print("\n--- 带基准测试 ---")
    print(service.predict_digg_count(test_video, test_baseline))
    
    print("\n--- 无基准测试 ---")
    print(service.predict_digg_count(test_video, None))

