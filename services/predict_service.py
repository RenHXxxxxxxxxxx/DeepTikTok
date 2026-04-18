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
from dataclasses import dataclass, field, replace

import joblib
import numpy as np
import pandas as pd
from django.conf import settings

from ml_pipeline.preprocessing_contract import normalize_preprocessing_spec, resolve_prep_spec_path

# 配置日志
logger = logging.getLogger(__name__)

# Full-bundle rollback switch:
# when enabled, rollback is attempted only with a complete previous runtime bundle
# (model + scaler + preprocessing spec + version-owned topology truth).
ENABLE_SAFE_FALLBACK = True

BUNDLE_MODE_UNLOADED = "unloaded"
BUNDLE_MODE_LEGACY = "legacy_compatibility"
BUNDLE_MODE_AWARE = "bundle_aware"

# 全局配置
GLOBAL_CONFIG = {
    'CHINESE_MAPPING': {
        '亮度': 'visual_brightness',
        '饱和度': 'visual_saturation',
        'BPM': 'audio_bpm',
        '剪频': 'cut_frequency',
        '剪辑频率': 'cut_frequency'
    },
    'MAX_DIGGS_LOG_SCALE': 12.0  # 用于质量分计算的上限基准 (~16万赞)
}


@dataclass(frozen=True)
class RuntimeAssetBundle:
    version_id: str = None
    model: object = None
    scaler: object = None
    prep: dict = None
    manifest: dict = field(default_factory=dict)
    ordered_feature_names: tuple = field(default_factory=tuple)
    bundle_mode: str = BUNDLE_MODE_UNLOADED
    feature_topology_source: str = "unresolved"
    manifest_mtime: float = 0.0

    def is_loaded(self) -> bool:
        return (
            self.version_id is not None
            and self.model is not None
            and self.scaler is not None
            and bool(self.ordered_feature_names)
        )

    def is_bundle_aware(self) -> bool:
        return self.bundle_mode == BUNDLE_MODE_AWARE and isinstance(self.prep, dict) and bool(self.prep)

class DiggPredictionService:
    """
    视频点赞预测服务 (高可用 Singleton 实现)
    具备: 热更新(Hot-reload)、安全降级(Safe Degradation)、特征自适应(Feature Defense)
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

        # 核心资产状态：服务侧只维护一个活动 bundle，避免热更新时出现混合版本内存状态。
        self._runtime_bundle = self._empty_runtime_bundle()
        self._manifest_mtime = 0.0
        self._last_reload_attempt_mtime = 0.0

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

    def _empty_runtime_bundle(self) -> RuntimeAssetBundle:
        return RuntimeAssetBundle()

    def _get_active_bundle(self) -> RuntimeAssetBundle:
        if not hasattr(self, '_runtime_bundle'):
            self._runtime_bundle = self._empty_runtime_bundle()
        return self._runtime_bundle

    def _replace_runtime_bundle(self, **changes):
        current_bundle = self._get_active_bundle()
        normalized = {}
        for key, value in changes.items():
            if key == 'manifest':
                normalized[key] = dict(value or {})
            elif key == 'ordered_feature_names':
                normalized[key] = tuple(value or ())
            else:
                normalized[key] = value
        self._runtime_bundle = replace(current_bundle, **normalized)
        self._manifest_mtime = self._runtime_bundle.manifest_mtime

    def _activate_runtime_bundle(self, runtime_bundle: RuntimeAssetBundle):
        self._runtime_bundle = runtime_bundle
        self._manifest_mtime = runtime_bundle.manifest_mtime
        self._last_reload_attempt_mtime = runtime_bundle.manifest_mtime

    @property
    def model(self):
        return self._get_active_bundle().model

    @model.setter
    def model(self, value):
        self._replace_runtime_bundle(model=value)

    @property
    def scaler(self):
        return self._get_active_bundle().scaler

    @scaler.setter
    def scaler(self, value):
        self._replace_runtime_bundle(scaler=value)

    @property
    def prep(self):
        return self._get_active_bundle().prep

    @prep.setter
    def prep(self, value):
        self._replace_runtime_bundle(prep=value)

    @property
    def manifest(self):
        return self._get_active_bundle().manifest

    @manifest.setter
    def manifest(self, value):
        self._replace_runtime_bundle(manifest=value)

    @property
    def loaded_version(self):
        return self._get_active_bundle().version_id

    @loaded_version.setter
    def loaded_version(self, value):
        self._replace_runtime_bundle(version_id=value)

    @property
    def bundle_mode(self):
        return self._get_active_bundle().bundle_mode

    @bundle_mode.setter
    def bundle_mode(self, value):
        self._replace_runtime_bundle(bundle_mode=value)

    @property
    def ordered_feature_names(self):
        return list(self._get_active_bundle().ordered_feature_names)

    @ordered_feature_names.setter
    def ordered_feature_names(self, value):
        self._replace_runtime_bundle(ordered_feature_names=value)

    # ==========================================
    # 核心热重载与资产加载逻辑
    # ==========================================
    def _check_hot_reload(self):
        """轻量级探测：通过比对文件修改时间实现零宕机热重载"""
        try:
            if self.manifest_path.exists():
                current_mtime = self.manifest_path.stat().st_mtime
                active_bundle = self._get_active_bundle()
                if current_mtime > max(active_bundle.manifest_mtime, self._last_reload_attempt_mtime):
                    with self._lock:
                        # 双重检查防止并发穿透
                        latest_mtime = self.manifest_path.stat().st_mtime
                        if latest_mtime > max(self._get_active_bundle().manifest_mtime, self._last_reload_attempt_mtime):
                            self._last_reload_attempt_mtime = latest_mtime
                            logger.info(" [Hot Reload] 监测到新模型版本发布，执行热更新...")
                            new_bundle = self._load_current_runtime_bundle()
                            if new_bundle is not None:
                                self._activate_runtime_bundle(new_bundle)
                                logger.info(
                                    " [Hot Reload] 已原子切换到新活动 bundle: version=%s bundle_mode=%s topology=%s",
                                    new_bundle.version_id,
                                    new_bundle.bundle_mode,
                                    new_bundle.feature_topology_source
                                )
                            else:
                                logger.error(
                                    " [Hot Reload] 新版本 bundle 加载失败；保留当前活动 bundle version=%s。",
                                    self.loaded_version
                                )
        except Exception as e:
            logger.error(f" 热重载检查异常: {e}")

    def _load_assets(self):
        """从磁盘加载当前活动 bundle，并在验证通过后一次性切换。"""
        try:
            new_bundle = self._load_current_runtime_bundle()
            if new_bundle is None:
                logger.error(" [Bundle Load Failed] 当前活动 bundle 初始化失败，服务进入未加载状态。")
                self._reset_state()
                return

            self._activate_runtime_bundle(new_bundle)
            logger.info(
                " 成功挂载冠军模型: %s (%s) [bundle_mode=%s, topology=%s]",
                new_bundle.version_id,
                new_bundle.manifest.get('best_model'),
                new_bundle.bundle_mode,
                new_bundle.feature_topology_source
            )

        except Exception as e:
            logger.error(f" 资产加载严重崩溃: {e}\n{traceback.format_exc()}")
            self._reset_state()

    def _reset_state(self):
        """容错保护：清空损坏状态"""
        self._activate_runtime_bundle(self._empty_runtime_bundle())

    def _load_current_runtime_bundle(self):
        """读取当前 manifest 并构建可原子切换的运行时 bundle。"""
        if not self.manifest_path.exists():
            logger.error(f" 找不到版本清单: {self.manifest_path}")
            return None

        manifest_mtime = self.manifest_path.stat().st_mtime
        with open(self.manifest_path, 'r', encoding='utf-8') as f:
            manifest_payload = json.load(f)

        version_id = manifest_payload.get('current_version')
        if not version_id:
            logger.error(" 清单中缺失 current_version")
            return None

        return self._load_runtime_bundle_by_version(
            version_id=version_id,
            manifest_payload=manifest_payload,
            manifest_mtime=manifest_mtime,
            purpose="active"
        )

    def _load_runtime_bundle_by_version(
        self,
        version_id: str,
        manifest_payload: dict = None,
        manifest_mtime: float = 0.0,
        purpose: str = "active"
    ):
        """构建某个版本的运行时 bundle；调用方决定是否激活。"""
        bundle_manifest = dict(manifest_payload or {})
        bundle_manifest['bundle_version_id'] = version_id
        manifest_describes_requested_version = bundle_manifest.get('current_version') == version_id

        bundle_artifacts = bundle_manifest.get('bundle_artifacts', {}) if manifest_describes_requested_version else {}
        model_file = bundle_artifacts.get('model_file') or f'model_{version_id}.pkl'
        scaler_file = bundle_artifacts.get('scaler_file') or f'scaler_{version_id}.pkl'
        model_path = self.artifacts_dir / model_file
        scaler_path = self.artifacts_dir / scaler_file
        missing_artifacts = [path.name for path in (model_path, scaler_path) if not path.exists()]
        if missing_artifacts:
            logger.error(
                " [Bundle Load Failed] 无法构建 %s bundle version=%s；缺失文件: %s",
                purpose,
                version_id,
                ", ".join(missing_artifacts)
            )
            return None

        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
        except Exception as e:
            logger.error(
                " [Bundle Load Failed] 版本 %s 的 model/scaler 读取失败 (purpose=%s): %s",
                version_id,
                purpose,
                e,
                exc_info=True
            )
            return None

        prep, bundle_mode = self._load_optional_prep_spec(
            version_id,
            manifest_payload=bundle_manifest
        )
        ordered_feature_names, topology_source = self._resolve_ordered_feature_names(
            version_id=version_id,
            manifest_payload=bundle_manifest,
            model=model,
            scaler=scaler,
            prep=prep,
            manifest_describes_requested_version=manifest_describes_requested_version
        )

        runtime_bundle = RuntimeAssetBundle(
            version_id=version_id,
            model=model,
            scaler=scaler,
            prep=prep,
            manifest=bundle_manifest,
            ordered_feature_names=tuple(ordered_feature_names or ()),
            bundle_mode=bundle_mode,
            feature_topology_source=topology_source,
            manifest_mtime=manifest_mtime
        )

        validation_errors = self._validate_runtime_bundle(runtime_bundle, purpose=purpose)
        if validation_errors:
            logger.error(
                " [Bundle Load Failed] 版本 %s 的 %s bundle 校验失败: %s",
                version_id,
                purpose,
                "; ".join(validation_errors)
            )
            return None

        return runtime_bundle

    def _resolve_ordered_feature_names(
        self,
        version_id: str,
        manifest_payload: dict,
        model,
        scaler,
        prep: dict,
        manifest_describes_requested_version: bool
    ):
        """解析版本自有的特征拓扑来源，避免跨版本误用当前 manifest。"""
        prep_paths = [
            ('feature_names_in',),
            ('feature_order',),
            ('preprocessing', 'feature_names_in'),
            ('preprocessing', 'feature_order'),
            ('model_input', 'feature_names_in'),
            ('model_input', 'feature_order'),
        ]

        for path in prep_paths:
            candidate = self._get_nested_value(prep or {}, path)
            if isinstance(candidate, (list, tuple)) and candidate:
                return [str(name) for name in candidate], f"prep.{'.'.join(path)}"

        model_feature_names = getattr(model, 'feature_names_in_', None)
        if model_feature_names is not None and len(model_feature_names):
            return [str(name) for name in model_feature_names], "model.feature_names_in_"

        scaler_feature_names = getattr(scaler, 'feature_names_in_', None)
        if scaler_feature_names is not None and len(scaler_feature_names):
            return [str(name) for name in scaler_feature_names], "scaler.feature_names_in_"

        if manifest_describes_requested_version:
            manifest_feature_names = manifest_payload.get('feature_names_in')
            if isinstance(manifest_feature_names, list) and manifest_feature_names:
                return [str(name) for name in manifest_feature_names], "manifest.feature_names_in"

            feature_importances = manifest_payload.get('feature_importances', {})
            if isinstance(feature_importances, dict) and feature_importances:
                return [str(name) for name in feature_importances.keys()], "manifest.feature_importances"

        logger.warning(
            " [Bundle Topology] 版本 %s 缺少版本自有特征拓扑元数据；回退到 base_features 兼容模式。",
            version_id
        )
        return list(self.base_features), "base_features_fallback"

    def _validate_runtime_bundle(self, runtime_bundle: RuntimeAssetBundle, purpose: str = "active"):
        """校验 bundle 运行时完整性；rollback 会额外要求 bundle-aware 资产齐全。"""
        errors = []
        feature_count = len(runtime_bundle.ordered_feature_names)

        if runtime_bundle.version_id is None:
            errors.append("missing version_id")
        if runtime_bundle.model is None:
            errors.append("missing model")
        if runtime_bundle.scaler is None:
            errors.append("missing scaler")
        if feature_count == 0:
            errors.append("missing ordered_feature_names")

        model_feature_count = getattr(runtime_bundle.model, 'n_features_in_', None)
        if model_feature_count not in (None, feature_count):
            errors.append(
                f"model expects {model_feature_count} features but bundle topology has {feature_count}"
            )

        scaler_mean = getattr(runtime_bundle.scaler, 'mean_', None)
        if scaler_mean is not None and len(scaler_mean) != feature_count:
            errors.append(
                f"scaler mean length {len(scaler_mean)} != bundle topology length {feature_count}"
            )

        if purpose == "rollback":
            if not runtime_bundle.is_bundle_aware():
                errors.append("missing valid prep metadata for full-bundle rollback")
            if runtime_bundle.feature_topology_source == "base_features_fallback":
                errors.append("rollback topology is not version-owned")

        return errors

    def _load_optional_prep_spec(self, version_id: str, manifest_payload: dict = None):
        """按版本加载可选的预处理规格；缺失或无效时保持 legacy compatibility mode。"""
        prep_path = resolve_prep_spec_path(
            self.artifacts_dir,
            version_id,
            manifest_payload=manifest_payload
        )

        if not prep_path.exists():
            logger.info(
                f" [Bundle Mode] 版本 {version_id} 未找到 {prep_path.name}；"
                f"使用 {BUNDLE_MODE_LEGACY} (model + scaler only)。"
            )
            return None, BUNDLE_MODE_LEGACY

        try:
            with open(prep_path, 'r', encoding='utf-8') as f:
                prep_spec = normalize_preprocessing_spec(
                    json.load(f),
                    version_id=version_id
                )

            if not isinstance(prep_spec, dict):
                logger.warning(
                    f" [Bundle Mode] 版本 {version_id} 的 {prep_path.name} 已存在但格式无效"
                    f"（期望 JSON object）；回退到 {BUNDLE_MODE_LEGACY}。"
                )
                return None, BUNDLE_MODE_LEGACY

            logger.info(
                f" [Bundle Mode] 版本 {version_id} 已加载 {prep_path.name}；"
                f"进入 {BUNDLE_MODE_AWARE} (model + scaler + prep spec)。"
            )
            return prep_spec, BUNDLE_MODE_AWARE

        except json.JSONDecodeError as e:
            logger.warning(
                f" [Bundle Mode] 版本 {version_id} 的 {prep_path.name} 存在但 JSON 解析失败；"
                f"使用 {BUNDLE_MODE_LEGACY}。错误: {e}"
            )
            return None, BUNDLE_MODE_LEGACY
        except Exception as e:
            logger.warning(
                f" [Bundle Mode] 版本 {version_id} 的 {prep_path.name} 存在但读取失败；"
                f"使用 {BUNDLE_MODE_LEGACY}。错误: {e}"
            )
            return None, BUNDLE_MODE_LEGACY

    # ==========================================
    # 数据工程管道 (特征抽取)
    # ==========================================
    def _safe_float(self, val, default=0.0):
        try:
            return default if pd.isna(val) or val == '' else float(val)
        except:
            return default

    def _get_nested_value(self, data: dict, path: tuple):
        """安全读取嵌套字典值；用于兼容轻量演进中的 prep spec 结构。"""
        current = data
        for key in path:
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
        return current

    def _resolve_prep_feature_context(self, current_theme: str, runtime_bundle: RuntimeAssetBundle = None):
        """
        解析版本拥有的预处理真值。
        目标：当 prep spec 可用时，仅基于当前主题 + 版本内先验构建模型输入，不再把 display baseline 当作权威来源。
        """
        bundle = runtime_bundle or self._get_active_bundle()
        if not isinstance(bundle.prep, dict) or not bundle.prep:
            return None

        def pick_value(*paths):
            for path in paths:
                value = self._get_nested_value(bundle.prep, path)
                if value is not None:
                    return value
            return None

        def pick_float_from_mapping(mapping, *keys):
            if not isinstance(mapping, dict):
                return None
            for key in keys:
                if key not in mapping:
                    continue
                parsed = self._safe_float(mapping.get(key), default=np.nan)
                if not pd.isna(parsed):
                    return float(parsed)
            return None

        def pick_float(*paths):
            for path in paths:
                parsed = self._safe_float(pick_value(path), default=np.nan)
                if not pd.isna(parsed):
                    return float(parsed)
            return None

        def coerce_numeric_map(raw_mapping):
            coerced = {}
            if not isinstance(raw_mapping, dict):
                return coerced

            for feature_name, value in raw_mapping.items():
                numeric = None
                if isinstance(value, dict):
                    numeric = pick_float_from_mapping(
                        value, 'value', 'median', 'default', 'fill_value', 'impute_value'
                    )
                else:
                    parsed = self._safe_float(value, default=np.nan)
                    if not pd.isna(parsed):
                        numeric = float(parsed)

                if numeric is not None:
                    coerced[str(feature_name)] = numeric

            return coerced

        def coerce_clip_map(raw_mapping):
            clips = {}
            if not isinstance(raw_mapping, dict):
                return clips

            for feature_name, value in raw_mapping.items():
                lower = None
                upper = None
                if isinstance(value, dict):
                    lower = pick_float_from_mapping(value, 'lower', 'min')
                    upper = pick_float_from_mapping(value, 'upper', 'max', 'value')
                else:
                    parsed = self._safe_float(value, default=np.nan)
                    if not pd.isna(parsed):
                        upper = float(parsed)

                if lower is not None or upper is not None:
                    clips[str(feature_name)] = {'lower': lower, 'upper': upper}

            return clips

        raw_theme_stats = pick_value(
            ('theme_stats',),
            ('theme_priors',),
            ('theme_encoding', 'theme_stats'),
            ('theme_encoding', 'per_theme'),
            ('preprocessing', 'theme_stats'),
            ('preprocessing', 'theme_encoding', 'theme_stats'),
            ('preprocessing', 'theme_encoding', 'per_theme'),
        )

        theme_entry = {}
        if isinstance(raw_theme_stats, dict):
            candidate = raw_theme_stats.get(current_theme)
            if isinstance(candidate, dict):
                theme_entry = candidate
            elif candidate is not None:
                numeric_candidate = self._safe_float(candidate, default=np.nan)
                if not pd.isna(numeric_candidate):
                    theme_entry = {'theme_encoded': float(numeric_candidate)}
        elif isinstance(raw_theme_stats, list):
            for item in raw_theme_stats:
                if not isinstance(item, dict):
                    continue
                theme_label = str(item.get('theme_label') or item.get('theme') or item.get('label') or '')
                if theme_label == current_theme:
                    theme_entry = item
                    break

        global_mean_log = pick_float(
            ('bayesian_global_mean',),
            ('global_prior',),
            ('theme_encoding', 'global_mean_log'),
            ('theme_encoding', 'global_mean'),
            ('theme_encoding', 'global_prior'),
            ('preprocessing', 'bayesian_global_mean'),
            ('preprocessing', 'global_prior'),
            ('preprocessing', 'theme_encoding', 'global_mean_log'),
            ('preprocessing', 'theme_encoding', 'global_mean'),
            ('preprocessing', 'theme_encoding', 'global_prior'),
        )
        weight = pick_float(
            ('bayesian_weight',),
            ('theme_encoding', 'weight'),
            ('theme_encoding', 'bayesian_weight'),
            ('preprocessing', 'bayesian_weight'),
            ('preprocessing', 'theme_encoding', 'weight'),
            ('preprocessing', 'theme_encoding', 'bayesian_weight'),
        )
        if weight is None:
            weight = 10.0

        theme_encoded = pick_float_from_mapping(theme_entry, 'theme_encoded', 'encoded', 'encoded_value')

        if theme_encoded is None:
            local_mean_log = pick_float_from_mapping(theme_entry, 'local_mean', 'local_mean_log', 'mean_log')
            if local_mean_log is None and 'mean' in theme_entry:
                raw_mean = self._safe_float(theme_entry.get('mean'), default=np.nan)
                if not pd.isna(raw_mean):
                    local_mean_log = np.log1p(raw_mean)

            theme_count = pick_float_from_mapping(theme_entry, 'count', 'sample_count', 'n')

            if local_mean_log is not None and global_mean_log is not None:
                theme_count = max(theme_count or 0.0, 0.0)
                denominator = theme_count + weight
                theme_encoded = (
                    (theme_count * local_mean_log + weight * global_mean_log) / denominator
                    if denominator > 0 else global_mean_log
                )
            elif global_mean_log is not None:
                # 冷启动主题直接回落到版本拥有的全局先验，而不是 display baseline。
                theme_encoded = global_mean_log

        if theme_encoded is None:
            return None

        numeric_defaults = coerce_numeric_map(
            pick_value(
                ('numeric_imputations',),
                ('numeric_defaults',),
                ('imputations',),
                ('preprocessing', 'numeric_imputations'),
                ('preprocessing', 'numeric_defaults'),
                ('preprocessing', 'imputations'),
            )
        )
        clip_values = coerce_clip_map(
            pick_value(
                ('clip_values',),
                ('feature_clips',),
                ('preprocessing', 'clip_values'),
                ('preprocessing', 'feature_clips'),
            )
        )

        return {
            'theme_encoded': float(theme_encoded),
            'numeric_defaults': numeric_defaults,
            'clip_values': clip_values,
            'theme_source': 'prep_theme_stats' if theme_entry else 'prep_global_prior'
        }

    def _resolve_display_theme_baseline(self, theme_baseline: dict = None, display_theme_baseline: dict = None):
        """统一 display baseline 参数名；theme_baseline 保留为 legacy compatibility alias。"""
        return display_theme_baseline if display_theme_baseline is not None else theme_baseline

    def _build_model_input_features(
        self,
        video_data: dict,
        display_theme_baseline: dict = None,
        runtime_bundle: RuntimeAssetBundle = None
    ) -> pd.DataFrame:
        """
        构建模型输入特征。
        优先使用版本 bundle 拥有的预处理真值；display baseline 永远只属于展示层。
        """
        bundle = runtime_bundle or self._get_active_bundle()

        # 1. 键名中文化映射
        for cn_key, en_key in GLOBAL_CONFIG['CHINESE_MAPPING'].items():
            if cn_key in video_data:
                video_data[en_key] = video_data.pop(cn_key)

        current_theme = str(video_data.get('theme_label', 'Unknown') or 'Unknown')
        prep_context = self._resolve_prep_feature_context(current_theme, bundle)
        numeric_defaults = prep_context.get('numeric_defaults', {}) if prep_context else {}
        clip_values = prep_context.get('clip_values', {}) if prep_context else {}

        def apply_clip_bounds(feature_name: str, feature_value: float) -> float:
            bounds = clip_values.get(feature_name)
            if not bounds:
                return feature_value
            if bounds.get('lower') is not None:
                feature_value = max(feature_value, bounds['lower'])
            if bounds.get('upper') is not None:
                feature_value = min(feature_value, bounds['upper'])
            return feature_value

        # 2. 基础物理特征提取与对数平滑
        fol_raw = self._safe_float(video_data.get('follower_count'), numeric_defaults.get('follower_count', 10000.0))
        fol_log = apply_clip_bounds('follower_count_log', np.log1p(fol_raw))

        hrr = apply_clip_bounds(
            'publish_hour',
            self._safe_float(video_data.get('publish_hour'), numeric_defaults.get('publish_hour', 0.0))
        )
        dur = apply_clip_bounds(
            'duration_sec',
            self._safe_float(video_data.get('duration_sec'), numeric_defaults.get('duration_sec', 0.0))
        )
        sent = apply_clip_bounds(
            'avg_sentiment',
            self._safe_float(video_data.get('avg_sentiment'), numeric_defaults.get('avg_sentiment', 0.5))
        )
        vb = apply_clip_bounds(
            'visual_brightness',
            self._safe_float(video_data.get('visual_brightness'), numeric_defaults.get('visual_brightness', 128.0))
        )
        vs = apply_clip_bounds(
            'visual_saturation',
            self._safe_float(video_data.get('visual_saturation'), numeric_defaults.get('visual_saturation', 100.0))
        )
        cf = apply_clip_bounds(
            'cut_frequency',
            self._safe_float(video_data.get('cut_frequency'), numeric_defaults.get('cut_frequency', 0.5))
        )
        bpm = apply_clip_bounds(
            'audio_bpm',
            self._safe_float(video_data.get('audio_bpm'), numeric_defaults.get('audio_bpm', 110.0))
        )

        # 2.1 主题编码：优先使用版本拥有的 prep 真值；legacy compatibility 也只使用版本/全局先验。
        if prep_context:
            theme_enc = prep_context['theme_encoded']
            logger.info(
                " [Feature Build] 使用版本预处理元数据构建模型输入 (theme=%s, version=%s, source=%s)；display baseline 仅用于展示层。",
                current_theme,
                bundle.version_id,
                prep_context.get('theme_source', 'prep')
            )
        else:
            theme_enc = float(
                self._safe_float(
                    bundle.manifest.get('bayesian_global_mean'),
                    np.log1p(10000.0)
                )
            )

            if bundle.is_bundle_aware():
                logger.warning(
                    " [Feature Build] 版本 %s 的 prep metadata 缺少可用主题先验；回退到版本拥有的全局先验，不使用 display baseline。",
                    bundle.version_id
                )
            else:
                logger.info(
                    " [Feature Build] 版本 %s 处于 legacy compatibility；模型输入仅使用版本/全局先验。display baseline 仍只用于展示层。",
                    bundle.version_id
                )

        # 3. 构造占位 DataFrame
        ordered_feature_names = list(bundle.ordered_feature_names)
        X_final = pd.DataFrame(0.0, index=[0], columns=ordered_feature_names)

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

        for feature_name, feature_value in list(features_to_inject.items()):
            features_to_inject[feature_name] = apply_clip_bounds(feature_name, feature_value)

        for col, val in features_to_inject.items():
            if col in ordered_feature_names:
                X_final.at[0, col] = val

        # 5. 分类特征（主题）独热映射
        theme_mapped = False
        for col in ordered_feature_names:
            if col.startswith('theme_'):
                if current_theme == col.replace('theme_', ''):
                    X_final.at[0, col] = 1.0
                    theme_mapped = True

        if not theme_mapped and 'theme_Unknown' in ordered_feature_names:
            X_final.at[0, 'theme_Unknown'] = 1.0

        # 6. 均值插补兜底 (防止未覆盖特征传入0.0导致缩放器漂移)
        injected_keys = set(features_to_inject.keys())
        if bundle.scaler is not None and hasattr(bundle.scaler, 'mean_'):
            for idx, col in enumerate(ordered_feature_names):
                if idx < len(bundle.scaler.mean_):
                    if col not in injected_keys and not col.startswith('theme_'):
                        X_final.at[0, col] = bundle.scaler.mean_[idx]

        return X_final[ordered_feature_names] # 强制列序对齐

    def _build_features(self, video_data: dict, theme_baseline: dict = None, display_theme_baseline: dict = None) -> pd.DataFrame:
        """兼容旧调用方的私有别名；保留 theme_baseline 仅用于现有调用方兼容。"""
        resolved_display_theme_baseline = self._resolve_display_theme_baseline(
            theme_baseline=theme_baseline,
            display_theme_baseline=display_theme_baseline
        )
        return self._build_model_input_features(video_data, resolved_display_theme_baseline)

    def _predict_raw_model_log(
        self,
        model_input_features: pd.DataFrame,
        runtime_bundle: RuntimeAssetBundle = None
    ) -> float:
        """执行纯模型推理，返回模型的原始 log-space 输出，不混入 display/explanation 映射。"""
        bundle = runtime_bundle or self._get_active_bundle()
        ordered_feature_names = list(bundle.ordered_feature_names)
        X_scaled_ndarray = bundle.scaler.transform(model_input_features)
        X_scaled_df = pd.DataFrame(X_scaled_ndarray, columns=ordered_feature_names)
        return bundle.model.predict(X_scaled_df)[0]

    # ==========================================
    # 预测与后处理逻辑
    # ==========================================
    def predict_digg_count(
        self,
        video_data: dict,
        theme_baseline: dict = None,
        display_theme_baseline: dict = None
    ) -> dict:
        """
        主入口：先得到模型原始输出，再构建 display-facing 结果。
        返回结构中的 `predicted_digg` 保持现有接口语义，表示最终展示值，而非原始模型输出。
        """
        self._check_hot_reload()

        resolved_display_theme_baseline = self._resolve_display_theme_baseline(
            theme_baseline=theme_baseline,
            display_theme_baseline=display_theme_baseline
        )

        active_bundle = self._get_active_bundle()
        if not active_bundle.is_loaded():
            logger.error(
                " [Service Unavailable] 活动 runtime bundle 未就绪；无法执行推理。version=%s bundle_mode=%s",
                active_bundle.version_id,
                active_bundle.bundle_mode
            )
            return {"predicted_digg": 0, "quality_score": 0.0, "percentile_rank": "服务暂不可用", "baseline_ref": None}

        try:
            # 1. 构建模型输入：与 display baseline 解释层分离
            model_input_features = self._build_model_input_features(
                video_data,
                resolved_display_theme_baseline,
                runtime_bundle=active_bundle
            )

            # 2. 主模型预测；若失败则尝试完整版本 bundle 回滚，否则进入安全降级
            try:
                #下面是bug开关
                #raise ValueError("【混沌测试】模拟新模型遭遇未知特征崩溃！")
                raw_model_pred_log = self._predict_raw_model_log(
                    model_input_features,
                    runtime_bundle=active_bundle
                )
            except Exception as e:
                logger.error(
                    " [Primary Inference Failed] 主模型推理失败；active version=%s bundle_mode=%s。错误: %s",
                    active_bundle.version_id,
                    active_bundle.bundle_mode,
                    e,
                    exc_info=True
                )
                rollback_result = self._execute_fallback(
                    video_data=video_data,
                    display_theme_baseline=resolved_display_theme_baseline,
                    failed_bundle=active_bundle
                )
                if rollback_result is None:
                    logger.warning(" [Safe Degradation] 主模型失败且完整回滚不可用；返回稳定安全降级响应。")
                    return self._build_safe_degraded_response(resolved_display_theme_baseline)

                raw_model_pred_log, serving_bundle = rollback_result
                logger.info(
                    " [Rollback Success] 使用完整上一版本 bundle 恢复推理。serving_version=%s bundle_mode=%s",
                    serving_bundle.version_id,
                    serving_bundle.bundle_mode
                )

            # 3. 原始模型语义：
            # raw_model_pred_log 是模型直接输出的 log-space 分数；
            # raw_model_pred_digg 是对该原始输出做 expm1 后得到的原始点赞量估计。
            raw_model_pred_log = max(0.0, float(raw_model_pred_log))
            raw_model_pred_digg = max(0.0, float(np.expm1(raw_model_pred_log)))
            quality_score = min(max((raw_model_pred_log / GLOBAL_CONFIG['MAX_DIGGS_LOG_SCALE']) * 100.0, 0.0), 100.0)

            # 4. display-layer 映射：
            # display_pred_digg 是最终展示值，可在当前主题 benchmark 上做解释层映射，
            # 因此它与 raw_model_pred_digg 可能相同，也可能不同。
            display_pred_digg, display_percentile_rank = self._map_raw_prediction_to_display_output(
                raw_model_pred_digg,
                quality_score,
                resolved_display_theme_baseline
            )

            # 5. 对外继续返回现有字段结构，避免打破调用方。
            return {
                "predicted_digg": int(display_pred_digg),
                "quality_score": round(quality_score, 1),
                "percentile_rank": display_percentile_rank,
                "baseline_ref": resolved_display_theme_baseline
            }

        except Exception as e:
            logger.error(f" 预测防线彻底击穿，返回安全默认值: {e}\n{traceback.format_exc()}")
            return self._build_safe_degraded_response(resolved_display_theme_baseline)

    def _execute_fallback(
        self,
        video_data: dict,
        display_theme_baseline: dict = None,
        failed_bundle: RuntimeAssetBundle = None
    ):
        """仅使用完整上一版本 bundle 尝试回滚；任何不完整状态都直接放弃并交给安全降级。"""
        if not ENABLE_SAFE_FALLBACK:
            logger.warning(" [Rollback Disabled] ENABLE_SAFE_FALLBACK=False；不会执行任何回滚。")
            return None

        current_bundle = failed_bundle or self._get_active_bundle()
        previous_version = current_bundle.manifest.get('previous_version') if current_bundle.manifest else None
        if not previous_version:
            logger.warning(
                " [Rollback Unavailable] 活动 version=%s 未声明 previous_version；无法执行完整 bundle 回滚。",
                current_bundle.version_id
            )
            return None

        logger.info(
            " [Rollback Attempt] 主模型失败后尝试完整上一版本 bundle 回滚：from=%s -> previous=%s",
            current_bundle.version_id,
            previous_version
        )

        rollback_bundle = self._load_runtime_bundle_by_version(
            version_id=previous_version,
            manifest_payload=current_bundle.manifest,
            purpose="rollback"
        )
        if rollback_bundle is None:
            logger.warning(
                " [Rollback Unavailable] previous version=%s 缺少完整可用 bundle；不会执行部分回滚。",
                previous_version
            )
            return None

        try:
            rollback_features = self._build_model_input_features(
                video_data,
                display_theme_baseline,
                runtime_bundle=rollback_bundle
            )
            rollback_raw_model_pred_log = self._predict_raw_model_log(
                rollback_features,
                runtime_bundle=rollback_bundle
            )
            return rollback_raw_model_pred_log, rollback_bundle
        except Exception as e:
            logger.error(
                " [Rollback Failed] 完整上一版本 bundle 推理失败：version=%s。错误: %s",
                rollback_bundle.version_id,
                e,
                exc_info=True
            )
            return None

    def _build_safe_degraded_response(self, display_theme_baseline: dict = None) -> dict:
        """主模型/回滚都不可用时返回稳定且兼容的 display-facing 安全降级响应。"""
        return {
            "predicted_digg": 0,
            "quality_score": 0.0,
            "percentile_rank": "安全降级模式",
            "baseline_ref": display_theme_baseline
        }

    def _map_raw_prediction_to_display_output(
        self,
        raw_model_pred_digg: float,
        quality_score: float,
        display_theme_baseline: dict
    ):
        """
        将原始点赞量估计映射为最终 display-facing 输出。
        raw_model_pred_digg 代表模型原始预测；
        display_pred_digg 代表 UI/解释层最终展示值。
        """
        # Strategy anchor: display_theme_baseline is current-theme UI context, not model-input preprocessing truth.
        if display_theme_baseline and 'p25' in display_theme_baseline and 'p75' in display_theme_baseline:
            p25 = float(display_theme_baseline.get('p25', 0))
            p75 = float(display_theme_baseline.get('p75', 0))
            baseline_range = max(p75 - p25, 1.0)
            display_pred_digg = p25 + (quality_score / 100.0) * baseline_range
            display_percentile_rank = f"超越 {int(quality_score)}% 的同类作品"
        else:
            display_pred_digg = raw_model_pred_digg
            display_percentile_rank = "通用绝对模型评分"
        return display_pred_digg, display_percentile_rank

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

