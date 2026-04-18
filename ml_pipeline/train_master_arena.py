import os
import sys
import time
import json
import shutil
import datetime
import traceback
import django
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from django.db.models import Avg
from sklearn.model_selection import train_test_split, ParameterGrid, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
from tqdm import tqdm

try:
    from ml_pipeline.preprocessing_contract import (
        apply_numeric_preprocessing as shared_apply_numeric_preprocessing,
        add_derived_features as shared_add_derived_features,
        build_feature_matrix as shared_build_feature_matrix,
        build_preprocessing_spec as shared_build_preprocessing_spec,
        build_versioned_evaluation_features,
        normalize_preprocessing_spec,
        resolve_prep_spec_path,
        transform_with_preprocessing_context as shared_transform_with_preprocessing_context,
    )
except ImportError:
    from preprocessing_contract import (
        apply_numeric_preprocessing as shared_apply_numeric_preprocessing,
        add_derived_features as shared_add_derived_features,
        build_feature_matrix as shared_build_feature_matrix,
        build_preprocessing_spec as shared_build_preprocessing_spec,
        build_versioned_evaluation_features,
        normalize_preprocessing_spec,
        resolve_prep_spec_path,
        transform_with_preprocessing_context as shared_transform_with_preprocessing_context,
    )

# 核心路径解析与环境配置
current_dir = os.path.dirname(os.path.abspath(__file__))
outer_root = os.path.dirname(current_dir)
django_inner_root = os.path.join(outer_root, 'renhangxi_tiktok_bysj')
artifacts_dir = os.path.join(outer_root, 'artifacts')
manifest_path = os.path.join(artifacts_dir, 'version_manifest.json')

if outer_root in sys.path:
    sys.path.remove(outer_root)
if django_inner_root not in sys.path:
    sys.path.insert(0, django_inner_root)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'settings')
os.environ['ML_TRAINING_MODE'] = '1'
try:
    django.setup()
except Exception as e:
    # 保护环境初始化
    print(f"[ERROR] Django setup failed: {str(e)}")

from django.apps import apps

# 全局配置项：避免魔法数字硬编码
GLOBAL_CONFIG = {
    'test_size': 0.15,
    'selection_validation_size': 0.17647058823529413,
    'random_state': 42,
    'min_theme_samples': 5,
    'target_ratio': 0.70,
    'theme_encode_smoothing_weight': 10.0,
    'cap_quantile': 0.95,
    'cv_folds': 5,
    'n_jobs': 1,
    'clip_max': 20,
    'win_margin': 0.02,
    'default_fallback': {
        'avg_sentiment': 0.5,
        'visual_brightness': 128,
        'visual_saturation': 100,
        'cut_frequency': 0.5,
        'audio_bpm': 110
    },
    'param_grids': {
        'RandomForest': {
            'n_estimators': [100, 300],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5]
        },
        'LightGBM': {
            'n_estimators': [300, 500],
            'max_depth': [5, 7],
            'learning_rate': [0.05, 0.1]
        },
        'XGBoost': {
            'n_estimators': [300, 500],
            'max_depth': [5, 7],
            'learning_rate': [0.05, 0.1]
        }
    }
}

# 基础工具函数
def convert_duration_to_seconds(duration_str):
    try:
        d_str = str(duration_str).strip()
        if ':' in d_str:
            parts = d_str.split(':')
            if len(parts) == 2:
                return int(parts[0]) * 60 + int(parts[1])
            elif len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        return int(float(d_str))
    except Exception:
        # 异常处理保证健壮性
        return 0

def prepare_base_training_dataframe(df):
    # 只做与目标值无关的基础清洗与解析，允许在切分前执行
    df_prepared = df.copy()
    df_prepared['duration_sec'] = df_prepared['duration'].apply(convert_duration_to_seconds)
    df_prepared['publish_hour'] = pd.to_datetime(df_prepared['create_time'], errors='coerce').dt.hour
    df_prepared['follower_count_log'] = np.log1p(df_prepared['follower_count'])
    return df_prepared

def fit_train_owned_theme_encoding(train_df, smoothing_weight):
    # 所有目标感知主题统计量必须只从训练集拟合
    if train_df.empty:
        raise ValueError("Training split is empty; cannot fit theme statistics.")

    theme_fit_df = train_df[['theme_label', 'digg_count']].copy()
    theme_fit_df['digg_count_log'] = np.log1p(theme_fit_df['digg_count'])
    global_mean = theme_fit_df['digg_count_log'].mean()

    theme_stats = theme_fit_df.groupby('theme_label')['digg_count_log'].agg(['count', 'mean']).reset_index()
    theme_stats.rename(columns={'mean': 'local_mean'}, inplace=True)
    theme_stats['theme_encoded'] = (
        theme_stats['count'] * theme_stats['local_mean'] + smoothing_weight * global_mean
    ) / (theme_stats['count'] + smoothing_weight)

    return {
        'global_mean': float(global_mean),
        'smoothing_weight': float(smoothing_weight),
        'theme_stats': theme_stats.copy(),
        'theme_count_by_theme': theme_stats.set_index('theme_label')['count'].to_dict(),
        'theme_local_mean_by_theme': theme_stats.set_index('theme_label')['local_mean'].to_dict(),
        'theme_encoded_by_theme': theme_stats.set_index('theme_label')['theme_encoded'].to_dict()
    }

def build_feature_matrix(df, fitted_theme_encoding, known_theme_cols):
    return shared_build_feature_matrix(df, fitted_theme_encoding, known_theme_cols)

def add_derived_features(df):
    return shared_add_derived_features(df)

def build_preprocessing_spec(version_id, feature_names_in, known_theme_cols,
                             fitted_theme_encoding, numeric_imputation_values,
                             follower_clip_upper):
    return shared_build_preprocessing_spec(
        version_id=version_id,
        feature_names_in=feature_names_in,
        known_theme_cols=known_theme_cols,
        fitted_theme_encoding=fitted_theme_encoding,
        numeric_imputation_values=numeric_imputation_values,
        follower_clip_upper=follower_clip_upper,
        producer='ml_pipeline.train_master_arena',
    )

def apply_numeric_preprocessing(df, follower_clip_upper, numeric_imputation_values):
    return shared_apply_numeric_preprocessing(df, follower_clip_upper, numeric_imputation_values)

def fit_preprocessing_context(train_df):
    # 训练侧拟合预处理真相：供 fold-local CV、selection validation、final showdown 共用
    fitted_theme_encoding = fit_train_owned_theme_encoding(
        train_df,
        GLOBAL_CONFIG['theme_encode_smoothing_weight']
    )
    known_theme_cols = pd.get_dummies(train_df['theme_label'], prefix='theme').columns.tolist()

    X_train = build_feature_matrix(train_df, fitted_theme_encoding, known_theme_cols)
    follower_clip_upper = X_train['follower_count_log'].quantile(0.99)
    if pd.isna(follower_clip_upper):
        follower_clip_upper = 0.0
    follower_clip_upper = float(follower_clip_upper)

    X_train = X_train.copy()
    X_train.loc[:, 'follower_count_log'] = X_train['follower_count_log'].clip(upper=follower_clip_upper)

    impute_cols = ['avg_sentiment', 'visual_brightness', 'visual_saturation', 'cut_frequency', 'audio_bpm']
    numeric_imputation_values = {}
    for col in impute_cols:
        median_val = X_train[col].median()
        if pd.isna(median_val):
            median_val = GLOBAL_CONFIG['default_fallback'].get(col, 0)
        median_val = float(median_val)
        numeric_imputation_values[col] = median_val
        X_train.loc[:, col] = X_train[col].fillna(median_val)

    X_train = add_derived_features(X_train)
    feature_cols = [
        'duration_sec', 'follower_count_log', 'publish_hour',
        'avg_sentiment', 'visual_brightness', 'visual_saturation', 'cut_frequency', 'audio_bpm',
        'theme_encoded',
        'visual_impact', 'sensory_pace', 'sentiment_intensity', 'audio_visual_energy', 'content_density'
    ] + known_theme_cols
    X_train = X_train[feature_cols]

    target_cap_value = pd.Series(train_df['digg_count'].values).quantile(GLOBAL_CONFIG['cap_quantile'])
    if pd.isna(target_cap_value):
        target_cap_value = 0.0

    return {
        'fitted_theme_encoding': fitted_theme_encoding,
        'known_theme_cols': known_theme_cols,
        'numeric_imputation_values': numeric_imputation_values,
        'follower_clip_upper': follower_clip_upper,
        'feature_cols': feature_cols,
        'target_cap_value': float(target_cap_value),
        'X_train_preprocessed': X_train.copy()
    }

def transform_with_preprocessing_context(df, preprocessing_context):
    return shared_transform_with_preprocessing_context(df, preprocessing_context)

def prepare_model_training_data(train_df, eval_df=None):
    # 为某一次训练职责（fold、selection、showdown）准备独立的预处理和缩放结果
    preprocessing_context = fit_preprocessing_context(train_df)
    X_train = preprocessing_context['X_train_preprocessed'].copy()
    y_train_orig_pre = train_df['digg_count'].values
    y_train_orig = np.clip(y_train_orig_pre, 0, preprocessing_context['target_cap_value'])
    y_train = np.log1p(y_train_orig)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    bundle = {
        'preprocessing_context': preprocessing_context,
        'scaler': scaler,
        'X_train': X_train,
        'X_train_scaled': X_train_scaled,
        'y_train_orig': y_train_orig,
        'y_train': y_train
    }

    if eval_df is not None:
        X_eval = transform_with_preprocessing_context(eval_df, preprocessing_context)
        bundle['X_eval'] = X_eval
        bundle['X_eval_scaled'] = scaler.transform(X_eval)
        bundle['y_eval_orig'] = eval_df['digg_count'].values

    return bundle

def calculate_log_space_metrics(y_true_orig, y_pred_log):
    # 统一在对数空间比较验证与最终 showdown，保证指标口径一致
    clipped_pred_log = np.clip(y_pred_log, 0, GLOBAL_CONFIG['clip_max'])
    y_true_log = np.log1p(y_true_orig)
    return {
        'RMSE': float(root_mean_squared_error(y_true_log, clipped_pred_log)),
        'R2': float(r2_score(y_true_log, clipped_pred_log)),
        'MAE': float(mean_absolute_error(y_true_log, clipped_pred_log))
    }

def run_fold_local_cv(model_class, model_params, training_df):
    # fold-local CV：每个 fold 都重新拟合主题编码、数值拟合与缩放，杜绝全块预处理泄露
    n_splits = min(GLOBAL_CONFIG['cv_folds'], len(training_df))
    if n_splits < 2:
        raise ValueError("Not enough samples for fold-local CV.")

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=GLOBAL_CONFIG['random_state'])
    fold_metrics = []

    for fold_train_idx, fold_val_idx in cv.split(training_df):
        fold_train_df = training_df.iloc[fold_train_idx].copy()
        fold_val_df = training_df.iloc[fold_val_idx].copy()

        fold_bundle = prepare_model_training_data(fold_train_df, fold_val_df)
        fold_model = model_class(**model_params)
        fold_model.fit(fold_bundle['X_train_scaled'], fold_bundle['y_train'])
        fold_preds_log = fold_model.predict(fold_bundle['X_eval_scaled'])
        fold_metrics.append(calculate_log_space_metrics(fold_bundle['y_eval_orig'], fold_preds_log))

    return {
        'RMSE': float(np.mean([m['RMSE'] for m in fold_metrics])),
        'R2': float(np.mean([m['R2'] for m in fold_metrics])),
        'MAE': float(np.mean([m['MAE'] for m in fold_metrics]))
    }

def load_champion_metadata_and_artifacts():
    # 加载卫冕冠军及其版本拥有的预处理规格
    try:
        if not os.path.exists(manifest_path):
            return None, None, None, None
            
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
            
        champion_id = manifest.get('current_version')
        if not champion_id:
            return None, None, None, None
            
        bundle_artifacts = manifest.get('bundle_artifacts', {})
        model_file = os.path.join(
            artifacts_dir,
            bundle_artifacts.get('model_file') or f'model_{champion_id}.pkl'
        )
        scaler_file = os.path.join(
            artifacts_dir,
            bundle_artifacts.get('scaler_file') or f'scaler_{champion_id}.pkl'
        )
        
        if not os.path.exists(model_file) or not os.path.exists(scaler_file):
            return None, None, None, None
            
        champion_model = joblib.load(model_file)
        champion_scaler = joblib.load(scaler_file)
        champion_prep_spec = None
        prep_path = resolve_prep_spec_path(artifacts_dir, champion_id, manifest_payload=manifest)
        if prep_path.exists():
            try:
                with open(prep_path, 'r', encoding='utf-8') as f:
                    champion_prep_spec = normalize_preprocessing_spec(
                        json.load(f),
                        version_id=champion_id
                    )
            except Exception as prep_error:
                print(f"[WARN] Failed to load champion preprocessing spec for {champion_id}: {str(prep_error)}")
        return manifest, champion_model, champion_scaler, champion_prep_spec
    except Exception as e:
        # 异常处理保证健壮性
        print(f"[ERROR] Failed to load champion: {str(e)}")
        return None, None, None, None

def filter_rare_themes(df, min_samples):
    # 按主题标签分组，过滤掉样本数不足阈值的稀有主题
    # Strategy anchor (docs/ml_data_strategy.md):
    # training continues to learn from all valid themes; this is data hygiene, not current-theme-only training.
    try:
        theme_counts = df['theme_label'].value_counts()
        rare_themes = theme_counts[theme_counts < min_samples]
        dropped_info = rare_themes.to_dict()
        valid_themes = theme_counts[theme_counts >= min_samples].index.tolist()
        df_filtered = df[df['theme_label'].isin(valid_themes)].copy()
        return df_filtered, dropped_info
    except Exception as e:
        # 异常时返回原始数据，不中断流程
        print(f"[WARN] Rare theme filtering failed: {str(e)}")
        return df.copy(), {}

def identify_active_theme(df):
    # 根据最近一条记录的create_time动态识别活跃主题（主角）
    try:
        df_with_time = df.dropna(subset=['create_time'])
        if df_with_time.empty:
            # 无有效时间戳时回退至样本量最大的主题
            return df['theme_label'].value_counts().idxmax()
        latest_idx = pd.to_datetime(df_with_time['create_time']).idxmax()
        return df.loc[latest_idx, 'theme_label']
    except Exception as e:
        # 异常时回退至众数主题
        print(f"[WARN] Active theme identification failed: {str(e)}")
        return df['theme_label'].value_counts().idxmax()

def build_target_foil_dataset(df, active_theme, target_ratio, random_state):
    # 构建"目标与陪衬"偏置采样数据集
    # Strategy anchor: active_theme may bias sampling focus, but non-current themes remain part of the training distribution.
    try:
        df_target = df[df['theme_label'] == active_theme].copy()
        df_foil = df[df['theme_label'] != active_theme].copy()
        n_target = len(df_target)
        n_foil = len(df_foil)

        if n_target == 0:
            # 目标主题为空时回退至全量数据
            print("[WARN] Active theme has 0 samples after filtering. Using full dataset.")
            return df, n_target, n_foil

        # 计算理想陪衬数量：target占target_ratio => foil_needed = target  (1-ratio) / ratio
        desired_foil = int(n_target * (1.0 - target_ratio) / target_ratio)

        if desired_foil >= n_foil:
            # 陪衬数据不足时使用全部陪衬
            sampled_foil = df_foil
        else:
            # 随机采样陪衬数据
            sampled_foil = df_foil.sample(n=desired_foil, random_state=random_state)

        actual_foil_count = len(sampled_foil)
        df_biased = pd.concat([df_target, sampled_foil], ignore_index=True)
        # 打乱顺序避免模型学到排列偏差
        df_biased = df_biased.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        return df_biased, n_target, actual_foil_count
    except Exception as e:
        # 异常时回退至全量数据
        print(f"[WARN] Biased sampling failed: {str(e)}. Using full dataset.")
        n_t = len(df[df['theme_label'] == active_theme])
        n_f = len(df[df['theme_label'] != active_theme])
        return df, n_t, n_f

def print_statistics_panel(total_valid, active_theme, target_count, foil_count,
                           inner_train_size, selection_valid_size, showdown_test_size,
                           dropped_themes):
    # 渲染训练前数据统计面板
    w = 62
    border = '╔' + '═' * w + '╗'
    mid_sep = '╠' + '═' * w + '╣'
    bottom = '╚' + '═' * w + '╝'
    def row(text):
        return '║ ' + text.ljust(w - 1) + '║'

    lines = [
        border,
        row('  DATA INGESTION STATISTICS PANEL'),
        mid_sep,
        row(f'Total Valid Samples (after filter) : {total_valid}'),
        row(f'Active Target Theme (Protagonist)  : {active_theme}'),
        mid_sep,
        row(f'  Target Theme Count             : {target_count}'),
        row(f'  Foil Themes Count (sampled)     : {foil_count}'),
        row(f'    Biased Dataset Total            : {target_count + foil_count}'),
        mid_sep,
        row(f'Inner Training Set                 : {inner_train_size}'),
        row(f'Selection Validation Set          : {selection_valid_size}'),
        row(f'Final Showdown Test Set           : {showdown_test_size}'),
    ]

    if dropped_themes:
        lines.append(mid_sep)
        lines.append(row('  Dropped Rare Themes (< min_samples):'))
        for theme_name, cnt in dropped_themes.items():
            lines.append(row(f'    • {theme_name} ({cnt} samples)'))

    lines.append(bottom)
    print('\n'.join(lines))

def deploy_new_champion(challenger_model, challenger_scaler, metadata):
    # 原子化部署策略
    try:
        os.makedirs(artifacts_dir, exist_ok=True)
        version_id = metadata['version_id']
        
        new_model_path = os.path.join(artifacts_dir, f'model_{version_id}.pkl')
        new_scaler_path = os.path.join(artifacts_dir, f'scaler_{version_id}.pkl')
        new_prep_path = os.path.join(artifacts_dir, f'prep_{version_id}.json')
        temp_model_path = os.path.join(artifacts_dir, f'model_{version_id}.tmp.pkl')
        temp_scaler_path = os.path.join(artifacts_dir, f'scaler_{version_id}.tmp.pkl')
        temp_prep_path = os.path.join(artifacts_dir, f'prep_{version_id}.tmp.json')
        temp_manifest = os.path.join(artifacts_dir, 'version_manifest_tmp.json')
        preprocessing_spec = metadata.get('preprocessing_spec')

        joblib.dump(challenger_model, temp_model_path)
        joblib.dump(challenger_scaler, temp_scaler_path)
        if preprocessing_spec:
            with open(temp_prep_path, 'w', encoding='utf-8') as f:
                json.dump(preprocessing_spec, f, indent=4, ensure_ascii=False)
        
        prev_version = None
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    old_manifest = json.load(f)
                    prev_version = old_manifest.get('current_version')
            except Exception:
                pass
                
        new_manifest = {
            'current_version': version_id,
            'previous_version': prev_version,
            'updated_at': datetime.datetime.now().isoformat(),
            'best_model': metadata['best_model'],
            'test_rmse': metadata['test_rmse'],
            'test_r2': metadata['test_r2'],
            'hyperparameters': metadata['hyperparameters'],
            'bundle_mode': 'bundle_aware' if preprocessing_spec else 'legacy_compatibility',
            'bundle_artifacts': {
                'version_id': version_id,
                'model_file': os.path.basename(new_model_path),
                'scaler_file': os.path.basename(new_scaler_path),
                'prep_spec_file': os.path.basename(new_prep_path) if preprocessing_spec else None,
                'bundle_complete': bool(preprocessing_spec),
            }
        }

        if 'feature_names_in' in metadata and metadata['feature_names_in']:
            new_manifest['feature_names_in'] = metadata['feature_names_in']
        if 'bayesian_global_mean' in metadata:
            new_manifest['bayesian_global_mean'] = metadata['bayesian_global_mean']
        if preprocessing_spec:
            new_manifest['prep_spec_file'] = os.path.basename(new_prep_path)
            new_manifest['bundle_complete'] = True
            new_manifest['preprocessing_derivation_version'] = preprocessing_spec.get('derivation_version')
            new_manifest['preprocessing_schema_version'] = preprocessing_spec.get('schema_version')
        else:
            new_manifest['bundle_complete'] = False
        
        if 'feature_importances' in metadata and metadata['feature_importances']:
            new_manifest['feature_importances'] = metadata['feature_importances']
        
        with open(temp_manifest, 'w', encoding='utf-8') as f:
            json.dump(new_manifest, f, indent=4)

        shutil.move(temp_model_path, new_model_path)
        shutil.move(temp_scaler_path, new_scaler_path)
        if preprocessing_spec:
            shutil.move(temp_prep_path, new_prep_path)
            print(
                f" [DEPLOY] Persisted preprocessing spec for version '{version_id}': "
                f"{os.path.basename(new_prep_path)} (future consumers should use version-owned preprocessing metadata)."
            )
            
        # 原子替换
        shutil.move(temp_manifest, manifest_path)
        print(
            f" [DEPLOY] New Champion '{metadata['best_model']}' (v: {version_id}) successfully deployed "
            f"with model/scaler{'/prep spec' if preprocessing_spec else ''} artifacts."
        )
        
    except Exception as e:
        # 异常处理保证健壮性
        print(f"[ERROR] Deployment aborted due to error: {str(e)}")
        traceback.print_exc()

# 主流程
def build_master_arena():
    version_id = datetime.datetime.now().strftime("v%Y%m%d_%H%M%S")
    print(f"========== PHASE 2: 3+1 CHAMPION-CHALLENGER ARENA ==========")
    print(f"[INFO] Run Version: {version_id}")
    
    try:
        Video = apps.get_model('douyin_hangxi', 'Video')
        # 保护数据库查询阶段
        queryset = Video.objects.annotate(
            avg_sentiment=Avg('comments__sentiment_score')
        ).values(
            'duration', 'follower_count', 'create_time',
            'collect_count', 'comment_count', 'share_count',
            'digg_count', 'avg_sentiment',
            'visual_brightness', 'visual_saturation', 'cut_frequency', 'audio_bpm',
            'theme_label'
        )
        df = pd.DataFrame(list(queryset))
    except Exception as e:
        print(f"[ERROR] Database query failed: {str(e)}")
        return

    if df.empty:
        print("[ERROR] No data in database. Please import dataset first.")
        return

    print(f"[INFO] Raw samples loaded: {len(df)}")

    try:
        # Stage 1: raw cleaning / parsing
        df['theme_label'] = df['theme_label'].fillna('Unknown')
        df, dropped_themes = filter_rare_themes(df, GLOBAL_CONFIG['min_theme_samples'])
        total_valid = len(df)
        if total_valid == 0:
            print("[ERROR] No valid samples after rare theme filtering.")
            return

        active_theme = identify_active_theme(df)
        # 禁用偏置采样，释放全量 2147 条数据进行全局训练
        df_biased = df.copy()
        target_count = len(df_biased[df_biased['theme_label'] == active_theme])
        foil_count = len(df_biased) - target_count
        df_prepared = prepare_base_training_dataframe(df_biased)

        # Stage 2: outer split for final showdown; inner split for challenger selection
        development_df, showdown_test_df = train_test_split(
            df_prepared, test_size=GLOBAL_CONFIG['test_size'], random_state=GLOBAL_CONFIG['random_state']
        )
        development_df = development_df.reset_index(drop=True)
        showdown_test_df = showdown_test_df.reset_index(drop=True)

        selection_train_df, selection_valid_df = train_test_split(
            development_df,
            test_size=GLOBAL_CONFIG['selection_validation_size'],
            random_state=GLOBAL_CONFIG['random_state']
        )
        selection_train_df = selection_train_df.reset_index(drop=True)
        selection_valid_df = selection_valid_df.reset_index(drop=True)
        development_df = pd.concat([selection_train_df, selection_valid_df], ignore_index=True)

        # 渲染统计面板
        print_statistics_panel(
            total_valid=total_valid,
            active_theme=active_theme,
            target_count=target_count,
            foil_count=foil_count,
            inner_train_size=len(selection_train_df),
            selection_valid_size=len(selection_valid_df),
            showdown_test_size=len(showdown_test_df),
            dropped_themes=dropped_themes
        )
        print("[INFO] Challenger selection uses fold-local CV on the inner training block plus a separate validation split.")
        print("[INFO] Final showdown is reserved for the untouched outer test split only.")

    except Exception as e:
        # 捕获数据预处理及切割过程中的异常
        print(f"[ERROR] Data preprocessing failed: {str(e)}")
        return

    # ---------------- Step 1: Challenger Selection ----------------
    print("\n========== STEP 1: CHALLENGER SELECTION (FOLD-LOCAL CV + CLEAN VALIDATION) ==========")
    print("[INFO] Fold-local CV fits preprocessing inside each fold. Selection validation stays outside CV and outside the final showdown test.")
    
    models_to_train = {
        'RandomForest': {
            'class': RandomForestRegressor,
            'fixed_params': {'random_state': GLOBAL_CONFIG['random_state'], 'n_jobs': -1}
        },
        'LightGBM': {
            'class': lgb.LGBMRegressor,
            'fixed_params': {'objective': 'regression', 'random_state': GLOBAL_CONFIG['random_state'], 'verbosity': -1, 'n_jobs': -1}
        },
        'XGBoost': {
            'class': xgb.XGBRegressor,
            'fixed_params': {
                'objective': 'reg:squarederror',
                'tree_method': 'hist',
                'device': 'cuda',
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': GLOBAL_CONFIG['random_state'],
                'verbosity': 0
            }
        }
    }

    internal_leaderboard = []
    model_selection_results = {}

    try:
        selection_bundle = prepare_model_training_data(selection_train_df, selection_valid_df)
        print(f"[INFO] Challenger-selection feature dimension: {len(selection_bundle['preprocessing_context']['feature_cols'])}")
    except Exception as e:
        print(f"[ERROR] Challenger-selection preprocessing failed: {str(e)}")
        return

    for model_name, config in models_to_train.items():
        print(f"[Training] {model_name}...")
        try:
            param_grid = GLOBAL_CONFIG['param_grids'].get(model_name, {})
            param_list = list(ParameterGrid(param_grid))

            best_cv_metrics = None
            best_params = None
            best_cv_rmse = float('inf')

            for params in param_list:
                combined_params = {**config['fixed_params'], **params}
                cv_metrics = run_fold_local_cv(config['class'], combined_params, selection_train_df)
                if cv_metrics['RMSE'] < best_cv_rmse:
                    best_cv_rmse = cv_metrics['RMSE']
                    best_cv_metrics = cv_metrics
                    best_params = params

            if best_params is None:
                raise ValueError("No valid hyperparameter configuration was selected.")

            final_params = {**config['fixed_params'], **best_params}
            selection_model = config['class'](**final_params)
            selection_model.fit(selection_bundle['X_train_scaled'], selection_bundle['y_train'])

            validation_metrics = calculate_log_space_metrics(
                selection_bundle['y_eval_orig'],
                selection_model.predict(selection_bundle['X_eval_scaled'])
            )

            model_selection_results[model_name] = {
                'params': final_params,
                'cv_metrics': best_cv_metrics,
                'validation_metrics': validation_metrics
            }
            internal_leaderboard.append({
                'Model': model_name,
                'CV_RMSE': best_cv_metrics['RMSE'],
                'CV_R2': best_cv_metrics['R2'],
                'CV_MAE': best_cv_metrics['MAE'],
                'Validation_RMSE': validation_metrics['RMSE'],
                'Validation_R2': validation_metrics['R2'],
                'Validation_MAE': validation_metrics['MAE']
            })
            print(
                f" -> CV RMSE: {best_cv_metrics['RMSE']:.4f} | "
                f"Validation RMSE: {validation_metrics['RMSE']:.4f} | "
                f"Validation MAE: {validation_metrics['MAE']:.4f} | "
                f"Validation R2: {validation_metrics['R2']:.4f}"
            )

        except Exception as e:
            # 异常处理允许个别失败但不整体崩溃
            print(f"[ERROR] {model_name} challenger selection failed: {str(e)}")

    if not internal_leaderboard:
        print("[ERROR] Internal selection failed to produce any models.")
        return

    internal_leaderboard.sort(key=lambda x: x['Validation_RMSE'])
    challenger_name = internal_leaderboard[0]['Model']
    challenger_stats = internal_leaderboard[0]
    challenger_params = model_selection_results[challenger_name]['params']

    print(
        f"\n[WINNER] The Challenger is {challenger_name} "
        f"(selection validation RMSE: {challenger_stats['Validation_RMSE']:.4f})"
    )

    # ---------------- Step 2: Load the Champion ----------------
    print("\n========== STEP 2: LOAD THE CHAMPION (1 GATEKEEPER) ==========")
    manifest, champion_model, champion_scaler, champion_prep_spec = load_champion_metadata_and_artifacts()

    # ---------------- Step 3: The Showdown ----------------
    print("\n========== STEP 3: THE SHOWDOWN (FINAL TEST ONLY) ==========")
    print("[INFO] Final showdown trains the selected challenger on development data and evaluates only on the reserved outer test split.")

    try:
        final_training_bundle = prepare_model_training_data(development_df, showdown_test_df)
        feature_cols = final_training_bundle['preprocessing_context']['feature_cols']
        print(f"[INFO] Final feature dimension: {len(feature_cols)}")
    except Exception as e:
        print(f"[ERROR] Final showdown preprocessing failed: {str(e)}")
        return

    challenger_model = models_to_train[challenger_name]['class'](**challenger_params)
    challenger_model.fit(final_training_bundle['X_train_scaled'], final_training_bundle['y_train'])

    # 提取重要特征用于保存
    feat_importances = {}
    if hasattr(challenger_model, 'feature_importances_'):
        importances = challenger_model.feature_importances_
        if len(importances) == len(feature_cols):
            feat_importances = dict(zip(feature_cols, [float(x) for x in importances]))
    elif hasattr(challenger_model, 'coef_'):
        importances = challenger_model.coef_
        if len(importances) == len(feature_cols):
            feat_importances = dict(zip(feature_cols, [float(x) for x in importances]))

    challenger_metadata = {
        'version_id': version_id,
        'best_model': challenger_name,
        'test_rmse': challenger_stats['Validation_RMSE'],
        'test_r2': challenger_stats['Validation_R2'],
        'hyperparameters': challenger_params,
        'feature_importances': feat_importances
    }
    challenger_metadata['feature_names_in'] = list(final_training_bundle['X_train'].columns)
    challenger_metadata['bayesian_global_mean'] = float(
        final_training_bundle['preprocessing_context']['fitted_theme_encoding']['global_mean']
    )
    challenger_metadata['preprocessing_spec'] = build_preprocessing_spec(
        version_id=version_id,
        feature_names_in=list(final_training_bundle['X_train'].columns),
        known_theme_cols=final_training_bundle['preprocessing_context']['known_theme_cols'],
        fitted_theme_encoding=final_training_bundle['preprocessing_context']['fitted_theme_encoding'],
        numeric_imputation_values=final_training_bundle['preprocessing_context']['numeric_imputation_values'],
        follower_clip_upper=final_training_bundle['preprocessing_context']['follower_clip_upper']
    )

    # 数学公平性：使用各自的数据缩放器，禁止使用同一套防止漂移误差
    try:
        # 挑战者使用在 development data 上重新拟合出的预处理器，并仅在 final showdown test 上评估
        challenger_final_metrics = calculate_log_space_metrics(
            final_training_bundle['y_eval_orig'],
            challenger_model.predict(final_training_bundle['X_eval_scaled'])
        )
        print(
            f" -> Challenger ({challenger_name}) RMSE: {challenger_final_metrics['RMSE']:.4f} | "
            f"MAE: {challenger_final_metrics['MAE']:.4f} | R2: {challenger_final_metrics['R2']:.4f}"
        )

        if not champion_model or not champion_scaler:
            print("[INFO] No existing champion found. Promoting Challenger after clean final-test evaluation.")
            challenger_metadata['test_rmse'] = challenger_final_metrics['RMSE']
            challenger_metadata['test_r2'] = challenger_final_metrics['R2']
            challenger_metadata['test_mae'] = challenger_final_metrics['MAE']
            deploy_new_champion(challenger_model, final_training_bundle['scaler'], challenger_metadata)
            return

        print("[VS] Evaluating Champion vs. Challenger on reserved final showdown test set...")

        # 卫冕冠军必须使用其版本拥有的预处理真值重建 showdown 特征；
        # 绝不再复用 challenger-prepared X_eval。
        champ_expected_features = []
        if hasattr(champion_model, 'feature_names_in_'):
            champ_expected_features = champion_model.feature_names_in_
        elif hasattr(champion_scaler, 'feature_names_in_'):
            champ_expected_features = champion_scaler.feature_names_in_
        else:
            champ_expected_features = feature_cols

        X_test_champ = build_versioned_evaluation_features(
            eval_df=showdown_test_df,
            expected_features=list(champ_expected_features),
            scaler=champion_scaler,
            prep_spec=champion_prep_spec,
            manifest_payload=manifest
        )

        X_test_scaled_champion = champion_scaler.transform(X_test_champ)
        champion_final_metrics = calculate_log_space_metrics(
            final_training_bundle['y_eval_orig'],
            champion_model.predict(X_test_scaled_champion)
        )
        
    except Exception as e:
        print(f"[ERROR] Showdown evaluation failed: {str(e)}")
        traceback.print_exc()
        # 如若公平测试失败，安全起见停止部署
        print(" System retaining existing Champion due to evaluation error.")
        return

    print(
        f" -> Champion (v: {manifest.get('current_version')}) RMSE: {champion_final_metrics['RMSE']:.4f} | "
        f"MAE: {champion_final_metrics['MAE']:.4f} | R2: {champion_final_metrics['R2']:.4f}"
    )

    # ---------------- Step 4: Atomic Deployment & Telemetry ----------------
    margin = GLOBAL_CONFIG['win_margin']
    if challenger_final_metrics['RMSE'] <= (champion_final_metrics['RMSE'] - margin):
        print(f"\n[VICTORY] Challenger outperformed Champion by at least {margin}! Proceeding to deployment.")
        # 实时以最终评估成绩覆盖元数据
        challenger_metadata['test_rmse'] = challenger_final_metrics['RMSE']
        challenger_metadata['test_r2'] = challenger_final_metrics['R2']
        challenger_metadata['test_mae'] = challenger_final_metrics['MAE']
        deploy_new_champion(challenger_model, final_training_bundle['scaler'], challenger_metadata)
    else:
        print(f"\n Challenger failed. System retaining existing Champion.")

if __name__ == '__main__':
    build_master_arena()

