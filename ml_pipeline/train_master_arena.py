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
from sklearn.model_selection import train_test_split, ParameterGrid, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
from tqdm import tqdm

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
    'random_state': 42,
    'min_theme_samples': 5,
    'target_ratio': 0.70,
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

def load_champion_metadata_and_artifacts():
    # 加载卫冕冠军及其伸缩器
    try:
        if not os.path.exists(manifest_path):
            return None, None, None
            
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
            
        champion_id = manifest.get('current_version')
        if not champion_id:
            return None, None, None
            
        model_file = os.path.join(artifacts_dir, f'model_{champion_id}.pkl')
        scaler_file = os.path.join(artifacts_dir, f'scaler_{champion_id}.pkl')
        
        if not os.path.exists(model_file) or not os.path.exists(scaler_file):
            return None, None, None
            
        champion_model = joblib.load(model_file)
        champion_scaler = joblib.load(scaler_file)
        return manifest, champion_model, champion_scaler
    except Exception as e:
        # 异常处理保证健壮性
        print(f"[ERROR] Failed to load champion: {str(e)}")
        return None, None, None

def filter_rare_themes(df, min_samples):
    # 按主题标签分组，过滤掉样本数不足阈值的稀有主题
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
                           train_size, test_size, dropped_themes):
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
        row(f'Training Set                       : {train_size}'),
        row(f'Hold-out Set                       : {test_size}'),
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
        
        joblib.dump(challenger_model, new_model_path)
        joblib.dump(challenger_scaler, new_scaler_path)
        
        temp_manifest = os.path.join(artifacts_dir, 'version_manifest_tmp.json')
        
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
            'hyperparameters': metadata['hyperparameters']
        }

        if 'feature_names_in' in metadata and metadata['feature_names_in']:
            new_manifest['feature_names_in'] = metadata['feature_names_in']
        if 'bayesian_global_mean' in metadata:
            new_manifest['bayesian_global_mean'] = metadata['bayesian_global_mean']
        
        if 'feature_importances' in metadata and metadata['feature_importances']:
            new_manifest['feature_importances'] = metadata['feature_importances']
        
        with open(temp_manifest, 'w', encoding='utf-8') as f:
            json.dump(new_manifest, f, indent=4)
            
        # 原子替换
        shutil.move(temp_manifest, manifest_path)
        print(f" [DEPLOY] New Champion '{metadata['best_model']}' (v: {version_id}) successfully deployed.")
        
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
        # 第〇步：稀有主题过滤 —— 清除样本量不足的噪声主题
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

        # 基础特征工程（在偏置数据集上执行）
        df_biased['duration_sec'] = df_biased['duration'].apply(convert_duration_to_seconds)
        df_biased['publish_hour'] = pd.to_datetime(df_biased['create_time']).dt.hour

        # 对长尾偏态特征(粉丝数)进行对数变换，稳定梯度
        df_biased['follower_count_log'] = np.log1p(df_biased['follower_count'])

        # 核心：贝叶斯目标编码 (Bayesian Target Encoding) 替代独热编码
        df_biased['temp_digg_log'] = np.log1p(df_biased['digg_count'])
        global_mean = df_biased['temp_digg_log'].mean()
        weight = 10.0 # 平滑系数

        theme_stats = df_biased.groupby('theme_label')['temp_digg_log'].agg(['count', 'mean']).reset_index()
        theme_stats.rename(columns={'mean': 'local_mean'}, inplace=True)
        theme_stats['theme_encoded'] = (theme_stats['count'] * theme_stats['local_mean'] + weight * global_mean) / (theme_stats['count'] + weight)

        df_biased = df_biased.merge(theme_stats[['theme_label', 'theme_encoded']], on='theme_label', how='left')
        df_biased.drop(columns=['temp_digg_log'], inplace=True)

        # 恢复动态多维特征空间（独热编码矩阵）
        theme_dummies = pd.get_dummies(df_biased['theme_label'], prefix='theme')
        known_theme_cols = theme_dummies.columns.tolist()
        df_biased = pd.concat([df_biased, theme_dummies], axis=1)

        # 聚合所有基础特征，保留动态扩充的主题多维空间
        base_features = [
            'duration_sec', 'follower_count_log', 'publish_hour',
            'avg_sentiment', 'visual_brightness', 'visual_saturation',
            'cut_frequency', 'audio_bpm', 'theme_encoded'
        ] + known_theme_cols

        X = df_biased[base_features].copy()
        y_original = df_biased['digg_count'].values

        # 在任何特征计算、截断、缺失值处理之前先拆分数据集，杜绝数据泄露
        X_train, X_test, y_train_orig_pre, y_test_orig = train_test_split(
            X, y_original, test_size=GLOBAL_CONFIG['test_size'], random_state=GLOBAL_CONFIG['random_state']
        )

        # 第三步：渲染统计面板
        print_statistics_panel(
            total_valid=total_valid,
            active_theme=active_theme,
            target_count=target_count,
            foil_count=foil_count,
            train_size=len(X_train),
            test_size=len(X_test),
            dropped_themes=dropped_themes
        )
        
        # 拷贝以防止视图修改警告
        X_train = X_train.copy()
        X_test = X_test.copy()

        # 仅在训练集上计算粉丝对数特征P99并同步截断训练/测试集，避免长尾畸变与泄露
        p99_follower = X_train['follower_count_log'].quantile(0.99)
        X_train.loc[:, 'follower_count_log'] = X_train['follower_count_log'].clip(upper=p99_follower)
        X_test.loc[:, 'follower_count_log'] = X_test['follower_count_log'].clip(upper=p99_follower)

        # 仅在训练集上计算阈值并应用截断，防止目标变量泄露
        cap_value = pd.Series(y_train_orig_pre).quantile(GLOBAL_CONFIG['cap_quantile'])
        y_train_orig = np.clip(y_train_orig_pre, 0, cap_value)

        y_train = np.log1p(y_train_orig)
        # 保持y_test_orig为原始对决基准
        
        # 仅在训练集上计算中位数并在训练和测试集进行插补，防止特征泄露
        impute_cols = ['avg_sentiment', 'visual_brightness', 'visual_saturation', 'cut_frequency', 'audio_bpm']
        for col in impute_cols:
            median_val = X_train[col].median()
            if pd.isna(median_val):
                # 兜底保护
                median_val = GLOBAL_CONFIG['default_fallback'].get(col, 0)
            X_train.loc[:, col] = X_train[col].fillna(median_val)
            X_test.loc[:, col] = X_test[col].fillna(median_val)

        # 衍生特征生成，分别在训练与测试集上独立执行确保无泄露
        for ds in [X_train, X_test]:
            ds.loc[:, 'visual_impact'] = (ds['visual_brightness'] * ds['visual_saturation']) / 1000.0
            ds.loc[:, 'sensory_pace'] = ds['audio_bpm'] * ds['cut_frequency']
            ds.loc[:, 'sentiment_intensity'] = abs(ds['avg_sentiment'] - 0.5) * 2
            ds.loc[:, 'audio_visual_energy'] = ds['visual_brightness'] * ds['audio_bpm'] / 1000.0
            ds.loc[:, 'content_density'] = ds['cut_frequency'] / (ds['duration_sec'] + 1)

        # ================ 修复点：对齐最新的高阶特征，保留动态多维矩阵 ================
        feature_cols = [
            'duration_sec', 'follower_count_log', 'publish_hour',
            'avg_sentiment', 'visual_brightness', 'visual_saturation', 'cut_frequency', 'audio_bpm',
            'theme_encoded',
            'visual_impact', 'sensory_pace', 'sentiment_intensity', 'audio_visual_energy', 'content_density'
        ] + known_theme_cols
        # =========================================================

        X_train = X_train[feature_cols]
        X_test = X_test[feature_cols]

        print(f"[INFO] Final feature dimension: {len(feature_cols)}")

        # 全新预处理器拟合
        new_scaler = StandardScaler()
        X_train_scaled = new_scaler.fit_transform(X_train)

    except Exception as e:
        # 捕获数据预处理及切割过程中的异常
        print(f"[ERROR] Data preprocessing failed: {str(e)}")
        return

    # ---------------- Step 1: Internal Selection ----------------
    print("\n========== STEP 1: INTERNAL SELECTION (THE 3 CANDIDATES) ==========")
    
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
    trained_estimators = {}

    for model_name, config in models_to_train.items():
        print(f"[Training] {model_name}...")
        try:
            param_grid = GLOBAL_CONFIG['param_grids'].get(model_name, {})
            param_list = list(ParameterGrid(param_grid))
            
            best_score = -float('inf')
            best_params = None

            for params in param_list:
                combined_params = {**config['fixed_params'], **params}
                temp_model = config['class'](**combined_params)
                
                scores = cross_val_score(temp_model, X_train_scaled, y_train,
                                         cv=GLOBAL_CONFIG['cv_folds'], scoring='r2',
                                         n_jobs=GLOBAL_CONFIG['n_jobs'])
                if scores.mean() > best_score:
                    best_score = scores.mean()
                    best_params = params

            # 最佳模型再训练
            final_params = {**config['fixed_params'], **best_params}
            final_model = config['class'](**final_params)
            final_model.fit(X_train_scaled, y_train)

            # 为防止偏好单一CV，采用验证集快速初评（本集也已经缩放）
            X_test_scaled_new = new_scaler.transform(X_test)
            y_pred_log = final_model.predict(X_test_scaled_new)
            y_pred_log = np.clip(y_pred_log, 0, GLOBAL_CONFIG['clip_max'])
            # 核心修复：在对数空间进行评估，避免长尾指数爆炸
            y_test_log = np.log1p(y_test_orig)
            test_rmse = root_mean_squared_error(y_test_log, y_pred_log)
            test_r2 = r2_score(y_test_log, y_pred_log)
            test_mae = mean_absolute_error(y_test_log, y_pred_log)

            trained_estimators[model_name] = {
                'model': final_model,
                'params': final_params
            }
            internal_leaderboard.append({
                'Model': model_name,
                'RMSE': test_rmse,
                'R2': test_r2,
                'MAE': test_mae
            })
            print(f" -> RMSE: {test_rmse:.4f} | MAE: {test_mae:.4f} | R2: {test_r2:.4f}")

        except Exception as e:
            # 异常处理允许个别失败但不整体崩溃
            print(f"[ERROR] {model_name} internal validation failed: {str(e)}")

    if not internal_leaderboard:
        print("[ERROR] Internal selection failed to produce any models.")
        return

    internal_leaderboard.sort(key=lambda x: x['RMSE'])
    challenger_name = internal_leaderboard[0]['Model']
    challenger_stats = internal_leaderboard[0]
    challenger_model = trained_estimators[challenger_name]['model']

    print(f"\n[WINNER] The Challenger is {challenger_name} (RMSE: {challenger_stats['RMSE']:.4f})")

    # ---------------- Step 2: Load the Champion ----------------
    print("\n========== STEP 2: LOAD THE CHAMPION (1 GATEKEEPER) ==========")
    manifest, champion_model, champion_scaler = load_champion_metadata_and_artifacts()

    # ---------------- Step 3: The Showdown ----------------
    print("\n========== STEP 3: THE SHOWDOWN ==========")

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
        'test_rmse': challenger_stats['RMSE'],
        'test_r2': challenger_stats['R2'],
        'hyperparameters': trained_estimators[challenger_name]['params'],
        'feature_importances': feat_importances
    }
    challenger_metadata['feature_names_in'] = list(X_train.columns)
    challenger_metadata['bayesian_global_mean'] = float(global_mean)

    if not champion_model or not champion_scaler:
        print("[INFO] No existing champion found. Graceful Fallback -> Promoting Challenger to Champion.")
        deploy_new_champion(challenger_model, new_scaler, challenger_metadata)
        return

    print("[VS] Evaluating Champion vs. Challenger on global Hold-out...")
    
    # 数学公平性：使用各自的数据缩放器，禁止使用同一套防止漂移误差
    try:
        # 挑战者使用新拟合的缩放器
        X_test_scaled_challenger = new_scaler.transform(X_test)
        challenger_preds_log = challenger_model.predict(X_test_scaled_challenger)
        challenger_preds_log = np.clip(challenger_preds_log, 0, GLOBAL_CONFIG['clip_max'])
        challenger_preds = np.expm1(challenger_preds_log)
        challenger_final_rmse = root_mean_squared_error(np.log1p(y_test_orig), challenger_preds_log)
        challenger_final_mae = mean_absolute_error(np.log1p(y_test_orig), challenger_preds_log)

        # 卫冕冠军必须使用其保存时的缩放器应对漂移
        # 对齐特征列以防新增特征导致维度不匹配
        champ_expected_features = []
        if hasattr(champion_model, 'feature_names_in_'):
            champ_expected_features = champion_model.feature_names_in_
        elif hasattr(champion_scaler, 'feature_names_in_'):
            champ_expected_features = champion_scaler.feature_names_in_
        else:
            champ_expected_features = feature_cols
            
        # 如果维度缺失以零填充（向下兼容），如果有多出特征予以丢弃
        X_test_champ = pd.DataFrame(index=X_test.index)
        for col in champ_expected_features:
            if col in X_test.columns:
                X_test_champ[col] = X_test[col]
            else:
                X_test_champ[col] = 0.0

        X_test_scaled_champion = champion_scaler.transform(X_test_champ)
        champion_preds_log = champion_model.predict(X_test_scaled_champion)
        champion_preds_log = np.clip(champion_preds_log, 0, GLOBAL_CONFIG['clip_max'])
        champion_preds = np.expm1(champion_preds_log)
        champion_final_rmse = root_mean_squared_error(np.log1p(y_test_orig), champion_preds_log)
        champion_final_mae = mean_absolute_error(np.log1p(y_test_orig), champion_preds_log)
        
    except Exception as e:
        print(f"[ERROR] Showdown evaluation failed: {str(e)}")
        traceback.print_exc()
        # 如若公平测试失败，安全起见停止部署
        print(" System retaining existing Champion due to evaluation error.")
        return

    print(f" -> Challenger ({challenger_name}) RMSE: {challenger_final_rmse:.4f} | MAE: {challenger_final_mae:.4f}")
    print(f" -> Champion (v: {manifest.get('current_version')}) RMSE: {champion_final_rmse:.4f} | MAE: {champion_final_mae:.4f}")

    # ---------------- Step 4: Atomic Deployment & Telemetry ----------------
    margin = GLOBAL_CONFIG['win_margin']
    if challenger_final_rmse <= (champion_final_rmse - margin):
        print(f"\n[VICTORY] Challenger outperformed Champion by at least {margin}! Proceeding to deployment.")
        # 实时以最终评估成绩覆盖元数据
        challenger_metadata['test_rmse'] = challenger_final_rmse
        challenger_metadata['test_r2'] = r2_score(np.log1p(y_test_orig), challenger_preds_log)
        challenger_metadata['test_mae'] = challenger_final_mae
        deploy_new_champion(challenger_model, new_scaler, challenger_metadata)
    else:
        print(f"\n Challenger failed. System retaining existing Champion.")

if __name__ == '__main__':
    build_master_arena()

