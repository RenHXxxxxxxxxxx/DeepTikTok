import os
import sys
import json
import time
import django
import pandas as pd
import joblib
import numpy as np
import datetime
from django.db.models import Avg
from sklearn.model_selection import train_test_split, ParameterGrid, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, r2_score, root_mean_squared_error,
    median_absolute_error, explained_variance_score, mean_absolute_percentage_error
)
import lightgbm as lgb
from sklearn.inspection import permutation_importance
from tqdm import tqdm

# *核心路径解析与Django环境配置*
current_dir = os.path.dirname(os.path.abspath(__file__))
outer_root = os.path.dirname(os.path.dirname(current_dir))  # *[重构] 现位于 ml_pipeline/trainers/，需上两级至项目根*
django_inner_root = os.path.join(outer_root, 'renhangxi_tiktok_bysj')

if outer_root in sys.path:
    sys.path.remove(outer_root)
if django_inner_root not in sys.path:
    sys.path.insert(0, django_inner_root)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'settings')
try:
    django.setup()
except Exception as e:
    # *保护环境初始化*
    print(f"[ERROR] Django setup failed: {str(e)}")

from django.apps import apps
Video = apps.get_model('douyin_hangxi', 'Video')

# *全局配置项：避免魔法数字硬编码*
GLOBAL_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'cap_quantile': 0.95,
    'cv_folds': 5,
    'n_jobs': 1,
    'clip_max': 20,
    'default_fallback': {
        'avg_sentiment': 0.5,
        'visual_brightness': 128,
        'visual_saturation': 100,
        'cut_frequency': 0.5,
        'audio_bpm': 110
    },
    'param_grid': {
        'n_estimators': [300, 500, 1000],
        'max_depth': [5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1]
    }
}

# *基础工具函数*
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
        # *异常处理保证健壮性*
        return 0

# *主流程*
def build_multimodal_model():
    version_id = datetime.datetime.now().strftime("v%Y%m%d_%H%M%S")
    print(f"[INFO] Step 1/5: Starting training pipeline | Version: {version_id}")
    print(f"[INFO] Mode: LightGBM (Leaf-wise Boosting)")

    try:
        # *保护数据库查询阶段*
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
        print("[ERROR] No data in database. Please import multimodal dataset first.")
        return

    print(f"[INFO] Step 2/5: Data preprocessing (samples: {len(df)})...")

    try:
        df['duration_sec'] = df['duration'].apply(convert_duration_to_seconds)
        df['publish_hour'] = pd.to_datetime(df['create_time']).dt.hour

        # *处理离散主题特征，依据要求保留get_dummies形式不使用OneHotEncoder*
        print(f"[INFO] Step 2.1/5: Processing Theme Labels...")
        df['theme_label'] = df['theme_label'].fillna('Unknown')
        theme_dummies = pd.get_dummies(df['theme_label'], prefix='theme')
        known_theme_cols = theme_dummies.columns.tolist()
        print(f"[INFO] Generated {len(known_theme_cols)} discrete theme features.")
        
        df = pd.concat([df, theme_dummies], axis=1)

        # *聚合所有基础特征（不含标签）*
        base_features = [
            'duration_sec', 'follower_count', 'publish_hour',
            'avg_sentiment', 'visual_brightness', 'visual_saturation', 
            'cut_frequency', 'audio_bpm'
        ] + known_theme_cols

        X = df[base_features].copy()
        y_original = df['digg_count'].values

        # *在任何特征计算、截断、缺失值处理之前先拆分数据集，杜绝数据泄露*
        X_train, X_test, y_train_orig_pre, y_test_orig = train_test_split(
            X, y_original, test_size=GLOBAL_CONFIG['test_size'], random_state=GLOBAL_CONFIG['random_state']
        )
        
        # *拷贝以防止视图修改警告*
        X_train = X_train.copy()
        X_test = X_test.copy()

        # *仅在训练集上计算阈值并应用截断，防止目标变量泄露*
        cap_value = pd.Series(y_train_orig_pre).quantile(GLOBAL_CONFIG['cap_quantile'])
        y_train_orig = np.clip(y_train_orig_pre, 0, cap_value)
        print(f"[INFO] Outlier Cap Applied ONLY to Train: digg_count capped at {cap_value:.0f}")

        y_train = np.log1p(y_train_orig)
        y_test = np.log1p(y_test_orig)

        # *仅在训练集上计算中位数并在训练和测试集进行插补，防止特征泄露*
        impute_cols = ['avg_sentiment', 'visual_brightness', 'visual_saturation', 'cut_frequency', 'audio_bpm']
        for col in impute_cols:
            median_val = X_train[col].median()
            if pd.isna(median_val):
                # *兜底保护*
                median_val = GLOBAL_CONFIG['default_fallback'].get(col, 0)
            X_train.loc[:, col] = X_train[col].fillna(median_val)
            X_test.loc[:, col] = X_test[col].fillna(median_val)

        # *衍生特征生成，分别在训练与测试集上独立执行确保无泄露*
        for ds in [X_train, X_test]:
            ds.loc[:, 'visual_impact'] = (ds['visual_brightness'] * ds['visual_saturation']) / 1000.0
            ds.loc[:, 'sensory_pace'] = ds['audio_bpm'] * ds['cut_frequency']
            ds.loc[:, 'sentiment_intensity'] = abs(ds['avg_sentiment'] - 0.5) * 2
            ds.loc[:, 'audio_visual_energy'] = ds['visual_brightness'] * ds['audio_bpm'] / 1000.0
            ds.loc[:, 'content_density'] = ds['cut_frequency'] / (ds['duration_sec'] + 1)

        feature_cols = [
            'duration_sec', 'follower_count', 'publish_hour',
            'avg_sentiment', 'visual_brightness', 'visual_saturation', 'cut_frequency', 'audio_bpm',
            'visual_impact', 'sensory_pace', 'sentiment_intensity', 'audio_visual_energy', 'content_density'
        ] + known_theme_cols

        X_train = X_train[feature_cols]
        X_test = X_test[feature_cols]

        print(f"[INFO] Final feature dimension: {len(feature_cols)}")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

    except Exception as e:
        # *捕获数据预处理及切割过程中的异常*
        print(f"[ERROR] Data preprocessing failed: {str(e)}")
        return

    print("[INFO] Step 3/5: GridSearchCV hyperparameter optimization (LightGBM)...")

    try:
        param_list = list(ParameterGrid(GLOBAL_CONFIG['param_grid']))
        print(f"[INFO] Total hyperparameter combinations to evaluate: {len(param_list)}")

        best_score = -float('inf')
        best_params = None
        best_cv_std = 0.0

        # *记录训练开始时间*
        train_start = time.time()

        # *清洁的tqdm原生循环遍历参数网格*
        for params in tqdm(param_list, desc="Hyperparameter Tuning", ncols=100, colour='green', leave=True):
            temp_model = lgb.LGBMRegressor(objective='regression', random_state=GLOBAL_CONFIG['random_state'], verbosity=-1, n_jobs=-1, **params)
            # *该参数组合的5折交叉验证*
            scores = cross_val_score(temp_model, X_train_scaled, y_train,
                                     cv=GLOBAL_CONFIG['cv_folds'], scoring='r2',
                                     n_jobs=GLOBAL_CONFIG['n_jobs'])
            mean_score = scores.mean()

            if mean_score > best_score:
                best_score = mean_score
                best_cv_std = scores.std()
                best_params = params

        print(f"\n[RESULT] Best Parameters Found:")
        for param, value in best_params.items():
            print(f"   - {param}: {value}")
        print(f"[RESULT] Best CV R2 Score: {best_score:.4f} (±{best_cv_std:.4f})")

        print("[INFO] Training final model with best parameters...")
        model = lgb.LGBMRegressor(objective='regression', random_state=GLOBAL_CONFIG['random_state'], verbosity=-1, n_jobs=-1, **best_params)
        model.fit(X_train_scaled, y_train)

        # *记录训练总耗时*
        train_elapsed = time.time() - train_start
        print(f"[INFO] Total training wall-clock time: {train_elapsed:.1f}s")

    except Exception as e:
        # *保护训练过程与网格搜索*
        print(f"[ERROR] Model training failed: {str(e)}")
        return

    print("\n[INFO] Step 4/5: Model evaluation and diagnostics...")

    try:
        y_pred_log = model.predict(X_test_scaled)
        
        # *安全阀：裁剪对数预测，防止指数爆炸*
        y_pred_log = np.clip(y_pred_log, 0, GLOBAL_CONFIG['clip_max'])
        y_pred_original = np.expm1(y_pred_log)

        # *核心回归指标计算*
        r2_log_space = r2_score(y_test, y_pred_log)
        r2_original_space = r2_score(y_test_orig, y_pred_original)
        mae_original = mean_absolute_error(y_test_orig, y_pred_original)
        rmse_original = root_mean_squared_error(y_test_orig, y_pred_original)
        median_ae_original = median_absolute_error(y_test_orig, y_pred_original)
        evs_log = explained_variance_score(y_test, y_pred_log)

        # *MAPE安全计算：过滤零值样本防止除零*
        nonzero_mask = y_test_orig != 0
        if nonzero_mask.sum() > 0:
            mape_original = mean_absolute_percentage_error(
                y_test_orig[nonzero_mask], y_pred_original[nonzero_mask]
            )
        else:
            mape_original = float('nan')

        # *汇总指标字典，供后续持久化使用*
        metrics_dict = {
            'r2_log': round(float(r2_log_space), 4),
            'r2_original': round(float(r2_original_space), 4),
            'mae': round(float(mae_original), 2),
            'rmse': round(float(rmse_original), 2),
            'median_ae': round(float(median_ae_original), 2),
            'mape': round(float(mape_original), 4) if not np.isnan(mape_original) else None,
            'explained_variance': round(float(evs_log), 4),
            'cv_r2_mean': round(float(best_score), 4),
            'cv_r2_std': round(float(best_cv_std), 4),
            'training_time_seconds': round(float(train_elapsed), 1),
        }

        print(f"   - Log-Space R2 Score:        {r2_log_space:.4f}")
        print(f"   - Original-Space R2 Score:   {r2_original_space:.4f}")
        print(f"   - MAE:                       {mae_original:.2f} diggs")
        print(f"   - RMSE:                      {rmse_original:.2f} diggs")
        print(f"   - Median AE:                 {median_ae_original:.2f} diggs")
        print(f"   - MAPE:                      {mape_original:.2%}" if not np.isnan(mape_original) else "   - MAPE:                      N/A (zero values in test set)")
        print(f"   - Explained Variance (log):  {evs_log:.4f}")
        print(f"   - CV R2 (mean ± std):        {best_score:.4f} ± {best_cv_std:.4f}")

        print("\n[RESULT] Feature Importance Ranking (Top 10 via Permutation):")
        # Use Permutation Importance for unbiased evaluation on the test set
        perm_result = permutation_importance(model, X_test_scaled, y_test, n_repeats=10, random_state=GLOBAL_CONFIG['random_state'], n_jobs=GLOBAL_CONFIG['n_jobs'])

        feature_importances = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': perm_result.importances_mean
        }).sort_values('Importance', ascending=False)

        # Normalize to 0-1 for consistent bar visualization
        max_imp = feature_importances['Importance'].max()
        if max_imp > 0:
            feature_importances['Importance'] = feature_importances['Importance'] / max_imp

        for idx, row in feature_importances.head(10).iterrows():
            bar_length = int(row['Importance'] * 50)
            bar = '█' * bar_length
            print(f"   {row['Feature']:25s} | {bar} {row['Importance']:.4f}")

    except Exception as e:
        # *评估过程保护*
        print(f"[ERROR] Model evaluation failed: {str(e)}")
        return

    print(f"\n[INFO] Step 5/5: Persisting model assets...")
    try:
        assets_dir = os.path.join(outer_root, 'ml_pipeline', 'artifacts')  # *[重构] 产物统一输出至 ml_pipeline/artifacts/*
        if not os.path.exists(assets_dir):
            os.makedirs(assets_dir)

        # *共用时间戳保证模型验证匹配*
        model_path = os.path.join(assets_dir, f'model_LGBM_{version_id}.pkl')
        scaler_path = os.path.join(assets_dir, f'scaler_LGBM_{version_id}.pkl')

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        # *维护版本清单供服务端读取，含完整指标用于多模型横向对比*
        manifest = {
            'active_version': version_id,
            'model_type': 'LightGBM',
            'model_file': os.path.basename(model_path),
            'scaler_file': os.path.basename(scaler_path),
            'trained_at': datetime.datetime.now().isoformat(),
            'feature_count': len(feature_cols),
            'best_params': best_params,
            'metrics': metrics_dict,
        }
        manifest_path = os.path.join(assets_dir, 'version_manifest.json')
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        print(f"[SUCCESS] 模型权重已保存: {os.path.basename(model_path)}")
        print(f"[SUCCESS] 归一化器已保存: {os.path.basename(scaler_path)}")
        print(f"[SUCCESS] 版本清单已写入: version_manifest.json  (激活版本: {version_id})")
        print(f"[INFO] 主题列信息已内嵌于模型 feature_names 中，无需单独文件")
        print(f"\n[DONE] Training pipeline completed. LightGBM model ready.")

    except Exception as e:
        # *落地模型发生故障的阻截*
        print(f"[ERROR] Asset persisting failed: {str(e)}")
        return

if __name__ == '__main__':
    build_multimodal_model()
