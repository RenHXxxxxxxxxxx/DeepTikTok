import os
import sys
import json
import joblib
import django
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from django.db.models import Avg
from sklearn.model_selection import train_test_split

# *全局配置项：避免硬编码*
GLOBAL_CONFIG = {
    'test_size': 0.15,
    'random_state': 42,
    'cap_quantile': 0.95,
    'clip_max': 20,
    'min_theme_samples': 5,
    'font_family': ['SimHei', 'Microsoft YaHei', 'sans-serif'],
    'charts_dir': r'D:\renhangxi_tiktok_bysj\chartsofmodels',
    'default_fallback': {
        'avg_sentiment': 0.5,
        'visual_brightness': 128,
        'visual_saturation': 100,
        'cut_frequency': 0.5,
        'audio_bpm': 110
    }
}

# *配置Matplotlib字体以支持中文*
plt.rcParams['font.sans-serif'] = GLOBAL_CONFIG['font_family']
plt.rcParams['axes.unicode_minus'] = False

def filter_rare_themes(df, min_samples):
    # *过滤掉样本量低于阈值的稀有主题*
    try:
        theme_counts = df['theme_label'].value_counts()
        valid_themes = theme_counts[theme_counts >= min_samples].index.tolist()
        return df[df['theme_label'].isin(valid_themes)].copy()
    except Exception:
        # *保护错误返回原数据*
        return df.copy()

def convert_duration_to_seconds(duration_str):
    # *视频时长字符串转换为秒数*
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

def generate_evaluation_charts():
    # *主流程：读取模型、准备数据、生成图表*
    current_dir = os.path.dirname(os.path.abspath(__file__))
    outer_root = os.path.dirname(current_dir)
    django_inner_root = os.path.join(outer_root, 'renhangxi_tiktok_bysj')
    
    if outer_root in sys.path:
        sys.path.remove(outer_root)
    if django_inner_root not in sys.path:
        sys.path.insert(0, django_inner_root)

    # *初始化Django环境*
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'settings')
    try:
        django.setup()
    except Exception as e:
        print(f"[ERROR] Django setup failed: {str(e)}")

    from django.apps import apps
    
    # *加载资产*
    artifacts_dir = os.path.join(outer_root, 'artifacts')
    manifest_path = os.path.join(artifacts_dir, 'version_manifest.json')
    
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        current_version = manifest.get('current_version')
        if not current_version:
            return
            
        model_path = os.path.join(artifacts_dir, f'model_{current_version}.pkl')
        scaler_path = os.path.join(artifacts_dir, f'scaler_{current_version}.pkl')
        
        champion_model = joblib.load(model_path)
        champion_scaler = joblib.load(scaler_path)
    except Exception as e:
        print(f"[ERROR] Failed to load artifacts: {str(e)}")
        return

    # *准备数据与执行推断*
    try:
        Video = apps.get_model('douyin_hangxi', 'Video')
        # *保护数据库查询操作*
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
        
        df['theme_label'] = df['theme_label'].fillna('Unknown')
        df = filter_rare_themes(df, GLOBAL_CONFIG['min_theme_samples'])
        
        df['duration_sec'] = df['duration'].apply(convert_duration_to_seconds)
        df['publish_hour'] = pd.to_datetime(df['create_time']).dt.hour
        
        # *执行与训练完全一致的特征工程逻辑*
        df['follower_count_log'] = np.log1p(df['follower_count'])
        df['temp_digg_log'] = np.log1p(df['digg_count'])
        global_mean = df['temp_digg_log'].mean()
        weight = 10.0
        
        theme_stats = df.groupby('theme_label')['temp_digg_log'].agg(['count', 'mean']).reset_index()
        theme_stats.rename(columns={'mean': 'local_mean'}, inplace=True)
        theme_stats['theme_encoded'] = (theme_stats['count'] * theme_stats['local_mean'] + weight * global_mean) / (theme_stats['count'] + weight)
        
        df = df.merge(theme_stats[['theme_label', 'theme_encoded']], on='theme_label', how='left')
        df.drop(columns=['temp_digg_log'], inplace=True)
        
        theme_dummies = pd.get_dummies(df['theme_label'], prefix='theme')
        known_theme_cols = theme_dummies.columns.tolist()
        df = pd.concat([df, theme_dummies], axis=1)
        
        base_features = [
            'duration_sec', 'follower_count_log', 'publish_hour',
            'avg_sentiment', 'visual_brightness', 'visual_saturation',
            'cut_frequency', 'audio_bpm', 'theme_encoded'
        ] + known_theme_cols
        
        X = df[base_features].copy()
        y_original = df['digg_count'].values
        
        # *切分出测试集以供绘制实际与预测散点图*
        X_train, X_test, y_train_orig_pre, y_test_orig = train_test_split(
            X, y_original, test_size=GLOBAL_CONFIG['test_size'], random_state=GLOBAL_CONFIG['random_state']
        )
        
        X_train = X_train.copy()
        X_test = X_test.copy()
        
        impute_cols = ['avg_sentiment', 'visual_brightness', 'visual_saturation', 'cut_frequency', 'audio_bpm']
        for col in impute_cols:
            median_val = X_train[col].median()
            if pd.isna(median_val):
                median_val = GLOBAL_CONFIG['default_fallback'].get(col, 0)
            X_train.loc[:, col] = X_train[col].fillna(median_val)
            X_test.loc[:, col] = X_test[col].fillna(median_val)
            
        for ds in [X_train, X_test]:
            ds.loc[:, 'visual_impact'] = (ds['visual_brightness'] * ds['visual_saturation']) / 1000.0
            ds.loc[:, 'sensory_pace'] = ds['audio_bpm'] * ds['cut_frequency']
            ds.loc[:, 'sentiment_intensity'] = abs(ds['avg_sentiment'] - 0.5) * 2
            ds.loc[:, 'audio_visual_energy'] = ds['visual_brightness'] * ds['audio_bpm'] / 1000.0
            ds.loc[:, 'content_density'] = ds['cut_frequency'] / (ds['duration_sec'] + 1)
            
        feature_cols = [
            'duration_sec', 'follower_count_log', 'publish_hour',
            'avg_sentiment', 'visual_brightness', 'visual_saturation', 'cut_frequency', 'audio_bpm',
            'theme_encoded',
            'visual_impact', 'sensory_pace', 'sentiment_intensity', 'audio_visual_energy', 'content_density'
        ] + known_theme_cols
        
        X_test = X_test[feature_cols]
        
        # *对齐模型期望输入特征*
        champ_expected_features = []
        if hasattr(champion_model, 'feature_names_in_'):
            champ_expected_features = champion_model.feature_names_in_
        elif hasattr(champion_scaler, 'feature_names_in_'):
            champ_expected_features = champion_scaler.feature_names_in_
        else:
            champ_expected_features = feature_cols
            
        X_test_champ = pd.DataFrame(index=X_test.index)
        for col in champ_expected_features:
            if col in X_test.columns:
                X_test_champ[col] = X_test[col]
            else:
                X_test_champ[col] = 0.0
                
        # *获取预测结果*
        X_test_scaled_champion = champion_scaler.transform(X_test_champ)
        champion_preds_log = champion_model.predict(X_test_scaled_champion)
        champion_preds_log = np.clip(champion_preds_log, 0, GLOBAL_CONFIG['clip_max'])
        y_test_log = np.log1p(y_test_orig)
        
        # *创建导出目录*
        os.makedirs(GLOBAL_CONFIG['charts_dir'], exist_ok=True)
        
        # *图表 1：实际与预测的对数空间散点图*
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=y_test_log, y=champion_preds_log, alpha=0.6, color='blue')
        min_val = min(y_test_log.min(), champion_preds_log.min())
        max_val = max(y_test_log.max(), champion_preds_log.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='y=x')
        plt.title('Actual vs. Predicted (Log Space)')
        plt.xlabel('True Digg Count (Log1p)')
        plt.ylabel('Predicted Digg Count (Log1p)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(GLOBAL_CONFIG['charts_dir'], 'scatter_plot_log.png'), dpi=300)
        plt.close()
        
        # *图表 2：Top 15特征重要性*
        if hasattr(champion_model, 'feature_importances_'):
            importances = champion_model.feature_importances_
            feature_names = list(champ_expected_features)
            
            feat_imp_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False).head(15)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(
                x='Importance', 
                y='Feature', 
                data=feat_imp_df, 
                hue='Feature',
                palette='viridis',
                legend=False
            )
            plt.title('Top 15 Feature Importances')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.savefig(os.path.join(GLOBAL_CONFIG['charts_dir'], 'feature_importance.png'), dpi=300)
            plt.close()
            
    except Exception as e:
        print(f"[ERROR] Chart generation pipeline failed: {str(e)}")

if __name__ == '__main__':
    generate_evaluation_charts()
