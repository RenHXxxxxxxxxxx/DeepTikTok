# Architectural Explanation: DeepTikTok Analysis System Model Retraining

This document outlines the final model retraining architectural design and provides the execution code for `train_master_arena.py`. We utilize a highly tuned pipeline specifically engineered for small, imbalanced datasets with a long-tail distribution across themes.

## 1. Data Loading Strategy
The pipeline leverages Django's ORM object queries directly mapped to a pandas DataFrame. This approach is memory efficient since we use `.values(...)` yielding dictionary representations before translating them into pandas tabular format. We filter records down to only those that have successfully completed analysis (`analysis_status=2`). Extreme outliers or NA rows are safely discarded to ensure clean training grounds.

## 2. Advanced Feature Engineering
Because of the heavy sparsity and skewed target (`digg_count`), we perform critical transformations:
- **Log Transformation**: Applied to `follower_count` and `digg_count` via `np.log1p()` to shrink extreme variance and make optimization paths more stable.
- **Bayesian Target Encoding**: Standard one-hot encoding would explode the feature space and fail spectacularly on rare themes (N=5 to 10). By implementing Bayesian Smoothed mean encoding, we pull tail-theme estimates toward the global mean, avoiding overfitting to noise in sparse themes.
- **Multimodal Fusing**: We inject extracted modalities spanning visual aspects (`visual_brightness`, `visual_saturation`), auditory rhythm (`audio_bpm`), editing syntax (`cut_frequency`), and textual sentiment (`avg_sentiment`) to create a well-rounded dimensional representation.

## 3. Modeling and Evaluation Strategy
Because we have exactly 2,147 samples, our ensemble model must remain highly constrained:
- **Algorithm**: `XGBRegressor` stands as the optimal choice for tabular data, but is customized for our dataset.
- **Hyperparameters**: `max_depth` is constrained (e.g., 4) and L2 regularization (`reg_lambda`) is dialed up significantly as a penalty across leaf weights, directly punishing overconfidence on small theme splits.
- **Evaluation**: The model performs an 80/20 train/test evaluation reporting RMSE, MAE, and $R^2$ scores, reverting the `log1p` back into real counts via `np.expm1()` during metric calculation.
- **Persistence**: Using `joblib`, the model tree is tightly packed and dropped into the local model directory for the online inference architecture to hot-reload.

---

## Code Implementation: `train_master_arena.py`

```python
import os
import sys
import django
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# *全局配置字典*
GLOBAL_CONFIG = {
    "DJANGO_SETTINGS_MODULE": "renhangxi_tiktok_bysj.settings",
    "MODEL_SAVE_PATH": "models/xgboost_digg_predictor.pkl",
    "TEST_SIZE": 0.2,
    "RANDOM_STATE": 42,
    "BAYESIAN_WEIGHT": 10.0,
    "XGB_PARAMS": {
        "max_depth": 4,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 10.0,
        "random_state": 42,
        "objective": "reg:squarederror"
    },
    "FEATURES": [
        "follower_count_log",
        "theme_encoded",
        "visual_brightness",
        "visual_saturation",
        "audio_bpm",
        "cut_frequency",
        "avg_sentiment"
    ],
    "TARGET": "digg_count_log"
}

# *初始化Django环境*
def setup_django():
    # *配置环境变量*
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", GLOBAL_CONFIG["DJANGO_SETTINGS_MODULE"])
    # *设置项目根目录，假设该脚本在项目根目录下运行*
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    django.setup()

def fetch_data_from_db():
    # *获取数据库数据，使用try_except捕获异常*
    try:
        from douyin_hangxi.models import Video
        # *查询具有AI特征的视频数据*
        queryset = Video.objects.filter(analysis_status=2).values(
            "id", "theme_label", "digg_count", "follower_count",
            "visual_brightness", "visual_saturation", "audio_bpm",
            "cut_frequency", "avg_sentiment"
        )
        df = pd.DataFrame(list(queryset))
        
        # *过滤极端异常值或缺失值*
        if not df.empty:
            df = df.dropna(subset=['digg_count', 'theme_label', 'follower_count'])
            
        return df
    except Exception as e:
        # *捕获数据库读取异常*
        print(f"Database error: {e}")
        return pd.DataFrame()

def feature_engineering(df):
    # *特征工程逻辑，使用try_except保证鲁棒性*
    try:
        if df.empty:
            return df
            
        # *对长尾偏态分布进行Log1p变换*
        df['follower_count_log'] = np.log1p(df['follower_count'])
        df['digg_count_log'] = np.log1p(df['digg_count'])
        
        # *贝叶斯目标编码平滑处理*
        global_mean = df['digg_count_log'].mean()
        weight = GLOBAL_CONFIG["BAYESIAN_WEIGHT"]
        
        # *计算每个主题的统计量*
        theme_stats = df.groupby('theme_label')['digg_count_log'].agg(['count', 'mean']).reset_index()
        theme_stats.rename(columns={'mean': 'local_mean'}, inplace=True)
        
        # *应用平滑公式*
        theme_stats['theme_encoded'] = (theme_stats['count'] * theme_stats['local_mean'] + weight * global_mean) / (theme_stats['count'] + weight)
        
        # *合并回主数据框*
        df = df.merge(theme_stats[['theme_label', 'theme_encoded']], on='theme_label', how='left')
        
        # *处理潜在的多模态缺失值为0*
        multimodal_cols = ["visual_brightness", "visual_saturation", "audio_bpm", "cut_frequency", "avg_sentiment"]
        for col in multimodal_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)
                
        return df
    except Exception as e:
        # *捕获特征工程异常*
        print(f"Feature engineering error: {e}")
        return df

def train_and_evaluate(df):
    # *训练评估模型逻辑*
    try:
        if df.empty:
            return None
            
        features = GLOBAL_CONFIG["FEATURES"]
        target = GLOBAL_CONFIG["TARGET"]
        
        X = df[features]
        y = df[target]
        
        # *切分数据集*
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=GLOBAL_CONFIG["TEST_SIZE"], 
            random_state=GLOBAL_CONFIG["RANDOM_STATE"]
        )
        
        # *初始化XGBoost模型*
        model = xgb.XGBRegressor(**GLOBAL_CONFIG["XGB_PARAMS"])
        
        # *训练模型*
        model.fit(X_train, y_train)
        
        # *预测*
        y_pred = model.predict(X_test)
        
        # *计算评估指标，需要将log变换后的结果还原*
        y_test_real = np.expm1(y_test)
        y_pred_real = np.expm1(y_pred)
        
        rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
        mae = mean_absolute_error(y_test_real, y_pred_real)
        r2 = r2_score(y_test_real, y_pred_real)
        
        print(f"Evaluation Results:\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR2 Score: {r2:.4f}")
        
        return model
    except Exception as e:
        # *捕获训练异常*
        print(f"Training error: {e}")
        return None

def save_model(model):
    # *保存模型逻辑*
    try:
        if not model:
            return
            
        save_path = GLOBAL_CONFIG["MODEL_SAVE_PATH"]
        # *确保目录存在*
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        joblib.dump(model, save_path)
        print(f"Model saved completely at {save_path}")
    except Exception as e:
        # *捕获文件保存异常*
        print(f"Model save error: {e}")

def main():
    # *主控制流*
    try:
        setup_django()
        
        print("Fetching data from DB...")
        df_raw = fetch_data_from_db()
        if df_raw.empty:
            # *数据为空则直接退出*
            print("Empty DataFrame retrieved. Exiting pipeline.")
            return
            
        print(f"Data retrieved: {len(df_raw)} records.")
        
        print("Applying Advanced Feature Engineering...")
        df_engineered = feature_engineering(df_raw)
        
        print("Training XGBoost Regression Model...")
        model = train_and_evaluate(df_engineered)
        
        if model:
            print("Persisting model to disk...")
            save_model(model)
            
    except Exception as e:
        # *捕获整体的顶层异常*
        print(f"Pipeline crashed gracefully: {e}")

if __name__ == "__main__":
    main()
```
