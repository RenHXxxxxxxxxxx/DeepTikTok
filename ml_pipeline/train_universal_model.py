# -*- coding: utf-8 -*-
"""
*XGBoost 点赞数预测模型训练脚本*
*使用标准化特征 + 对数变换目标值进行回归训练*
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb

# *========== 全局配置 ==========*
GLOBAL_CONFIG = {
    # *数据路径*
    "input_csv": r"d:\renhangxi_tiktok_bysj\training_tools\exported_data\ml_training_dataset.csv",
    
    # *模型资产输出路径*
    "assets_dir": r"d:\renhangxi_tiktok_bysj\build_model\assets",
    "model_filename": "universal_model.pkl",
    "scaler_filename": "universal_scaler.pkl",
    
    # *特征列定义*
    "feature_columns": [
        "follower_count",
        "publish_hour",
        "duration_sec",
        "visual_brightness",
        "visual_saturation",
        "cut_frequency",
        "audio_bpm"
    ],
    
    # *目标列*
    "target_column": "digg_count",
    
    # *训练参数*
    "test_size": 0.2,
    "random_state": 42,
    
    # *XGBoost 参数*
    "xgb_params": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 5,
        "random_state": 42
    }
}


def check_cuda_available():
    """
    *检测是否有可用的 CUDA GPU*
    """
    try:
        # *尝试导入 torch 检测 CUDA*
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[INFO] *检测到 CUDA GPU: {gpu_name}，将使用 GPU 加速训练*")
            return True
    except ImportError:
        print("[WARN] *PyTorch 未安装，尝试其他方式检测 CUDA*")
    
    try:
        # *直接通过 xgboost 检测*
        # *XGBoost 2.0+ 支持 device='cuda' 参数*
        import numpy as np
        test_X = np.array([[1.0, 2.0], [3.0, 4.0]])
        test_y = np.array([1.0, 2.0])
        test_params = {"device": "cuda", "n_estimators": 1, "verbosity": 0}
        test_model = xgb.XGBRegressor(**test_params)
        test_model.fit(test_X, test_y)
        # *如果训练成功完成，说明 CUDA 可用*
        print("[INFO] *通过 XGBoost 验证 CUDA 可用，将使用 GPU 加速训练*")
        return True
    except Exception as e:
        print(f"[WARN] *XGBoost CUDA 测试失败: {e}*")
    
    print("[INFO] *未检测到 CUDA GPU，将使用 CPU 训练*")
    return False


def load_and_prepare_data(config):
    """
    *加载 CSV 数据并准备特征和目标*
    """
    input_path = config["input_csv"]
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"*训练数据文件不存在: {input_path}*")
    
    print(f"[INFO] *正在加载训练数据: {input_path}*")
    df = pd.read_csv(input_path)
    print(f"[INFO] *加载完成，共 {len(df)} 条样本*")
    
    # *提取特征列*
    feature_cols = config["feature_columns"]
    target_col = config["target_column"]
    
    # *检查列是否存在*
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"*缺少特征列: {missing_cols}*")
    
    if target_col not in df.columns:
        raise ValueError(f"*缺少目标列: {target_col}*")
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # *处理缺失值 - 使用中位数填充*
    for col in feature_cols:
        if X[col].isnull().any():
            median_val = X[col].median()
            X[col].fillna(median_val, inplace=True)
            print(f"[WARN] *特征列 {col} 存在缺失值，已用中位数 {median_val:.2f} 填充*")
    
    # *处理目标值缺失*
    if y.isnull().any():
        valid_mask = ~y.isnull()
        X = X[valid_mask]
        y = y[valid_mask]
        print(f"[WARN] *目标列存在缺失值，已移除，剩余 {len(y)} 条样本*")
    
    return X, y


def train_model(X, y, config):
    """
    *训练 XGBoost 回归模型*
    """
    # *1. 划分训练集和测试集*
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["test_size"],
        random_state=config["random_state"]
    )
    print(f"[INFO] *训练集: {len(X_train)} 条，测试集: {len(X_test)} 条*")
    
    # *2. 特征标准化 (Z-Score)*
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("[INFO] *特征标准化完成 (StandardScaler)*")
    
    # *3. 目标变量对数变换 (log1p)*
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)
    print("[INFO] *目标变量已进行 log1p 变换*")
    
    # *4. 配置 XGBoost 参数*
    xgb_params = config["xgb_params"].copy()
    
    # *检测 CUDA 并配置设备*
    if check_cuda_available():
        xgb_params["device"] = "cuda"
    else:
        xgb_params["device"] = "cpu"
    
    # *5. 训练模型*
    print("[INFO] *开始训练 XGBoost 回归模型...*")
    model = xgb.XGBRegressor(**xgb_params)
    
    try:
        model.fit(
            X_train_scaled, y_train_log,
            eval_set=[(X_test_scaled, y_test_log)],
            verbose=False
        )
    except Exception as e:
        print(f"[ERROR] *训练出错: {e}*")
        raise
    
    print("[INFO] *模型训练完成*")
    
    # *6. 预测并评估*
    y_pred_log = model.predict(X_test_scaled)
    
    # *计算 R2 Score (在对数空间)*
    r2 = r2_score(y_test_log, y_pred_log)
    print(f"\n{'='*50}")
    print(f"[RESULT] *R2 Score (测试集, log空间): {r2:.4f}*")
    
    # *还原预测值到真实点赞数空间*
    y_pred_real = np.expm1(y_pred_log)
    y_test_real = y_test.values  # *原始真实值*
    
    # *防止负数预测*
    y_pred_real = np.maximum(y_pred_real, 0)
    
    # *计算 MAE (在真实点赞数空间)*
    mae = mean_absolute_error(y_test_real, y_pred_real)
    print(f"[RESULT] *Mean Absolute Error (真实点赞数空间): {mae:.2f}*")
    print(f"{'='*50}\n")
    
    return model, scaler, {
        "r2_score": r2,
        "mae": mae,
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    }


def save_assets(model, scaler, config):
    """
    *保存模型和 Scaler 到磁盘*
    """
    assets_dir = Path(config["assets_dir"])
    
    # *创建目录*
    assets_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] *资产目录: {assets_dir}*")
    
    # *保存模型*
    model_path = assets_dir / config["model_filename"]
    joblib.dump(model, model_path)
    print(f"[INFO] *模型已保存: {model_path}*")
    
    # *保存 Scaler*
    scaler_path = assets_dir / config["scaler_filename"]
    joblib.dump(scaler, scaler_path)
    print(f"[INFO] *Scaler 已保存: {scaler_path}*")
    
    return model_path, scaler_path


def main():
    """
    *主函数入口*
    """
    print("\n" + "="*60)
    print("*XGBoost 点赞数预测模型训练*")
    print("="*60 + "\n")
    
    try:
        # *1. 加载数据*
        X, y = load_and_prepare_data(GLOBAL_CONFIG)
        
        # *2. 训练模型*
        model, scaler, metrics = train_model(X, y, GLOBAL_CONFIG)
        
        # *3. 保存资产*
        model_path, scaler_path = save_assets(model, scaler, GLOBAL_CONFIG)
        
        # *4. 打印最终摘要*
        print("\n" + "="*60)
        print("*训练完成摘要*")
        print("="*60)
        print(f"  *训练样本数*: {metrics['train_samples']}")
        print(f"  *测试样本数*: {metrics['test_samples']}")
        print(f"  *R2 Score*: {metrics['r2_score']:.4f}")
        print(f"  *MAE (点赞数)*: {metrics['mae']:.2f}")
        print(f"  *模型路径*: {model_path}")
        print(f"  *Scaler 路径*: {scaler_path}")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n[FATAL] *训练流程失败: {e}*")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
