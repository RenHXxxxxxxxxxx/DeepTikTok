import joblib
import numpy as np

# 定位到你硬盘上遗留的那个最大、最新的历史模型
model_path = r"D:\renhangxi_tiktok_bysj\ml_pipeline\artifacts\model_RF_v20260225_021615.pkl"

try:
    print("============================================================")
    print(" 开始执行：特征拓扑逆向工程 (Feature Topology Reverse Engineering)")
    print("============================================================")
    
    # 加载历史模型
    model = joblib.load(model_path)
    print(f" 模型加载成功！类型: {type(model)}")
    
    # 尝试提取特征元数据
    if hasattr(model, "feature_names_in_"):
        features = model.feature_names_in_
        print(f"\n 历史多维特征空间维度总数: {len(features)} 维")
        print("\n 完整的特征拓扑列表 (Feature Topology):")
        
        # 将所有特征打印出来，方便我们复制对齐
        for i, feature in enumerate(features):
            print(f"  [{i}] -> {feature}")
            
    elif hasattr(model, "n_features_in_"):
        print(f"\n 警告：该模型未保存具体的特征名，但期待的特征维度总数为: {model.n_features_in_} 维。")
    
except Exception as e:
    print(f" 逆向解析失败，错误信息: {e}")
print("\n============================================================")
