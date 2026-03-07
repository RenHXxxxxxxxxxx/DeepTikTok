import os
import sys
import django
import pandas as pd
import numpy as np
from datetime import timedelta

# *========================================================*
# *1. 全局配置区 (Configuration-Driven Design)*
# *========================================================*
GLOBAL_CONFIG = {
    # *项目路径*
    "PROJECT_ROOT": r"D:\renhangxi_tiktok_bysj",
    
    # *输出配置*
    "OUTPUT_DIR": "training_tools/exported_data",
    "OUTPUT_FILENAME": "ml_training_dataset.csv",
    
    # *缺失值填充默认值*
    "DEFAULTS": {
        "visual_brightness": 128,
        "visual_saturation": 100,
        "cut_frequency": 0.5,
        "audio_bpm": 110,
    },
    
    # *数据清洗规则*
    "DATA_CLEANING": {
        "time_validity_years": 1,           # *剔除 N 年以前的样本*
        "digg_count_cap_percentile": 0.95,  # *digg_count 截断分位数*
    },
    
    # *特征字段定义*
    "FEATURES": {
        "input": ["follower_count", "publish_hour", "duration_sec"],
        "gpu": ["visual_brightness", "visual_saturation", "cut_frequency", "audio_bpm"],
        "target": ["digg_count"],
        "auxiliary": ["video_id", "theme_label"],
    }
}

# *========================================================*
# *2. Django 环境初始化*
# *========================================================*
PROJECT_ROOT = GLOBAL_CONFIG["PROJECT_ROOT"]
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'renhangxi_tiktok_bysj.settings')

try:
    django.setup()
except Exception as e:
    print(f"[ERROR] Django *初始化失败*: {e}")
    sys.exit(1)

from django.apps import apps

# *获取 Video 模型*
Video = apps.get_model('douyin_hangxi', 'Video')


# *========================================================*
# *3. 辅助函数*
# *========================================================*
def duration_to_seconds(duration_str):
    """
    *将时长字符串转换为秒数*
    *支持格式: "MM:SS", "HH:MM:SS", 或纯数字秒*
    """
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
        return 0


def extract_publish_hour(dt_value):
    """
    *从 publish_time (create_time) 提取发布小时*
    """
    try:
        if pd.isna(dt_value) or dt_value is None:
            return 12  # *默认中午12点*
        if isinstance(dt_value, datetime):
            return dt_value.hour
        # *尝试解析字符串*
        parsed = pd.to_datetime(dt_value, errors='coerce')
        if pd.isna(parsed):
            return 12
        return parsed.hour
    except Exception:
        return 12


# *========================================================*
# *4. 核心数据导出函数*
# *========================================================*
def export_training_data():
    """
    *从 Django Video 模型提取数据并导出为 ML 训练集 CSV*
    *严格执行数据清洗规则: 缺失值填充、时间过滤、异常值截断、去重*
    """
    # *构建输出路径*
    output_dir = os.path.join(PROJECT_ROOT, GLOBAL_CONFIG["OUTPUT_DIR"])
    output_csv = os.path.join(output_dir, GLOBAL_CONFIG["OUTPUT_FILENAME"])
    
    # *创建输出目录*
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 *已创建输出目录*: {output_dir}")
    
    print("=" * 65)
    print("🚀 *[ML 数据导出工具 v2.0] 启动*")
    print("=" * 65)
    
    # *----------------------------------------------------*
    # *Step 1: 从数据库查询原始数据*
    # *----------------------------------------------------*
    print("\n📊 *[Step 1/5] 正在从数据库查询视频数据...*")
    
    queryset = Video.objects.values(
        'video_id',           # *辅助字段*
        'theme_label',        # *辅助字段*
        'follower_count',     # *输入特征*
        'create_time',        # *用于提取 publish_hour*
        'duration',           # *用于转换为 duration_sec*
        'digg_count',         # *目标变量*
        'visual_brightness',  # *GPU 特征*
        'visual_saturation',  # *GPU 特征*
        'cut_frequency',      # *GPU 特征*
        'audio_bpm',          # *GPU 特征*
    )
    
    df = pd.DataFrame(list(queryset))
    
    if df.empty:
        print("❌ *[错误] 数据库中没有视频数据!*")
        return None
    
    original_count = len(df)
    print(f"   ✅ *成功加载 {original_count} 条原始记录*")
    
    # *----------------------------------------------------*
    # *Step 2: 特征工程 (转换字段)*
    # *----------------------------------------------------*
    print("\n🔧 *[Step 2/5] 特征工程转换...*")
    
    # *转换 duration -> duration_sec*
    df['duration_sec'] = df['duration'].apply(duration_to_seconds)
    print(f"   ✅ *duration_sec 转换完成*")
    
    # *提取 publish_hour*
    df['publish_hour'] = df['create_time'].apply(extract_publish_hour)
    print(f"   ✅ *publish_hour 提取完成*")
    
    # *----------------------------------------------------*
    # *Step 3: 数据清洗 - 缺失值填充*
    # *----------------------------------------------------*
    print("\n🧹 *[Step 3/5] 数据清洗 - 缺失值填充...*")
    
    defaults = GLOBAL_CONFIG["DEFAULTS"]
    for col, default_val in defaults.items():
        if col in df.columns:
            null_count = df[col].isna().sum()
            df[col] = df[col].fillna(default_val)
            if null_count > 0:
                print(f"   📌 *{col}*: 填充 {null_count} 个空值 -> {default_val}")
    
    # *----------------------------------------------------*
    # *Step 4: 数据清洗 - 时间有效性过滤*
    # *----------------------------------------------------*
    print("\n📅 *[Step 4/5] 数据清洗 - 时间有效性过滤...*")
    
    # *使用 Django 的 timezone 工具确保时区感知的日期比较*
    from django.utils import timezone
    
    years_threshold = GLOBAL_CONFIG["DATA_CLEANING"]["time_validity_years"]
    cutoff_date = timezone.now() - timedelta(days=365 * years_threshold)
    
    # *转换 create_time 为 datetime 类型 (保留时区信息)*
    df['create_time_dt'] = pd.to_datetime(df['create_time'], errors='coerce', utc=True)
    
    # *将 cutoff_date 转换为 pandas 兼容格式*
    cutoff_pd = pd.Timestamp(cutoff_date)
    
    # *过滤掉 1 年前的样本*
    before_filter = len(df)
    df = df[df['create_time_dt'] >= cutoff_pd]
    after_filter = len(df)
    removed_by_time = before_filter - after_filter
    
    print(f"   📌 *时间阈值*: {cutoff_date.strftime('%Y-%m-%d')}")
    print(f"   📌 *剔除过旧样本*: {removed_by_time} 条")
    
    if df.empty:
        print("❌ *[错误] 时间过滤后没有剩余数据!*")
        return None
    
    # *----------------------------------------------------*
    # *Step 5: 数据清洗 - 异常值截断 (Capping)*
    # *----------------------------------------------------*
    print("\n📉 *[Step 5/5] 数据清洗 - 异常值截断 (Capping)...*")
    
    cap_percentile = GLOBAL_CONFIG["DATA_CLEANING"]["digg_count_cap_percentile"]
    digg_cap_value = df['digg_count'].quantile(cap_percentile)
    
    before_cap_max = df['digg_count'].max()
    capped_count = (df['digg_count'] > digg_cap_value).sum()
    df['digg_count'] = df['digg_count'].clip(upper=digg_cap_value)
    
    print(f"   📌 *95% 分位数阈值*: {digg_cap_value:,.0f}")
    print(f"   📌 *原始最大值*: {before_cap_max:,.0f}")
    print(f"   📌 *截断样本数*: {capped_count} 条")
    
    # *----------------------------------------------------*
    # *Step 6: 数据清洗 - 去重*
    # *----------------------------------------------------*
    print("\n🔄 *去重处理 (基于 video_id)...*")
    
    before_dedup = len(df)
    df = df.drop_duplicates(subset=['video_id'], keep='first')
    after_dedup = len(df)
    duplicates_removed = before_dedup - after_dedup
    
    print(f"   📌 *移除重复记录*: {duplicates_removed} 条")
    
    # *----------------------------------------------------*
    # *构建最终输出 DataFrame (仅保留指定字段)*
    # *----------------------------------------------------*
    features = GLOBAL_CONFIG["FEATURES"]
    final_columns = (
        features["auxiliary"] + 
        features["input"] + 
        features["gpu"] + 
        features["target"]
    )
    
    # *确保所有列存在*
    for col in final_columns:
        if col not in df.columns:
            df[col] = 0
            print(f"   ⚠️ *警告*: 列 '{col}' 不存在，已填充默认值 0")
    
    df_final = df[final_columns].copy()
    
    # *----------------------------------------------------*
    # *保存到 CSV*
    # *----------------------------------------------------*
    try:
        df_final.to_csv(output_csv, index=False, encoding='utf_8_sig')
        print(f"\n💾 *数据已导出至*: {output_csv}")
    except Exception as e:
        print(f"❌ *[错误] CSV 写入失败*: {e}")
        return None
    
    # *----------------------------------------------------*
    # *打印统计摘要*
    # *----------------------------------------------------*
    print("\n" + "=" * 65)
    print("📈 *数据集统计摘要*")
    print("=" * 65)
    
    print(f"\n   📊 *样本统计*:")
    print(f"      • 原始记录数: {original_count}")
    print(f"      • 时间过滤剔除: {removed_by_time}")
    print(f"      • 去重移除: {duplicates_removed}")
    print(f"      • 最终样本数: {len(df_final)}")
    
    print(f"\n   🏷️ *主题分布*:")
    theme_counts = df_final['theme_label'].value_counts()
    for theme, count in theme_counts.items():
        pct = count / len(df_final) * 100
        print(f"      • {theme}: {count} 条 ({pct:.1f}%)")
    
    print(f"\n   🎯 *目标变量 (digg_count) 分布 [已截断]*:")
    print(f"      • 最小值: {df_final['digg_count'].min():,.0f}")
    print(f"      • 最大值: {df_final['digg_count'].max():,.0f}")
    print(f"      • 平均值: {df_final['digg_count'].mean():,.2f}")
    print(f"      • 中位数: {df_final['digg_count'].median():,.2f}")
    print(f"      • 标准差: {df_final['digg_count'].std():,.2f}")
    
    print(f"\n   🎨 *GPU 特征覆盖率* (非默认值占比):")
    for col, default_val in defaults.items():
        if col in df_final.columns:
            non_default = (df_final[col] != default_val).sum()
            coverage = non_default / len(df_final) * 100
            print(f"      • {col}: {coverage:.1f}% 有效")
    
    print("\n" + "=" * 65)
    print("✅ *导出完成! 数据已准备好用于机器学习训练*")
    print("=" * 65)
    
    return df_final


# *========================================================*
# *5. 运行入口*
# *========================================================*
if __name__ == '__main__':
    export_training_data()
