import pandas as pd
import numpy as np
import os
# ==========================================
# 用户配置区 (已修改为 Sikachi 主题)
# ==========================================
# 1. 输入 A: 刚才特征工厂产出的结果 (视频+GPU特征)
VIDEO_DATA_PATH = r"D:\renhangxi_tiktok_bysj\multy_video_data\douyin_data_sikachi_with_gpu_features.csv"

# 2. 输入 B: 爬虫抓取的原始评论数据
# (注意：如果之前爬虫生成的文件名是 douyin_comment_sikachi.csv，请核对这里)
# 暂时指向爬虫原始产出即可，除非你有单独跑过 comment_refiner
COMMENT_DATA_PATH = r"D:\renhangxi_tiktok_bysj\data\douyin_comment_sikachi.csv"

# 3. 输出: 最终清洗完成的“黄金数据集”
FINAL_OUTPUT_PATH = r"D:\renhangxi_tiktok_bysj\final_multimodal_dataset_cleaned\final_sikachi.csv"
# ==========================================

def clean_and_fusion():
    print("开始执行多模态数据清洗与融合链路...")

    # 1. 加载数据
    if not os.path.exists(VIDEO_DATA_PATH) or not os.path.exists(COMMENT_DATA_PATH):
        print("❌ 错误：找不到输入的 CSV 文件，请检查路径。")
        return

    v_df = pd.read_csv(VIDEO_DATA_PATH)
    c_df = pd.read_csv(COMMENT_DATA_PATH)

    # --- 第一部分：视频数据深度清洗 ---
    print("-> 正在清洗视频元数据...")

    # A. 剔除全空维度
    if '下载数量' in v_df.columns:
        v_df.drop(columns=['下载数量'], inplace=True)

    # B. 转化视频时长为秒 (01:05 -> 65)
    def time_to_sec(t_str):
        try:
            m, s = map(int, str(t_str).split(':'))
            return m * 60 + s
        except:
            return 0

    v_df['duration_sec'] = v_df['视频时长'].apply(time_to_sec)

    # C. 提取发布小时 (2024/12/10 18:46:44 -> 18)
    v_df['publish_hour'] = pd.to_datetime(v_df['发表时间']).dt.hour

    # D. 计算互动率 (Interaction Rate)
    # 公式：(点赞+收藏+评论+分享) / 粉丝数
    v_df['interaction_rate'] = (v_df['点赞数量'] + v_df['收藏数量'] +
                                v_df['评论数量'] + v_df['分享数量']) / (v_df['粉丝数量'] + 1)

    # --- 第二部分：评论数据聚合 ---
    print("-> 正在分析评论区情感/热度维度...")

    # 我们按视频ID汇总评论，计算该视频下评论的平均点赞数 (代表评论区的共鸣程度)
    comment_agg = c_df.groupby('视频ID').agg(
        avg_comment_likes=('点赞数', 'mean'),
        total_comments_count=('评论ID', 'count')
    ).reset_index()

    # --- 第三部分：跨模态终极融合 ---
    print("-> 正在进行跨模态多维对齐...")

    # 以视频表为主表，左连接评论汇总表
    final_df = pd.merge(v_df, comment_agg, on='视频ID', how='left')

    # 填充那些没有评论的视频数据 (填0)
    final_df['avg_comment_likes'] = final_df['avg_comment_likes'].fillna(0)

    # --- 第四部分：特征对齐与输出 ---
    # 为了模型训练，我们将列名统一为英文，方便代码调用
    rename_dict = {
        '粉丝数量': 'follower_count',
        '点赞数量': 'digg_count',
        '收藏数量': 'collect_count',
        '评论数量': 'comment_count',
        '分享数量': 'share_count'
    }
    final_df.rename(columns=rename_dict, inplace=True)

    # 最后整理一下列的顺序，把 Label (点赞数) 放在显眼位置
    # 并保留你的 GPU 算出的 visual_brightness, audio_bpm 等

    final_df.to_csv(FINAL_OUTPUT_PATH, index=False, encoding='utf_8_sig')

    print("\n" + "=" * 50)
    print(f"✅ 清洗与融合任务完成！")
    print(f"📊 最终样本量: {len(final_df)} 条")
    print(f"💎 最终维度数: {len(final_df.columns)} 维")
    print(f"💾 产物路径: {FINAL_OUTPUT_PATH}")
    print("=" * 50)


if __name__ == "__main__":
    clean_and_fusion()