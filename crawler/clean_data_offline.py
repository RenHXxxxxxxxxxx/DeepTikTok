import os
import re
import django
import pandas as pd
from datetime import datetime

# =================  环境与配置初始化 =================
# 1. 自动定位 Django 项目配置
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'renhangxi_tiktok_bysj.settings')
django.setup()

# 2. 导入模型
from renhangxi_tiktok_bysj.douyin_hangxi.models import Video, Comment
from config import Config

# =================  文件路径配置 (对接爬虫脚本) =================
# 输入：爬虫生成的原始文件
RAW_VIDEO_FILE = 'douyin_data1.csv'  # 对应 spyder_videos.py 的输出
RAW_COMMENT_FILE = 'douyin_comments_collected.csv'  # 对应 spyder_comment.py 的输出

# 输出：清洗后的中间文件 (用于备份检查)
CLEAN_VIDEO_FILE = 'videos_clean_hangxi.csv'
CLEAN_COMMENT_FILE = 'comments_clean_hangxi.csv'


# =================  核心清洗工具函数 =================

def clean_text_nuclear(text):
    """
    【核弹级文本清洗】
    1. 去除 @xxx, [xxx], 网址
    2. 仅保留汉字、英数、常用标点
    """
    if not isinstance(text, str):
        return ""

    # 剔除 @用户 和 回复
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'回复 \S+:', '', text)

    # 剔除 [表情] 【话题】
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'【.*?】', '', text)

    # 剔除 URL
    text = re.sub(r'http\S+', '', text)

    # 白名单过滤：只留 中文、英文、数字、标点
    # 注意：这里会把 Emoji 过滤掉
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？、…,.!?"\']', '', text)

    return text.strip()


def parse_douyin_num(x):
    """
    【数值标准化】
    将 '1.2w', '1000+', 'NaN' 统一转换为 整数
    """
    if pd.isnull(x) or x == '':
        return 0

    x_str = str(x).lower().strip()

    # 处理 'w' / '万'
    if 'w' in x_str or '万' in x_str:
        x_str = re.sub(r'[w万]', '', x_str)
        try:
            return int(float(x_str) * 10000)
        except:
            return 0

    # 处理 '+' (如 10w+)
    x_str = x_str.replace('+', '')

    try:
        return int(float(x_str))
    except:
        return 0


def clean_time(t_str):
    """时间格式标准化"""
    try:
        return pd.to_datetime(t_str)
    except:
        return datetime.now()


# =================  主执行逻辑 =================

def run_pipeline():
    print(" *[DEPRECATED] 此脚本已弃用，请使用 data_manager.py 中的 UnifiedPersistenceManager*")
    print(" *[DEPRECATED] This script is legacy code. Logic has been moved to UnifiedPersistenceManager.*")
    print(" 启动数据清洗与入库流水线...")
    print(f"   当前项目主题配置: [{Config.PROJECT_TAG}]")

    # --- 第一阶段：视频数据处理 ---
    if os.path.exists(RAW_VIDEO_FILE):
        print(f"\n 正在处理视频数据: {RAW_VIDEO_FILE}")

        # 1. 读取原始数据 (指定 video_id 为字符串，防止科学计数法)
        # 注意：这里根据 spyder_videos.py 的输出并未包含表头，所以 header=None
        # 我们手动指定列名，确保与爬虫脚本一致
        v_cols = ['nickname', 'follower_count', 'desc', 'video_id', 'create_time',
                  'duration', 'digg_count', 'collect_count', 'comment_count',
                  'share_count', 'download_count']

        try:
            df_v = pd.read_csv(RAW_VIDEO_FILE, header=None, names=v_cols, dtype={'video_id': str}, on_bad_lines='skip')
        except Exception as e:
            print(f" 读取视频文件失败: {e}")
            return

        # 2. 清洗数据
        # 过滤掉非法的 video_id (如包含表头或空行)
        df_v = df_v[df_v['video_id'].str.len() > 10]

        # 数值转换
        num_cols = ['follower_count', 'digg_count', 'collect_count', 'comment_count', 'share_count', 'download_count']
        for col in num_cols:
            df_v[col] = df_v[col].apply(parse_douyin_num)

        # 时间转换
        df_v['create_time'] = pd.to_datetime(df_v['create_time'], errors='coerce')

        # 3. 保存清洗后的副本 (作为 Clean Data)
        df_v.to_csv(CLEAN_VIDEO_FILE, index=False, encoding='utf-8-sig')
        print(f"    清洗完成，已备份至: {CLEAN_VIDEO_FILE}")

        # 4. 入库 (增量更新)
        print("    正在写入数据库 (tb_video)...")
        new_videos = []
        # 获取库里已有的ID，避免重复
        existing_ids = set(Video.objects.values_list('video_id', flat=True))

        for _, row in df_v.iterrows():
            vid = str(row['video_id'])
            if vid not in existing_ids:
                new_videos.append(Video(
                    video_id=vid,
                    nickname=str(row['nickname']),
                    desc=str(row['desc'])[:500],  # 截断防止超长
                    create_time=row['create_time'],
                    duration=str(row['duration']),
                    follower_count=row['follower_count'],
                    digg_count=row['digg_count'],
                    comment_count=row['comment_count'],
                    collect_count=row['collect_count'],
                    share_count=row['share_count'],
                    download_count=row['download_count']
                ))
                existing_ids.add(vid)  # 更新缓存

        if new_videos:
            Video.objects.bulk_create(new_videos, batch_size=500)
            print(f"    成功插入 {len(new_videos)} 条新视频数据！")
        else:
            print("    没有新视频需要插入 (全部已存在)。")

    else:
        print(f" 未找到视频源文件: {RAW_VIDEO_FILE}，请先运行 spyder_videos.py")

    # --- 第二阶段：评论数据处理 ---
    if os.path.exists(RAW_COMMENT_FILE):
        print(f"\n 正在处理评论数据: {RAW_COMMENT_FILE}")

        # 1. 读取 (同样假设无表头，根据 spyder_comment.py 结构定义)
        c_cols = ['video_id', 'comment_id', 'create_time', 'nickname', 'content', 'digg_count', 'ip_label']
        try:
            df_c = pd.read_csv(RAW_COMMENT_FILE, header=None, names=c_cols, dtype={'video_id': str, 'comment_id': str},
                               on_bad_lines='skip')
        except Exception as e:
            print(f" 读取评论文件失败: {e}")
            return

        # 2. 深度清洗
        # 必须关联有效的视频ID
        valid_vids = set(Video.objects.values_list('video_id', flat=True))
        df_c = df_c[df_c['video_id'].isin(valid_vids)]

        # 文本清洗
        df_c['content_clean'] = df_c['content'].apply(clean_text_nuclear)

        # 剔除无效评论 (洗完变空的)
        df_c = df_c[df_c['content_clean'].str.len() > 0]

        # 数值与特征
        df_c['digg_count'] = df_c['digg_count'].apply(parse_douyin_num)
        df_c['create_time'] = pd.to_datetime(df_c['create_time'], errors='coerce')
        df_c['hour'] = df_c['create_time'].dt.hour
        df_c['text_len'] = df_c['content_clean'].str.len()

        # 3. 保存副本
        df_c.to_csv(CLEAN_COMMENT_FILE, index=False, encoding='utf-8-sig')
        print(f"    清洗完成，已备份至: {CLEAN_COMMENT_FILE}")

        # 4. 入库
        print("    正在写入数据库 (tb_comment)...")
        new_comments = []
        existing_cids = set(Comment.objects.values_list('comment_id', flat=True))

        for _, row in df_c.iterrows():
            cid = str(row['comment_id'])
            if cid not in existing_cids:
                new_comments.append(Comment(
                    video_id=str(row['video_id']),
                    comment_id=cid,
                    nickname=str(row['nickname']),
                    content=str(row['content']),  # 原始文本
                    content_clean=str(row['content_clean']),  # 干净文本
                    create_time=row['create_time'],
                    ip_label=str(row['ip_label']),
                    digg_count=row['digg_count'],

                    # 默认情感值 (后续由 data_analyse 或专门的脚本处理)
                    sentiment_score=0.5,
                    sentiment_label="中性",

                    hour=int(row['hour']),
                    text_len=int(row['text_len'])
                ))
                existing_cids.add(cid)

        if new_comments:
            Comment.objects.bulk_create(new_comments, batch_size=1000)
            print(f"    成功插入 {len(new_comments)} 条新评论数据！")
        else:
            print("    没有新评论需要插入。")

    else:
        print(f" 未找到评论源文件: {RAW_COMMENT_FILE}，请先运行 spyder_comment.py")

    print("\n 数据清洗与入库流程结束。下一步：运行 data_analyse(offline).py 进行分析。")


if __name__ == '__main__':
    run_pipeline()
