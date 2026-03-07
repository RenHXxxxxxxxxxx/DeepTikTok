# *================================================================*
# *评论补抓脚本：救援优化版（支持跳过 CSV 损坏行）*
# *================================================================*

import os
import sys
import pandas as pd
import csv # 增加 csv 库用于更精细的解析控制

# *项目根目录自动定位*
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from crawler.spyder_unified import DouyinUnifiedPipeline

def get_user_input():
    print("="*50)
    print("🎬 *抖音评论补抓助手 (Rescue & Recovery Mode)*")
    print("="*50)

    # 1. 扫描可用视频源
    print("\n🔍 *正在扫描数据目录...*")
    data_dir = os.path.join(PROJECT_ROOT, "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir) # 自动创建数据目录
        print(f"❌ *数据目录不存在，已为您创建: {data_dir}*")
        return None, None

    csv_files = [f for f in os.listdir(data_dir) if f.startswith("douyin_video_") and f.endswith(".csv")]
    if not csv_files:
        print("❌ *未找到任何视频数据文件 (douyin_video_*.csv)*")
        return None, None

    # 2. 列出选项
    print(f"✅ *找到 {len(csv_files)} 个任务主题:*")
    file_map = {}
    for idx, f in enumerate(csv_files, 1):
        theme = f.replace("douyin_video_", "").replace(".csv", "")
        file_map[idx] = {"file": f, "theme": theme}
        print(f"   [{idx}] {theme}   (文件: {f})")

    # 3. 用户交互
    while True:
        choice = input(f"\n👉 *请输入主题序号 (1-{len(csv_files)}):* ").strip()
        if choice.isdigit() and int(choice) in file_map:
            selected = file_map[int(choice)]
            break
        print("❌ *输入无效，请输入列表中存在的序号*")

    default_max = 500
    inp = input(f"👉 *每视频抓取多少条评论? (默认 {default_max}):* ").strip()
    max_fmt = int(inp) if inp.isdigit() else default_max

    return selected, max_fmt

def run_comment_only_crawler():
    selected_config, max_comments = get_user_input()
    if not selected_config: return

    video_csv_path = os.path.join(PROJECT_ROOT, "data", selected_config["file"])
    theme = selected_config["theme"]
    
    # --- 核心优化点：鲁棒性读取 ---
    print(f"\n🚀 *正在通过救援模式解析 CSV...*")
    try:
        # 1. 增加 on_bad_lines='skip'，完美跳过第 321 行那样的“逗号炸弹”
        # 2. 增加 dtype=str，防止 19 位长的视频 ID 被变成科学计数法
        # 3. 增加 engine='python'，应对包含复杂符号的视频标题
        df = pd.read_csv(
            video_csv_path, 
            dtype=str, 
            on_bad_lines='skip', 
            engine='python', 
            encoding='utf-8-sig'
        )

        # 智能匹配列名 (兼容 视频ID / video_id 以及 评论数量 / comment_count)
        id_col = next((c for c in df.columns if c in ['视频ID', 'video_id', 'id']), None)
        cnt_col = next((c for c in df.columns if c in ['评论数量', 'comment_count', 'count']), None)

        if not id_col:
            print(f"❌ *错误: 在 CSV 中找不到 ID 列。现有列名: {df.columns.tolist()}*")
            return

        # 统一映射为代码需要的字典格式
        video_data_list = []
        for _, row in df.iterrows():
            video_data_list.append({
                '视频ID': row[id_col],
                '评论数量': row[cnt_col] if cnt_col else 0
            })

        print(f"📂 *成功抢救出 {len(video_data_list)} 个有效视频 ID（已自动剔除坏行）*")
    except Exception as e:
        print(f"❌ *救援读取失败: {e}*")
        return
    # ----------------------------

    spider = None
    try:
        spider = DouyinUnifiedPipeline()
        spider.current_theme = theme
        spider.comment_save_path = os.path.join(PROJECT_ROOT, "data", f"douyin_comment_{theme}.csv")
        
        task_config = {
            "keyword": "", 
            "max_videos": 0,
            "max_comments": max_comments
        }
        
        print(f"\n💬 *启动评论补抓 | 主题: {theme} | 总进度: 0/{len(video_data_list)}*")
        spider.run_comment_crawler(task_config, video_data_list)
        
        print(f"\n✅ *补抓完成！评论已存入: {spider.comment_save_path}*")
        
    except Exception as e:
        print(f"❌ *评论采集阶段发生异常: {e}*")
    finally:
        if spider: spider.close()

if __name__ == "__main__":
    run_comment_only_crawler()
    input("\n[DONE] 按回车键退出脚本...")