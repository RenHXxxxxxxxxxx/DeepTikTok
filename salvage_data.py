import os
import sys
import django
import pandas as pd
from django.apps import apps

# 定义项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 1. 初始化 Django 独立环境
# ==========================================
# 获取当前脚本所在根目录 D:\renhangxi_tiktok_bysj
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 将内层项目目录加入路径 D:\renhangxi_tiktok_bysj\renhangxi_tiktok_bysj
sys.path.append(os.path.join(BASE_DIR, 'renhangxi_tiktok_bysj'))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'renhangxi_tiktok_bysj.settings')
django.setup()

# [关键修改] 使用反射机制动态获取模型，彻底解决 app_label 问题
try:
    Video = apps.get_model('douyin_hangxi', 'Video')
except LookupError:
    # 如果上面不行，尝试带前缀的完整路径
    Video = apps.get_model('renhangxi_tiktok_bysj.douyin_hangxi', 'Video')

print("✅ Django 环境与模型加载成功！")

def salvage_to_csv(theme_name):
    print(f"\n🔍 开始从数据库打捞主题 [{theme_name}] 的数据...")
    
    # 2. 从数据库查询该主题下的所有视频 (使用正确的字段名 theme_label)
    videos = Video.objects.filter(theme_label=theme_name)
    
    if not videos.exists():
        print(f"❌ 数据库中未找到主题为 '{theme_name}' 的数据，请检查拼写。")
        return
        
    print(f"✅ 找到 {videos.count()} 条视频记录！正在生成 CSV...")
    
    # 3. 提取关键字段（使用正确的模型属性名）
    data_list = []
    for v in videos:
        data_list.append({
            '视频ID': v.video_id,
            '用户名': getattr(v, 'nickname', '未知'),
            '视频描述': getattr(v, 'desc', ''),
            '评论数量': getattr(v, 'comment_count', 0),
            '点赞数量': getattr(v, 'digg_count', 0)
        })
        
    # 4. 写入 CSV，完美还原事故现场
    csv_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f"douyin_video_{theme_name}.csv")
    
    df = pd.DataFrame(data_list)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    print(f"🎉 抢救大成功！文件已生成: {csv_path}")
    print(f"💡 下一步：你可以直接使用 [仅评论采集模式] 读取这个文件了！")

if __name__ == "__main__":
    print("="*50)
    print("🚑 数据库逆向打捞工具 (Reverse ETL)")
    print("="*50)
    target_theme = input("👉 请输入需要抢救的主题名称 (例如 '卢本伟' 或 '鬼畜'): ").strip()
    if target_theme:
        salvage_to_csv(target_theme)