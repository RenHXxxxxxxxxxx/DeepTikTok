import os
import django
import sys
import time

# 1. 初始化 Django 环境（离线运行必须步骤）
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'renhangxi_tiktok_bysj.settings')
django.setup()

# 2. 导入爬虫核心服务
try:
    # 确保 data 目录在路径中
    sys.path.insert(0, os.path.join(os.getcwd(), 'data'))
    from spyder_unified import run_spider_service
    print("✅ 爬虫模块导入成功")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# 3. 模拟前端参数进行“原地热机”
def dry_run():
    keyword = "陈翔六点半"  # 测试关键词
    max_videos = 5         # 少量测试
    max_comments = 10      # 少量测试
    theme_name = "离线测试主题"
    
    print(f"🚀 开始离线任务: [{keyword}] -> 主题: {theme_name}")
    
    # 模拟进度回调函数
    def mock_callback(current, total, start_time, message=""):
        print(f"  [进度反馈] {current}/{total} | 状态: {message}")

    start = time.time()
    try:
        # 调用核心函数
        v_path, c_path = run_spider_service(
            keyword, max_videos, max_comments, theme_name, mock_callback
        )
        print(f"\n🎉 抓取成功！")
        print(f"   视频文件: {v_path}")
        print(f"   评论文件: {c_path}")
        print(f"   总耗时: {time.time() - start:.2f} 秒")
        
    except Exception as e:
        import traceback
        print(f"\n❌ 脚本运行崩溃:")
        traceback.print_exc()

if __name__ == "__main__":
    dry_run()