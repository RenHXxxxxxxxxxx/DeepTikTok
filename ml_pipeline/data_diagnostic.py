# -- coding: utf-8 --
import os
import sys
import django
from django.db.models import Count, Q

# ========== 全局配置 ==========
GLOBAL_CONFIG = {
    # 项目根目录
    "PROJECT_ROOT": r"d:\renhangxi_tiktok_bysj",
    # 不平衡分析阈值
    "MIN_SAMPLES_THRESHOLD": 200,
    # 目标变量分析阈值
    "VIRAL_THRESHOLD": 100000,
    "COLD_THRESHOLD": 100
}

# 加载 Django 环境
PROJECT_ROOT = GLOBAL_CONFIG["PROJECT_ROOT"]
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'renhangxi_tiktok_bysj.settings')

try:
    django.setup()
except Exception as e:
    print(f"[ERROR] *Django 初始化失败*: {e}")
    sys.exit(1)

from django.apps import apps

def run_diagnostic():
    """
    *运行数据质量诊断。*
    *1. 分析主题类别不平衡。*
    *2. 分析目标变量(点赞数)分布偏差。*
    """
    try:
        Video = apps.get_model('douyin_hangxi', 'Video')
        
        print("\n" + "="*60)
        print("*数据采集策略诊断报告 (Data Collection Strategy Report)*")
        print("="*60)
        
        total_videos = Video.objects.count()
        print(f"*[OVERALL] 数据库总视频样本数: {total_videos}*")
        
        if total_videos == 0:
            print("*[WARN] 数据库无数据，请先执行抓取!*")
            return
        
        # 1. 主题类别分布诊断
        print("\n*[1] 主题类别不平衡诊断 (Categorical Imbalance)*")
        print("-" * 50)
        
        theme_counts = Video.objects.values('theme_label').annotate(count=Count('video_id')).order_by('-count')
        min_threshold = GLOBAL_CONFIG['MIN_SAMPLES_THRESHOLD']
        
        needs_action = []
        for item in theme_counts:
            theme = item['theme_label']
            count = item['count']
            status = "*正常*" if count >= min_threshold else "*数据匮乏*"
            print(f"  - 主题: {theme: <15} | 样本量: {count: <5} | 状态: {status}")
            
            if count < min_threshold:
                needs_action.append((theme, count))
                
        # 2. 目标变量分布诊断 (Y-Distribution)
        print("\n*[2] 目标变量分布诊断 (Target Variable Bias)*")
        print("-" * 50)
        viral_t = GLOBAL_CONFIG['VIRAL_THRESHOLD']
        cold_t = GLOBAL_CONFIG['COLD_THRESHOLD']
        
        viral_count = Video.objects.filter(digg_count__gt=viral_t).count()
        cold_count = Video.objects.filter(digg_count__lt=cold_t).count()
        
        print(f"  - 爆款视频 (>10w赞): {viral_count} 个 (占 {viral_count/total_videos*100:.1f}%)")
        print(f"  - 冷门视频 (<100赞): {cold_count} 个 (占 {cold_count/total_videos*100:.1f}%)")
        
        # 行动建议输出
        print("\n*[ACTION REQUIRED] 执行策略:*")
        print("-" * 50)
        
        if needs_action:
            for theme, c in needs_action:
                shortage = min_threshold - c
                print(f"  * Warning: 极端长尾偏差。Action required: 请为主题 '{theme}' 补充抓取至少 {shortage} 个样本。*")
        else:
            print("  *[OK] 所有主题样本量均达标。*")
            
        if viral_count == 0:
            print(f"  * Warning: 缺少爆款特征表达。Action required: 请针对性抓取 >{viral_t // 1000}k 赞的头部视频。*")
        if cold_count == 0:
            print(f"  * Warning: 缺少冷门负样本特征。Action required: 请针对性抓取 <{cold_t} 赞的长尾视频。*")
        
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"[ERROR] *诊断过程发生异常: {e}*")

if __name__ == "__main__":
    run_diagnostic()

