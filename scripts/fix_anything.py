"""
*Fix Anything - TikTok System Unified Maintenance Tool*
*功能：状态恢复、孤儿文件回收、情感分重算、数据平衡报告*
"""
import os
import sys
import time
import argparse
import msvcrt
import django
from datetime import datetime

# ==========================================
# 环境引导与锁定
# ==========================================

# 配置项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'renhangxi_tiktok_bysj.settings')

# 全局配置
GLOBAL_CONFIG = {
    'LOCK_FILE': 'maintenance.lock',
    'PENDING_DIR': os.path.join(BASE_DIR, 'media', 'pending_videos'),
    'SENTINEL_BATCH_SIZE': 1000,
    'TARGET_PER_THEME': 5000,
}

class GlobalLock:
    """*文件级互斥锁，防止多实例运行*"""
    def __init__(self):
        self.lock_fp = open(GLOBAL_CONFIG['LOCK_FILE'], 'w')

    def __enter__(self):
        try:
            # 尝试获取排他锁，非阻塞模式
            msvcrt.locking(self.lock_fp.fileno(), msvcrt.LK_NBLCK, 1)
            print(f" *系统锁定成功 ({GLOBAL_CONFIG['LOCK_FILE']})*")
            return self
        except IOError:
            print(" *错误：另一个维护进程正在运行。请稍后再试。*")
            sys.exit(1)

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            msvcrt.locking(self.lock_fp.fileno(), msvcrt.LK_UNLCK, 1)
        except:
            pass
        self.lock_fp.close()
        # 删除锁文件 (可选，保留防止权限问题)
        # if os.path.exists(GLOBAL_CONFIG['LOCK_FILE']):
        #     os.remove(GLOBAL_CONFIG['LOCK_FILE'])

# 初始化 Django
try:
    django.setup()
    from django.db.models import Count
    from renhangxi_tiktok_bysj.douyin_hangxi.models import Video, Comment
    from renhangxi_tiktok_bysj.douyin_hangxi.views import clean_text_nuclear
    from snownlp import SnowNLP
except Exception as e:
    print(f" *Django 初始化失败: {e}*")
    sys.exit(1)


# ==========================================
# 功能模块
# ==========================================

def recover_state():
    """*1. 状态恢复：重置卡住的任务和清理坏文件*"""
    print("\n" + "="*50)
    print("*[模块 1] 状态恢复 (Recover State)*")
    print("="*50)
    
    # 清理 .part 文件
    if os.path.exists(GLOBAL_CONFIG['PENDING_DIR']):
        part_files = [f for f in os.listdir(GLOBAL_CONFIG['PENDING_DIR']) if f.endswith('.part')]
        if part_files:
            print(f"  *发现 {len(part_files)} 个未完成下载文件 (.part)，正在删除...*")
            for f in part_files:
                try:
                    os.remove(os.path.join(GLOBAL_CONFIG['PENDING_DIR'], f))
                except Exception as e:
                    print(f"     删除失败 {f}: {e}")
        else:
            print(" *无残留 .part 文件*")

    # 重置数据库状态
    stuck_videos = Video.objects.filter(analysis_status=1)
    count = stuck_videos.count()
    if count > 0:
        print(f" *发现 {count} 个卡在 'Processing(1)' 的视频，正在重置为 'Pending(0)'...*")
        stuck_videos.update(analysis_status=0)
        print(f" *已重置 {count} 条记录*")
    else:
        print(" *无卡顿任务*")


def handle_orphans():
    """*2. 孤儿文件处理：将未入库的视频导入系统*"""
    print("\n" + "="*50)
    print("*[模块 2] 孤儿文件回收 (Orphan Handling)*")
    print("="*50)

    if not os.path.exists(GLOBAL_CONFIG['PENDING_DIR']):
        print(" *Pending目录不存在，跳过*")
        return

    files = [f for f in os.listdir(GLOBAL_CONFIG['PENDING_DIR']) if f.endswith('.mp4')]
    print(f" *扫描目录: {GLOBAL_CONFIG['PENDING_DIR']}*")
    print(f" *发现 .mp4 文件数: {len(files)}*")

    recovered_count = 0
    for file_name in files:
        video_id = os.path.splitext(file_name)[0]
        
        # 检查数据库是否存在
        if not Video.objects.filter(video_id=video_id).exists():
            print(f" *发现孤儿文件: {file_name} -> 导入数据库*")
            try:
                Video.objects.create(
                    video_id=video_id,
                    nickname="Unknown_Orphan", # 占位符
                    theme_label="Recovered_Orphan",
                    desc="Recovered by fix_anything.py",
                    analysis_status=0, # 设置为待处理
                    local_temp_path=os.path.join(GLOBAL_CONFIG['PENDING_DIR'], file_name)
                )
                recovered_count += 1
            except Exception as e:
                print(f"     导入失败: {e}")
    
    if recovered_count == 0:
        print(" *未发现孤儿文件 (所有文件均已入库)*")
    else:
        print(f" *成功回收 {recovered_count} 个为 'Recovered_Orphan' 主题*")


def run_sentinel():
    """*3. 数据哨兵：批量修复情感分 (status=0.5)*"""
    print("\n" + "="*50)
    print("*[模块 3] 数据哨兵 (Sentinel Guard)*")
    print("="*50)

    # 查询目标
    targets = Comment.objects.filter(sentiment_score=0.5)
    total = targets.count()
    print(f" *发现情感分滞留 (Score=0.5) 记录数: {total}*")
    
    if total == 0:
        print(" *数据质量良好，无需修复*")
        return

    batch_size = GLOBAL_CONFIG['SENTINEL_BATCH_SIZE']
    processed = 0
    updated_batch = []

    print(f" *开始修复 (Batch Size: {batch_size})...*")
    start_time = time.time()

    # 使用 iterator 减少内存占用
    for comment in targets.iterator():
        # 复用 views.py 中的强力清洗逻辑
        cleaned = clean_text_nuclear(comment.content)
        
        # 重新计算
        score = 0.5
        label = "中性"
        is_valid = False

        if cleaned and len(cleaned) >= 1:
            try:
                s = SnowNLP(cleaned)
                score = s.sentiments
                label = "积极" if score > 0.6 else ("消极" if score < 0.4 else "中性")
                is_valid = True
            except:
                pass
        
        # 更新对象 (仅当分数改变或确实需要更新清洗字段时)
        comment.sentiment_score = score
        comment.sentiment_label = label
        comment.content_clean = cleaned
        updated_batch.append(comment)
        
        processed += 1
        
        # 批量提交
        if len(updated_batch) >= batch_size:
            Comment.objects.bulk_update(updated_batch, ['sentiment_score', 'sentiment_label', 'content_clean'])
            updated_batch = []
            print(f"     *已处理: {processed}/{total} ({(processed/total)*100:.1f}%)*")

    # 提交剩余
    if updated_batch:
        Comment.objects.bulk_update(updated_batch, ['sentiment_score', 'sentiment_label', 'content_clean'])
    
    duration = time.time() - start_time
    print(f" *修复完成! 耗时: {duration:.2f}s*")


def show_balance():
    """*4. 平衡报告：各主题采集进度*"""
    print("\n" + "="*50)
    print("*[模块 4] 数据平衡报告 (Balance Report)*")
    print("="*50)

    stats = Video.objects.values('theme_label').annotate(count=Count('video_id')).order_by('-count')
    target = GLOBAL_CONFIG['TARGET_PER_THEME']

    print(f"{'Theme Label':<30} | {'Count':<8} | {'Progress':<20}")
    print("-" * 65)
    
    total_videos = 0
    for s in stats:
        theme = s['theme_label']
        count = s['count']
        total_videos += count
        percent = min(100, (count / target) * 100)
        bar_len = int(percent / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        
        print(f"{theme:<30} | {count:<8} | {bar} {percent:.1f}%")

    print("-" * 65)
    print(f" *总视频数: {total_videos}*")


# ==========================================
# 主程序入口
# ==========================================

def show_menu():
    print("""
#############################################
#    TikTok System Maintenance Tool v1.0    #
#############################################
1.   Recover State    (修复卡顿任务 / 清理垃圾)
2.  Handle Orphans   (回收孤儿文件入库)
3.  Sentinel Guard   (重算 0.5 情感分)
4.  Balance Report   (查看采集进度)
5.  Run ALL          (执行所有维护任务)
0.  Exit
    """)

def main():
    parser = argparse.ArgumentParser(description="TikTok System Unified Maintenance Tool")
    parser.add_argument('--recover', action='store_true', help='Reset stuck videos and clean temp files')
    parser.add_argument('--orphans', action='store_true', help='Import orphaned .mp4 files')
    parser.add_argument('--sentinel', action='store_true', help='Recalculate neutral sentiment scores')
    parser.add_argument('--report', action='store_true', help='Show theme balance report')
    parser.add_argument('--all', action='store_true', help='Run all tasks')

    args = parser.parse_args()

    # 获取锁
    with GlobalLock():
        # CLI 模式
        if any([args.recover, args.orphans, args.sentinel, args.report, args.all]):
            if args.recover or args.all: recover_state()
            if args.orphans or args.all: handle_orphans()
            if args.sentinel or args.all: run_sentinel()
            if args.report or args.all: show_balance()
            return

        # 交互模式
        while True:
            show_menu()
            choice = input(" 请输入选项 [0-5]: ").strip()
            
            if choice == '1':
                recover_state()
            elif choice == '2':
                handle_orphans()
            elif choice == '3':
                run_sentinel()
            elif choice == '4':
                show_balance()
            elif choice == '5':
                recover_state()
                handle_orphans()
                run_sentinel()
                show_balance()
            elif choice == '0':
                print(" *Exiting...*")
                break
            else:
                print(" *无效选项，请重新输入*")
            
            input("\n按 Enter 键继续...")

if __name__ == '__main__':
    main()

