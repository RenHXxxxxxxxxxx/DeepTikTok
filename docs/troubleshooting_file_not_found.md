# DeepTikTok Analysis System: "File Not Found" Troubleshooting & Repair Guide

## 1. Pathing Logic Disconnect

**The Symptom:** 
The front-end successfully scans and displays the file list from the `data/` directory. However, when triggering the "Refining Task" (精炼任务), the system throws a specific error: `Task Exception: Uploaded file not found` (`精炼任务异常: 未找到上传的文件`).

**Root Cause Analysis:**
There is a fundamental architecture mismatch between the **Data Warehouse View** and the **Refining API**:
*   **Front-end Visibility:** The `data_warehouse` view successfully reads the physical disk using directory iteration (`os.listdir(data_dir)`). This is why the UI accurately echoes the file queue visually.
*   **Back-end API Expectation:** The `run_clean_data_api` endpoint is hardcoded to validate and parse an HTTP Multipart form upload (`csv_file = request.FILES.get('file')`).
*   **The Mismatch:** Because you bypassed the web crawler and manually dropped the backed-up CSV files directly into the server's `data/` directory, no file stream is uploaded when you click the UI button. Consequently, `request.FILES` remains empty, triggering the hardcoded validation exception (`if not csv_file:`) and completely ignoring the physical files already residing on the disk.

## 2. Database Record Integrity

**The Symptom:**
The database was rebuilt, but manual placement of CSV files (approx. 3000 records) means physical media exists on the disk while the relational dataset is detached.

**Root Cause Analysis:**
*   **Absolute vs. Relative Paths:** The backed-up CSV files likely contain absolute paths mapped to a previous directory structure (e.g., `C:\backup\...`). When parsed on the new Windows environment (`D:\renhangxi_tiktok_bysj\...`), these legacy paths will cause any background workers (like the AI Analysis Worker) to fail when attempting to read `local_temp_path` or `video_file`.
*   **Missing Database Linking:** New MySQL records currently do not exist for these 3000 entries. The AI logic expects an `analysis_status=0` database row to initiate media processing. Without these rows being structurally generated via `update_or_create()`, the background daemon fundamentally ignores the orphans.

## 3. Process Ownership/Permissions

**The Symptom:**
Under Windows, files copied manually might inherit restricted security contexts.

**Root Cause Analysis:**
*   **NTFS Inheritance Issues:** Manually copying files from an external backup or another user profile can strip or alter necessary NTFS permissions.
*   **Worker Execution Context:** The Python process running the Django server (`manage.py runserver`) operates under your specific user context. If the copied files are set to strict "Read-Only" via inherited permissions, the backend scripts (especially the AI Worker that eventually executes `os.remove(local_path)` to free up disk space) will raise an `OSError` or `PermissionError` when attempting modifying operations.

## 4. Emergency Recovery Script: Force Synchronization

To bypass the UI upload constraints and safely inject the manually placed CSV files directly into the reconstructed database, execute the following script.

**Execution Instructions:**
1. Save the code below as `force_sync.py` in your project's Django directory (`D:\renhangxi_tiktok_bysj\renhangxi_tiktok_bysj\`).
2. Execute it within the Django environment: `python manage.py shell < force_sync.py`

```python
# *导入必要的 Django 上下文和库*
import os
import pandas as pd
from datetime import datetime
from django.db import transaction
from django.conf import settings
from douyin_hangxi.models import Video

# *全局配置映射字典*
GLOBAL_CONFIG = {
    'target_theme': '手工恢复主题_2026',
    'target_csv': 'video_backup.csv',  # *根据实际 data/ 目录下的 CSV 名字修改*
    'default_datetime': datetime.now(),
}

def execute_force_sync():
    data_dir = os.path.join(settings.BASE_DIR, 'data')
    csv_path = os.path.join(data_dir, GLOBAL_CONFIG['target_csv'])
    
    if not os.path.exists(csv_path):
        print(f"❌ *异常: 未在物理磁盘找到指定数据文件: {csv_path}*")
        return
        
    print(f"✅ *找到目标数据文件: {csv_path}，开始绕过UI执行物理同步入库...*")
    
    try:
        # *直接读取本地物理硬盘上的CSV，规避request.FILES上传逻辑*
        df = pd.read_csv(csv_path, dtype={'视频ID': str, 'video_id': str})
        
        # *映射列表与兼容性清洗*
        v_col_map = {
            '用户名': 'nickname', '粉丝数量': 'follower_count', '视频描述': 'desc', 
            '视频ID': 'video_id', '发表时间': 'create_time', '视频时长': 'duration', 
            '点赞数量': 'digg_count', '收藏数量': 'collect_count', '评论数量': 'comment_count',
            '下载数量': 'download_count', '本地路径': 'video_file'
        }
        df = df.rename(columns={k: v for k, v in v_col_map.items() if k in df.columns})
        
        df = df.fillna({
            'follower_count': 0, 'digg_count': 0, 'comment_count': 0, 
            'collect_count': 0, 'share_count': 0, 'visual_brightness': 0.0,
            'visual_saturation': 0.0, 'audio_bpm': 0, 'cut_frequency': 0.0
        })
        
        synced_count = 0
        
        # *使用事务级原子操作确保批量同步一致性，防止中断产生脏数据*
        with transaction.atomic():
            for _, row in df.iterrows():
                vid = str(row.get('video_id', ''))
                if not vid or vid.lower() == 'nan':
                    continue
                    
                # *利用 update_or_create 实现幂等写入机制*
                Video.objects.update_or_create(
                    video_id=vid,
                    defaults={
                        'theme_label': GLOBAL_CONFIG['target_theme'],
                        'nickname': str(row.get('nickname', '未知作者')),
                        'desc': str(row.get('desc', '')),
                        'follower_count': int(row.get('follower_count')),
                        'digg_count': int(row.get('digg_count')),
                        'comment_count': int(row.get('comment_count')),
                        'collect_count': int(row.get('collect_count')),
                        'share_count': int(row.get('share_count')),
                        'duration': str(row.get('duration', '00:00')),
                        'create_time': pd.to_datetime(row.get('create_time', GLOBAL_CONFIG['default_datetime'])),
                        'video_file': str(row.get('video_file', '')),
                        'visual_brightness': float(row.get('visual_brightness')),
                        'visual_saturation': float(row.get('visual_saturation')),
                        'audio_bpm': int(row.get('audio_bpm')),
                        'cut_frequency': float(row.get('cut_frequency')),
                    }
                )
                synced_count += 1
                
        print(f"🎉 *强制底层同步完成，脱离前台UI束缚，成功注册 {synced_count} 条物理数据记录。*")
        
    except Exception as e:
        print(f"🚨 *数据文件解析或底层持久化遇到异常: {str(e)}*")

if __name__ == "__main__":
    execute_force_sync()
```
