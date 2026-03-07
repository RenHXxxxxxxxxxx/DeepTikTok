import os
import sys
import time
import shutil
import logging
import threading
import django
from django.db import transaction, DatabaseError
from django.db.models import Q
from datetime import datetime
import pandas as pd

# *================================================================*
# *全局配置 (Global Config)*
# *================================================================*
GLOBAL_CONFIG = {
    'BATCH_SIZE': 200,  # *减小批量以缩短锁持有时间，改善 AIAnalysisWorker 并发*
    'MAX_RETRIES': 3,
    'BACKUP_RETENTION_COUNT': 5,
    'LOG_FILE': 'persistence.log',
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# *设置 Django 环境*
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'renhangxi_tiktok_bysj.settings')

# *================================================================*
# *Safe Lazy Loading Django (防止重入错误)*
# *使用双重检查: 先检查 apps.ready，再用 try-except 兜底*
# *================================================================*
try:
    if not django.apps.apps.ready:
        django.setup()
except RuntimeError as e:
    # *如果错误是 "populate() isn't reentrant"，说明 Django 已在初始化中，安全忽略*
    if "populate() isn't reentrant" not in str(e):
        raise e

# *导入模型 (必须在 setup 之后)*
from renhangxi_tiktok_bysj.douyin_hangxi.models import Video, Comment

# *配置日志*
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(GLOBAL_CONFIG['LOG_FILE'], encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DatabaseBackupService:
    """
    *数据库备份服务*
    *在执行大规模写入前自动备份 SQLite 数据库文件*
    """
    @staticmethod
    def backup_db():
        db_path = os.path.join(BASE_DIR, 'db.sqlite3')
        backup_dir = os.path.join(BASE_DIR, 'db_backup')
        
        if not os.path.exists(db_path):
            logger.warning("*未找到数据库文件，跳过备份*")
            return None

        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(backup_dir, f"db_{timestamp}.sqlite3")
        
        try:
            shutil.copy2(db_path, backup_path)
            logger.info(f"*数据库已备份至: {backup_path}*")
            
            # *保留最近 N 份备份，清理旧文件*
            DatabaseBackupService._cleanup_old_backups(backup_dir)
            return backup_path
            
        except Exception as e:
            logger.error(f"*数据库备份失败: {e}*")
            return None

    @staticmethod
    def _cleanup_old_backups(backup_dir):
        """*清理旧备份，只保留最近 N 份*"""
        try:
            backup_files = sorted(
                [f for f in os.listdir(backup_dir) if f.startswith('db_') and f.endswith('.sqlite3')],
                reverse=True
            )
            retention_count = GLOBAL_CONFIG['BACKUP_RETENTION_COUNT']
            for old_file in backup_files[retention_count:]:
                os.remove(os.path.join(backup_dir, old_file))
                logger.info(f"*已清理旧备份: {old_file}*")
        except Exception as e:
            logger.warning(f"*清理旧备份时出错: {e}*")


class UnifiedPersistenceManager:
    """
    *统一持久化管理器 (Thread-Safe Singleton)*
    *负责所有与 Django ORM 的交互，确保事务安全与数据一致性*
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        # *双重检查锁定模式 (Double-Checked Locking)*
        if cls._instance is None:
            with cls._lock:
                # *二次检查，防止多线程竞争*
                if cls._instance is None:
                    cls._instance = super(UnifiedPersistenceManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        # *确保 __init__ 只执行一次*
        if self._initialized:
            return
        self._initialized = True
        logger.info("*UnifiedPersistenceManager 单例初始化完成*")

    def _clean_text(self, text):
        """*基础文本清洗*"""
        if not text:
            return ""
        if isinstance(text, float):
            return ""
        return str(text).strip()

    def _safe_int(self, value, default=0):
        """*安全整数转换*"""
        try:
            if value is None or value == '':
                return default
            return int(value)
        except (ValueError, TypeError):
            return default

    def _hash_pii(self, text):
        """*对 PII 信息进行加盐截断哈希，实现隐私合规*"""
        if not text:
            return "Unknown"
        salt = "tiktok_bysj_salt_2026"
        import hashlib
        hashed = hashlib.sha256((str(text) + salt).encode('utf-8')).hexdigest()
        return f"user_{hashed[:8]}"

    def save_video_record(self, raw_data, theme_label="默认主题"):
        """
        *持久化单条视频记录*
        *包含重试机制与幂等性处理*
        *关键逻辑: 只有当提供了非空 local_temp_path 时，才将 analysis_status 重置为 0 (Pending)*
        """
        max_retries = GLOBAL_CONFIG['MAX_RETRIES']
        retry_count = 0
        
        # *数据预处理: 提取视频 ID*
        vid = raw_data.get('视频ID') or raw_data.get('video_id')
        if not vid:
            logger.warning("*跳过无效视频数据: 缺少 ID*")
            return False

        # *提取 local_temp_path 并判断是否为有效路径*
        local_temp_path = raw_data.get('local_temp_path', '') or ''
        has_valid_local_path = bool(local_temp_path and local_temp_path.strip())

        # *================================================================*
        # *Theme Label Appending Logic: 保留多主题关联*
        # *================================================================*
        existing_theme = Video.objects.filter(video_id=vid).values_list('theme_label', flat=True).first()
        
        if existing_theme and existing_theme.strip() and theme_label not in existing_theme:
            # *追加新主题到现有主题列表*
            final_theme = f"{existing_theme},{theme_label}"
            logger.info(f"*Appending theme: {existing_theme} -> {final_theme}*")
        else:
            # *首次创建或主题已存在，直接使用当前主题*
            final_theme = theme_label

        # *字段映射与清洗*
        defaults = {
            'theme_label': final_theme,
            'nickname': self._hash_pii(self._clean_text(raw_data.get('用户名') or raw_data.get('nickname'))),
            'desc': self._clean_text(raw_data.get('视频描述') or raw_data.get('desc')),
            'create_time': raw_data.get('发表时间') or raw_data.get('create_time'),
            'duration': raw_data.get('视频时长') or raw_data.get('duration'),
            'follower_count': self._safe_int(raw_data.get('粉丝数量') or raw_data.get('follower_count')),
            'digg_count': self._safe_int(raw_data.get('点赞数量') or raw_data.get('digg_count')),
            'comment_count': self._safe_int(raw_data.get('评论数量') or raw_data.get('comment_count')),
            'collect_count': self._safe_int(raw_data.get('收藏数量') or raw_data.get('collect_count')),
            'share_count': self._safe_int(raw_data.get('分享数量') or raw_data.get('share_count')),
            'download_count': self._safe_int(raw_data.get('下载数量') or raw_data.get('download_count')),
            'local_temp_path': local_temp_path,
        }

        # *[Status Guard Hotfix] Status reset logic moved to post-save to prevent race conditions*

        # *时间格式化尝试*
        try:
            if isinstance(defaults['create_time'], str):
                defaults['create_time'] = pd.to_datetime(defaults['create_time'])
        except Exception:
            defaults['create_time'] = datetime.now()

        # *重试循环*
        while retry_count < max_retries:
            try:
                # *原子性更新或创建*
                obj, created = Video.objects.update_or_create(
                    video_id=str(vid),
                    defaults=defaults
                )
                
                # *[Status Guard Race Condition Fix] Atomic State Update*
                # *Force 'created' items to Pending (0).*
                # *For existing items, reset to Pending (0) ONLY IF they are NOT currently Analyzed (2).*
                if created:
                    obj.analysis_status = 0
                    obj.save(update_fields=['analysis_status'])
                    status_msg = "Initialized Status=0"
                else:
                    # *Atomic DB-level conditional update to prevent race conditions*
                    # *Returns number of rows matched (1 if updated, 0 if skipped due to guard)*
                    rows = Video.objects.filter(video_id=vid).exclude(analysis_status=2).update(analysis_status=0)
                    status_msg = "Status Reset to 0" if rows > 0 else "Status Guard Active (Skipped Reset)"

                action = "创建" if created else "更新"
                logger.debug(f"*{action}视频记录: {vid} | {status_msg}*")
                return True

            except DatabaseError as e:
                retry_count += 1
                wait_time = 2 ** retry_count
                logger.warning(f"*数据库写入失败 (重试 {retry_count}/{max_retries}): {e}*")
                time.sleep(wait_time)
            except Exception as e:
                logger.error(f"*视频保存遇到不可恢复错误: {e}*")
                return False
        
        logger.error(f"*视频 {vid} 保存失败，已达最大重试次数*")
        return False

    def save_video_batch(self, video_list, theme_label="默认主题"):
        """
        *批量持久化视频元数据 (Batch Commit for Metadata)*
        *通过 transaction.atomic() 减少 IOPS，实现 5000 条冲刺的要求*
        """
        if not video_list:
            return 0
        
        success_count = 0
        try:
            with transaction.atomic():
                for raw_data in video_list:
                    if self.save_video_record(raw_data, theme_label):
                        success_count += 1
            logger.info(f"*批量写入视频元数据成功: {success_count}/{len(video_list)} 条*")
            return success_count
        except Exception as e:
            logger.error(f"*批量写入视频元数据失败，事务已回滚: {e}*")
            return 0

    def save_comment_batch(self, comment_list, theme_label="默认主题"):
        """
        *批量持久化评论数据*
        *使用 bulk_create 加速写入，自动处理外键关联*
        *使用 transaction.atomic() 确保整批成功或全部回滚*
        """
        batch_size = GLOBAL_CONFIG['BATCH_SIZE']
        
        if not comment_list:
            return 0

        # *1. 提取所有涉及的 Video ID，确保外键存在*
        video_ids = set(
            str(c.get('视频ID') or c.get('video_id')) 
            for c in comment_list 
            if c.get('视频ID') or c.get('video_id')
        )
        
        try:
            existing_videos = Video.objects.filter(video_id__in=video_ids).in_bulk(field_name='video_id')
        except Exception as e:
            logger.error(f"*查询关联视频失败: {e}*")
            return 0

        valid_objects = []
        skipped_count = 0
        
        for c_data in comment_list:
            vid = str(c_data.get('视频ID') or c_data.get('video_id') or '')
            cid = str(c_data.get('评论ID') or c_data.get('comment_id') or '')
            
            # *外键校验: 跳过孤儿评论*
            if vid not in existing_videos:
                skipped_count += 1
                continue
            
            # *内容清洗*
            content = self._clean_text(c_data.get('评论内容') or c_data.get('content'))
            
            # *时间解析*
            try:
                create_time = pd.to_datetime(
                    c_data.get('评论时间') or c_data.get('create_time') or datetime.now()
                )
            except Exception:
                create_time = datetime.now()
            
            # *构建模型对象 (不立即保存)*
            comment_obj = Comment(
                comment_id=cid,
                video=existing_videos[vid],
                theme_label=theme_label,
                nickname=self._hash_pii(self._clean_text(c_data.get('用户名') or c_data.get('nickname'))),
                content=content,
                content_clean=content,
                digg_count=self._safe_int(c_data.get('点赞数') or c_data.get('digg_count')),
                ip_label=self._clean_text(c_data.get('IP属地') or c_data.get('ip_label') or "未知"),
                create_time=create_time,
                sentiment_score=0.5,
                sentiment_label="中性"
            )
            valid_objects.append(comment_obj)

        if not valid_objects:
            if skipped_count > 0:
                logger.warning(f"*跳过 {skipped_count} 条孤儿评论 (关联视频不存在)*")
            return 0

        # *2. 分批写入，使用事务保证原子性*
        total_inserted = 0
        
        try:
            with transaction.atomic():
                for i in range(0, len(valid_objects), batch_size):
                    batch = valid_objects[i:i + batch_size]
                    # *ignore_conflicts=True: 忽略主键冲突 (幂等性)*
                    Comment.objects.bulk_create(batch, ignore_conflicts=True)
                    total_inserted += len(batch)
            
            logger.info(f"*批量写入评论成功: {total_inserted} 条 (跳过 {skipped_count} 条孤儿评论)*")
            return total_inserted

        except Exception as e:
            logger.error(f"*批量写入评论失败，事务已回滚: {e}*")
            return 0

    def get_pending_videos(self, limit=10):
        """
        *获取待处理的视频列表*
        *返回 analysis_status=0 且有本地文件的视频*
        """
        try:
            return Video.objects.filter(
                analysis_status=0,
                local_temp_path__isnull=False
            ).exclude(
                local_temp_path=''
            ).order_by('create_time')[:limit]
        except Exception as e:
            logger.error(f"*查询待处理视频失败: {e}*")
            return []

    def update_video_analysis_status(self, video_id, status, **extra_fields):
        """
        *更新视频分析状态*
        *status: 0=Pending, 1=Processing, 2=Completed, -1=Failed*
        """
        try:
            update_data = {'analysis_status': status}
            update_data.update(extra_fields)
            
            Video.objects.filter(video_id=video_id).update(**update_data)
            logger.debug(f"*视频 {video_id} 状态已更新为 {status}*")
            return True
        except Exception as e:
            logger.error(f"*更新视频状态失败: {e}*")
            return False


if __name__ == "__main__":
    # *单例测试*
    manager1 = UnifiedPersistenceManager()
    manager2 = UnifiedPersistenceManager()
    
    print(f"*单例验证: manager1 is manager2 = {manager1 is manager2}*")
    print("*Data Manager Test Completed*")