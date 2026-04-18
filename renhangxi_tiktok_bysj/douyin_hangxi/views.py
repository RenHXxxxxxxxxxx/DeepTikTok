import os
import sys
import uuid
import logging
import json
import re
import cv2
import jieba
import jieba.analyse
import random
import threading
import pandas as pd
from datetime import datetime
from collections import Counter

# 引入 SnowNLP 情感分析库
try:
    from snownlp import SnowNLP
except ImportError:
    # 提醒：请确保环境已通过 pip install snownlp 安装
    pass

from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login, logout
from django.conf import settings
from django.core.paginator import Paginator
from django.db import transaction
from django.db.models import Avg, Count, Sum, Max, F, Q
# [修复] 引入 Coalesce 以处理聚合计算中的空值问题
from django.db.models.functions import Coalesce
from django.core.cache import cache
from django.db.utils import OperationalError
import time

# 导入数据库模型
from .models import Video, Comment, CreatorConfig

# 导入多模态分析工具
try:
    from .utils.video_analyzer import VideoContentAnalyzer
    from .utils.llm_service import LLMService
    # [重构] 导入路径更新：predict_service 已迁移至 services/
    from services import predict_service
    # [重构] 主题基准统计引擎已迁移至 ml_pipeline/
    from ml_pipeline.theme_baseline_engine import calculate_display_theme_baseline
    # [重构] 统一持久化管理器已迁移至 services/
    from services.data_manager import UnifiedPersistenceManager
except ImportError:
    pass

from django.contrib import messages


logger = logging.getLogger(__name__)

# === 资产精炼工厂全局配置 ===
REFINING_CONFIG = {
    'DATA_DIR_NAME': 'data',
    'DEFAULT_THEME': '新导入主题'
}

# === 异步 AI 分析后台工作线程 ===
class AIAnalysisWorker(threading.Thread):
    """
    *后台守护线程：轮询数据库处理待分析视频*
    *架构说明：实现生产者-消费者模式中的消费者角色*
    *功能：*
        *1. 轮询 analysis_status=0 的视频记录*
        *2. 调用 VideoContentAnalyzer 提取多模态特征*
        *3. 将结果回写数据库并更新状态*
    """
    
    daemon = True  # 守护进程模式：主线程退出时自动终止
    
    # [NEW] 单视频处理超时阈值 (秒) - 超过此时间自动跳过
    SINGLE_VIDEO_TIMEOUT = 180  # 3分钟

    def _has_readable_video_tail(self, local_path):
        cap = cv2.VideoCapture(local_path)
        if not cap.isOpened():
            return False
        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if frame_count <= 0:
                return False

            first_ok, _ = cap.read()
            if not first_ok:
                return False

            tail_start = max(frame_count - 3, 0)
            for idx in range(tail_start, frame_count):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                tail_ok, _ = cap.read()
                if tail_ok:
                    return True
            return False
        finally:
            cap.release()
    
    def __init__(self):
        super().__init__(name='AIAnalysisWorker')
        self._stop_event = threading.Event()
        self._batch_size = 5  # 批处理大小：避免锁定数据库
        self._poll_interval = 2  # 轮询间隔（秒）
        self._processing_timestamps = {}  # 内存中追踪 Processing 状态开始时间
        self._processed_count = 0  # 处理完成的数量
        self.current_processing_file = "等待中..."  # 当前处理的文件名
        self.processing_durations = []  # 处理耗时列表
        self.MAX_WINDOW_SIZE = 50  # 统计窗口大小
    
    def stop(self):
        """*优雅停止工作线程*"""
        self._stop_event.set()
    
    def run(self):
        """*主循环：持续轮询并处理待分析视频*"""
        logger.info("[AIWorker] *后台分析线程已启动*")
        
        while not self._stop_event.is_set():
            try:
                # ================================================================
                # [Stuck Detection] 使用内存追踪检测卡住的视频
                # 如果视频在 Processing 状态超过 SINGLE_VIDEO_TIMEOUT 秒，重置为 Pending
                # ================================================================
                current_time = time.time()
                videos_to_reset = []
                
                for video_id, start_time in list(self._processing_timestamps.items()):
                    if current_time - start_time > self.SINGLE_VIDEO_TIMEOUT:
                        videos_to_reset.append(video_id)
                        del self._processing_timestamps[video_id]
                
                if videos_to_reset:
                    Video.objects.filter(
                        video_id__in=videos_to_reset,
                        analysis_status=1
                    ).update(analysis_status=-1)  # 标记为失败而非重试，防止无限循环
                    logger.warning(f"[AIWorker] *超时任务: {videos_to_reset}*")
                
                # 原子获取待处理的视频（使用 select_for_update 避免并发竞态）
                with transaction.atomic():
                    # 仅消费“文件已落盘”的任务，避免下载前提前分析导致 File Missing 噪音
                    ready_qs = Video.objects.filter(
                        analysis_status=0,
                        local_temp_path__isnull=False
                    ).exclude(local_temp_path='')
                    total_pending = ready_qs.count()
                    video = ready_qs.select_for_update(skip_locked=True).first()
                    
                    global_status = cache.get('global_pipeline_status', {"status": "idle", "global_phase": "idle", "video": {"c": 0, "t": 0}, "comment": {"c": 0, "t": 0}, "ai": {"c": 0, "t": 0}, "msg": ""})
                    
                    if not video:
                        if global_status.get("global_phase") == "ai_processing" and total_pending == 0:
                            global_status["status"] = "finished"
                            global_status["global_phase"] = "finished"
                            global_status["msg"] = "AI 分析完成"
                            cache.set('global_pipeline_status', global_status, timeout=3600)
                        # 无任务时休眠，降低数据库压力
                        time.sleep(self._poll_interval)
                        continue
                    
                    global_status["status"] = "running"
                    global_status["global_phase"] = "ai_processing"
                    global_status["ai"] = {"c": getattr(self, '_processed_count', 0), "t": total_pending + getattr(self, '_processed_count', 0)}
                    global_status["msg"] = "AI 正在分析视频特征..."
                    cache.set('global_pipeline_status', global_status, timeout=3600)
                    
                    # 认领任务：立即更新状态为处理中 (1)
                    video.analysis_status = 1
                    video.save(update_fields=['analysis_status'])
                
                # 锁释放，开始执行耗时 AI 分析
                if self._stop_event.is_set():
                    break
                self._process_single_video(video)
                    
            except Exception as e:
                logger.error(f"[AIWorker] *主循环异常: {e}*")
                time.sleep(self._poll_interval)
        
        logger.info("[AIWorker] *后台分析线程已停止*")
    
    def _process_single_video(self, video):
        """
        *处理单个视频的 AI 分析任务*
        *参数: video - Video 模型实例*
        """
        local_path = video.local_temp_path  # 先获取路径供后续清理使用
        
        try:
            # Step 1: 状态已在外层原子事务中被标记为处理中 (1=Processing)
            
            # [NEW] 记录处理开始时间，用于超时检测
            start_t = time.time()
            self._processing_timestamps[video.video_id] = start_t
            if local_path:
                self.current_processing_file = os.path.basename(local_path)
            else:
                self.current_processing_file = str(video.video_id)
            
            logger.info(f"[AIWorker] *开始处理视频: {video.video_id}*")
            
            # Step 2: 校验本地临时文件是否存在
            if not local_path or not os.path.exists(local_path):
                raise FileNotFoundError(f"*视频文件不存在: {local_path}*")
            
            # === Step 2.1: 文件完整性检查（垃圾过滤器）===
            # 过滤掉小于 10KB 的损坏/不完整文件
            MIN_FILE_SIZE_BYTES = 10 * 1024  # 10KB 阈值
            try:
                file_size = os.path.getsize(local_path)
                if file_size < MIN_FILE_SIZE_BYTES:
                    logger.warning(f"[AIWorker] *文件过小或损坏 ({file_size} bytes): {video.video_id}*")
                    video.analysis_status = -1
                    video.save(update_fields=['analysis_status'])
                    # 立即删除损坏文件以释放磁盘空间
                    try:
                        os.remove(local_path)
                        video.local_temp_path = ""
                        video.save(update_fields=['local_temp_path'])
                    except OSError as del_err:
                        logger.warning(f"[AIWorker] *删除损坏文件失败: {del_err}*")
                    return  # 跳过处理
            except OSError as size_err:
                logger.warning(f"[AIWorker] *获取文件大小失败: {size_err}*")
                raise FileNotFoundError(f"*无法读取文件信息: {local_path}*")
            
            # === Step 2.2: 写入稳定性检查（防抖器）===
            # 确保文件已完全写入磁盘，未在持续增长中
            MAX_STABILITY_RETRIES = 3  # 最大重试次数
            STABILITY_INTERVAL = 1.0   # 检查间隔（秒）
            file_is_stable = False
            
            for retry in range(MAX_STABILITY_RETRIES):
                try:
                    current_size = os.path.getsize(local_path)
                    time.sleep(STABILITY_INTERVAL)
                    new_size = os.path.getsize(local_path)
                    
                    if current_size == new_size:
                        # 文件大小稳定，可以安全处理
                        file_is_stable = True
                        break
                    else:
                        logger.info(f"[AIWorker] *文件仍在写入中，等待重试 {retry + 1}/{MAX_STABILITY_RETRIES}: {video.video_id}*")
                except OSError as stable_err:
                    logger.warning(f"[AIWorker] *稳定性检查失败: {stable_err}*")
                    break
            
            if not file_is_stable:
                # 文件持续变化，跳过本轮处理（下次轮询再处理）
                logger.info(f"[AIWorker] *文件持续变化，延迟处理: {video.video_id}*")
                # 重置状态为 Pending (0)，不标记为失败
                video.analysis_status = 0
                video.save(update_fields=['analysis_status'])
                return  # 跳过本轮，下次轮询再处理

            if not self._has_readable_video_tail(local_path):
                logger.warning(f"[AIWorker] *尾帧完整性校验失败: {video.video_id}*")
                video.analysis_status = -1
                video.save(update_fields=['analysis_status'])
                try:
                    os.remove(local_path)
                    video.local_temp_path = ""
                    video.save(update_fields=['local_temp_path'])
                except OSError as del_err:
                    logger.warning(f"[AIWorker] *删除截断文件失败: {del_err}*")
                return
            
            # Step 3: 调用多模态分析器提取特征
            try:
                analyzer = VideoContentAnalyzer(local_path, video_id=video.video_id)
                ai_features = analyzer.run_full_analysis(include_script_keywords=False)
            except Exception as analyze_err:
                raise RuntimeError(f"*分析器执行失败: {analyze_err}*")
            
            # Step 4: 将 AI 特征回写数据库
            video.visual_brightness = float(ai_features.get('visual_brightness', 0.0) or 0.0)
            video.visual_saturation = float(ai_features.get('visual_saturation', 0.0) or 0.0)
            video.audio_bpm = int(ai_features.get('audio_bpm', 0) or 0)
            video.cut_frequency = float(ai_features.get('cut_frequency', 0.0) or 0.0)
            
            # Step 5: 标记状态为完成 (2=Completed)
            video.analysis_status = 2
            
            logger.info(f"[AIWorker] *视频分析完成: {video.video_id}*")
            
            # 性能指标记录
            end_t = time.time()
            duration = end_t - start_t
            self.processing_durations.append(duration)
            if len(self.processing_durations) > self.MAX_WINDOW_SIZE:
                self.processing_durations.pop(0)
                
            try:
                ts_list = cache.get('ai_throughput_ts', [])
                current_t = time.time()
                ts_list.append(current_t)
                ts_list = [ts for ts in ts_list if current_t - ts <= 300]
                cache.set('ai_throughput_ts', ts_list, timeout=3600)
            except Exception:
                pass
            
            # Step 6: 磁盘空间回收（核心新增逻辑）
            # 分析完成后立即删除视频文件，释放磁盘空间
            try:
                if local_path and os.path.exists(local_path):
                    os.remove(local_path)
                    video.local_temp_path = ""  # 更新数据库标记文件已清理
                    logger.info(f"[AIWorker] *已清理临时文件: {local_path}*")
            except OSError as cleanup_err:
                logger.warning(f"[AIWorker]   Cleanup Failed: {cleanup_err}")
            
        except FileNotFoundError as fnf_err:
            # 文件不存在：标记为失败
            logger.warning(f"[AIWorker] *文件缺失: {video.video_id} - {fnf_err}*")
            video.analysis_status = -1
            if hasattr(video, 'error_msg'):
                video.error_msg = str(fnf_err)
            
        except Exception as e:
            # 其他异常：标记为失败 (-1=Failed)
            logger.error(f"[AIWorker] *处理失败: {video.video_id} - {e}*")
            video.analysis_status = -1
            if hasattr(video, 'error_msg'):
                video.error_msg = str(e)
        
        finally:
            self._processed_count = getattr(self, '_processed_count', 0) + 1
            # [NEW] 清理内存中的处理时间戳记录
            self._processing_timestamps.pop(video.video_id, None)
            
            # [NEW] 释放资源和统一事务保存
            if 'cap' in locals() and cap:
                try:
                    cap.release()
                except:
                    pass
            import gc
            gc.collect()
            
            try:
                with transaction.atomic():
                    video.save()
            except Exception as save_err:
                logger.error(f"[AIWorker] *最终状态保存失败: {save_err}*")


# === 全局工作线程实例（可选：由 Django AppConfig 启动）===
_ai_worker_instance = None


def start_ai_worker():
    """*启动全局 AI 分析后台线程（确保单例）*"""
    # [Dead-letter Recovery] 服务器启动时，将所有孤儿 Processing(1) 记录重置为 Pending(0)
    # 场景：服务器崩溃后 status=1 的记录永远不会被消费，形成"死信"
    try:
        orphaned_count = Video.objects.filter(analysis_status=1).update(analysis_status=0)
        if orphaned_count > 0:
            logger.warning(f"[AIWorker] *Dead-letter Recovery: 已将 {orphaned_count} 个孤儿任务重置为待处理*")
    except Exception as recovery_err:
        logger.error(f"[AIWorker] *Dead-letter Recovery 失败: {recovery_err}*")

    global _ai_worker_instance
    if _ai_worker_instance is None or not _ai_worker_instance.is_alive():
        _ai_worker_instance = AIAnalysisWorker()
        _ai_worker_instance.start()
        logger.info("[AIWorker] *全局工作线程已创建并启动*")
    return _ai_worker_instance


def stop_ai_worker():
    """*停止全局 AI 分析后台线程*"""
    global _ai_worker_instance
    if _ai_worker_instance and _ai_worker_instance.is_alive():
        _ai_worker_instance.stop()
        _ai_worker_instance.join(timeout=5)
        logger.info("[AIWorker] *全局工作线程已停止*")
    _ai_worker_instance = None


# 深度文本清洗函数：过滤社交媒体噪声，提高情感特征提取精度
def clean_text_nuclear(text):
    if not isinstance(text, str):
        return ""
    # 递归剔除回复路径和提及信息
    text = re.sub(r'回复 @.*?:', '', text)
    text = re.sub(r'@\S+', '', text)
    # 剔除超链接
    text = re.sub(r'https?://\S+', '', text)
    # 仅保留中文字符、字母及数字，移除无意义符号
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text)
    return text.strip()


def extract_semantic_features(text, topK=5):
    """
    *使用 TF-IDF 算法提取文本核心语义特征，实现降维*
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    # 提取关键词
    tags = jieba.analyse.extract_tags(text, topK=topK)
    # 如果未能提取出有效特征，则回退为原清洗文本
    if not tags:
        return text.strip()
    # 以空格连接作为最终清洗后内容 (content_clean)
    return " ".join(tags)


def calculate_refined_sentiment(text):
    """
    *精细化情感计算核心引擎*
    *特点：*
    *1. 提供显性异常处理，异常时得分为 0.5001 区分自然中性。*
    *2. 自适应阈值划分，适配短词特征提取后的极化。*
    """
    try:
        s = SnowNLP(text)
        score = s.sentiments
    except Exception as e:
        logger.warning(f"[Sentiment] 计算异常: {e}, text: {text}")
        score = 0.5001

    if score > 0.8:
        label = "非常积极"
    elif score > 0.55:
        label = "积极"
    elif score >= 0.45:
        label = "中性"
    elif score >= 0.2:
        label = "消极"
    else:
        label = "非常消极"
        
    return score, label


# === 情感分析服务专用文本清洗工具函数 ===
def _clean_text_service(text):
    """
    *文本清洗服务函数：用于情感分析前的文本预处理*
    *功能：去除 @提及、URL超链接、表情包等社交媒体噪声*
    *返回：仅保留清洗后的中英文文本和数字*
    """
    # 类型防御性检查
    if not isinstance(text, str):
        return ""
    
    # 去除回复路径（如"回复 @用户名:"）
    text = re.sub(r'回复\s*@.*?[:：]', '', text)
    
    # 去除所有 @提及信息
    text = re.sub(r'@\S+', '', text)
    
    # 去除 URL 超链接
    text = re.sub(r'https?://\S+', '', text)
    
    # 去除表情包标签（如 [大笑]、[捂脸] 等）
    text = re.sub(r'\[.*?\]', '', text)
    
    # 去除 Emoji 表情符号（Unicode范围）
    text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', '', text)
    
    # 仅保留中文、英文字母及数字
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text)
    
    return text.strip()


# === 核心数据入库服务函数 (Refactored) ===
def import_data_service(theme_name, video_csv_path, comment_csv_path):
    """
    *核心数据入库服务函数*
    *功能：读取爬虫生成的CSV文件，使用 UnifiedPersistenceManager 写入数据库*
    """
    result = {'video_count': 0, 'comment_count': 0, 'success': False, 'message': ''}
    
    def _safe_vid(value):
        if value is None:
            return ""
        text = str(value).strip()
        if text.lower() in {"", "nan", "none", "null"}:
            return ""
        return text
    
    try:
        manager = UnifiedPersistenceManager()
        
        # === 第一阶段：视频数据入库 ===
        if not video_csv_path or not os.path.exists(video_csv_path):
            result['message'] = f"视频文件不存在: {video_csv_path}"
            return result
        
        try:
            # 采用 chunksize 分块读取以优化内存占用
            is_empty = True
            for df_video_chunk in pd.read_csv(video_csv_path, dtype={'视频ID': str, 'video_id': str}, chunksize=5000):
                if df_video_chunk.empty:
                    continue
                is_empty = False
                
                # 遍历视频数据并入库
                # 直接复用 manager.save_video_record 的逻辑
                # 需要将 DataFrame 行转换为字典格式
                video_records = df_video_chunk.to_dict('records')
                
                extracted_ids = [
                    _safe_vid(r.get('video_id') or r.get('视频ID'))
                    for r in video_records
                ]
                extracted_ids = [vid for vid in extracted_ids if vid]
                existing_active_vids = set(Video.objects.filter(video_id__in=extracted_ids, analysis_status__in=[1, 2]).values_list('video_id', flat=True))
                
                with transaction.atomic():
                    for record in video_records:
                        # === 幂等性检查 (Idempotency Check) ===
                        # 防止导入服务在数据分析中途重新排队，造成"File Missing"竞态条件
                        # Status Guard: 仅跳过 处理中(1) 和 已完成(2) 的视频
                        # 待处理(0) 和 失败(-1) 的视频允许重试
                        video_id = _safe_vid(record.get('video_id') or record.get('视频ID'))
                        if not video_id:
                            continue
            
                        try:
                            if video_id in existing_active_vids:
                                logger.info(f"[入库] ⏭ 跳过活跃视频: {video_id} (Batch Skipped)")
                                continue
                        except Exception as check_err:
                            logger.warning(f"[入库]  状态检查失败，将尝试强制更新: {video_id} - {check_err}")
            
                        if manager.save_video_record(record, theme_name):
                            result['video_count'] += 1
                        
            if is_empty:
                result['message'] = "视频CSV文件为空"
                return result
        except Exception as e:
            result['message'] = f"读取视频CSV失败: {e}"
            return result
        
        logger.info(f"[入库] 视频入库完成: {result['video_count']} 条")
        
        # === 第二阶段：评论数据入库 ===
        if not comment_csv_path or not os.path.exists(comment_csv_path):
            result['success'] = True
            result['message'] = f"视频入库成功 {result['video_count']} 条，评论文件不存在"
            return result
        
        try:
            is_empty = True
            for df_comment_chunk in pd.read_csv(comment_csv_path, dtype={'视频ID': str, 'video_id': str, '评论ID': str, 'comment_id': str}, chunksize=5000):
                if df_comment_chunk.empty:
                    continue
                is_empty = False
                
                # 批量处理评论
                # manager.save_comment_batch 接收字典列表
                comment_records = df_comment_chunk.to_dict('records')
                
                # 按 1000 条切割分批传入
                BATCH_SIZE = 1000
                for i in range(0, len(comment_records), BATCH_SIZE):
                    batch = comment_records[i:i + BATCH_SIZE]
                    inserted = manager.save_comment_batch(batch, theme_name)
                    result['comment_count'] += inserted
                    
            if is_empty:
                result['success'] = True
                result['message'] = f"视频入库成功 {result['video_count']} 条，评论CSV为空"
                return result
                
        except Exception as e:
            result['message'] = f"视频入库成功，但读取评论CSV失败: {e}"
            result['success'] = True
            return result
            
        result['success'] = True
        result['message'] = f"入库完成：视频 {result['video_count']} 条，评论 {result['comment_count']} 条"
        return result

    except Exception as e:
        logger.error(f"[入库] 服务异常: {e}")
        result['message'] = f"入库服务异常: {e}"
        return result



# 获取全局主题上下文
def get_theme_context(request):
    active_theme = request.session.get('active_theme', '默认主题')
    themes_list = list(Video.objects.values_list('theme_label', flat=True).distinct())
    if '默认主题' not in themes_list:
        themes_list.append('默认主题')
    return active_theme, themes_list


# 数据资产仓库管理视图
@login_required
def data_warehouse(request):
    active_theme, themes_list = get_theme_context(request)
    PROJECT_ROOT = settings.BASE_DIR
    # [修复] 读取项目 data/ 目录获取中文命名的视频 CSV
    data_dir = os.path.join(PROJECT_ROOT, 'data')
    video_files = []
    comment_files = []

    if os.path.exists(data_dir):
        all_csvs = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        video_files = [f for f in all_csvs if 'video' in f]
        comment_files = [f for f in all_csvs if 'comment' in f or 'refined' in f]

    # 统计主题资产规模
    themes_info = Video.objects.values('theme_label').annotate(
        video_count=Count('video_id'),
        avg_digg=Avg('digg_count')
    ).order_by('-video_count')

    return render(request, 'data/warehouse.html', {
        'themes_info': themes_info, 'active_theme': active_theme,
        'themes_list': themes_list, 'video_files': video_files,
        'comment_files': comment_files
    })


# 数据精炼 API：实施强力噪声物理拦截
@login_required
def run_clean_data_api(request):
    # 处理上传的CSV文件流或处理本地物理挂载路径并以事务方式存入数据库
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': '仅支持 POST 请求'})

    # 获取目标主题，兼容前端传入的 target_theme 或 theme_name
    target_theme = request.POST.get('target_theme') or request.POST.get('theme_name', REFINING_CONFIG['DEFAULT_THEME'])
    target_theme = target_theme.strip()

    # 获取文件参数，检查 video_csv/comment_csv 并兼容 target_file/target_comment_file
    video_csv = request.POST.get('video_csv') or request.POST.get('target_file')
    comment_csv = request.POST.get('comment_csv') or request.POST.get('target_comment_file')
    csv_file = request.FILES.get('file')

    if not video_csv and not comment_csv and not csv_file:
        return JsonResponse({'status': 'error', 'message': '未提供任何数据源文件或路径'})

    try:
        video_count = 0
        comment_count = 0
        
        # 阶段一：处理视频 CSV (逻辑解耦，支持本地物理路径)
        if video_csv or csv_file:
            if video_csv:
                # 利用 settings.BASE_DIR 构建绝对物理路径
                video_path = os.path.join(settings.BASE_DIR, REFINING_CONFIG['DATA_DIR_NAME'], video_csv)
                # 安全校验物理文件是否真实存在
                if not os.path.exists(video_path):
                    return JsonResponse({'status': 'error', 'message': f'视频物理文件未找到: {video_path}'})
                # Update the Pandas reading logic to handle local path strings.
                df = pd.read_csv(video_path, dtype={'视频ID': str, 'video_id': str})
            else:
                # 兼容传统的 Upload Stream 方式
                df = pd.read_csv(csv_file, dtype={'视频ID': str, 'video_id': str})
            
            # 列名动态映射，增强健壮性
            v_col_map = {
                '用户名': 'nickname', '粉丝数量': 'follower_count', '视频描述': 'desc', 
                '视频ID': 'video_id', '发表时间': 'create_time', '视频时长': 'duration', 
                '点赞数量': 'digg_count', '收藏数量': 'collect_count', '评论数量': 'comment_count',
                '下载数量': 'download_count', '本地路径': 'video_file', '分享数量': 'share_count'
            }
            df = df.rename(columns={k: v for k, v in v_col_map.items() if k in df.columns})

            # 全局空值填补防范数据结构破坏
            df = df.fillna({
                'follower_count': 0, 'digg_count': 0, 'comment_count': 0, 
                'collect_count': 0, 'share_count': 0, 'visual_brightness': 0.0,
                'visual_saturation': 0.0, 'audio_bpm': 0, 'cut_frequency': 0.0
            })

            # 安全的类型转换助手，防范 NaN 及 None
            safe_int = lambda x: int(float(x)) if x is not None and str(x).lower() != 'nan' else 0

            # 确保批量更新过程具备原子性
            with transaction.atomic():
                for _, row in df.iterrows():
                    vid = str(row.get('video_id', ''))
                    if not vid or vid.lower() == 'nan':
                        continue

                    # 根据指定的主键实现覆盖式的幂等创建，深度绑定用户输入的 target_theme
                    Video.objects.update_or_create(
                        video_id=vid,
                        defaults={
                            'theme_label': target_theme,
                            'nickname': str(row.get('nickname', '未知作者')),
                            'desc': str(row.get('desc', '')),
                            'follower_count': safe_int(row.get('follower_count')),
                            'digg_count': safe_int(row.get('digg_count')),
                            'comment_count': safe_int(row.get('comment_count')),
                            'collect_count': safe_int(row.get('collect_count')),
                            'share_count': safe_int(row.get('share_count')),
                            'duration': str(row.get('duration', '00:00')),
                            'create_time': pd.to_datetime(row.get('create_time', datetime.now())),
                            'video_file': str(row.get('video_file', '')),
                            'visual_brightness': float(row.get('visual_brightness')),
                            'visual_saturation': float(row.get('visual_saturation')),
                            'audio_bpm': safe_int(row.get('audio_bpm')),
                            'cut_frequency': float(row.get('cut_frequency')),
                        }
                    )
            video_count = len(df)
            
        # 阶段二：处理关联的评论 CSV (解耦逻辑，直接基于物理路径)
        if comment_csv:
            comment_path = os.path.join(settings.BASE_DIR, REFINING_CONFIG['DATA_DIR_NAME'], comment_csv)
            # 校验评论物理文件是否存在
            if not os.path.exists(comment_path):
                return JsonResponse({'status': 'error', 'message': f'评论物理文件未找到: {comment_path}'})
            
            # 解析本地物理文件，分块流式读取处理
            for df_comment_chunk in pd.read_csv(comment_path, dtype={'视频ID': str, 'video_id': str, '评论ID': str, 'comment_id': str}, chunksize=5000):
                if df_comment_chunk.empty:
                    continue
                comment_records = df_comment_chunk.to_dict('records')
                # 依赖UnifiedPersistenceManager进行批量持久化
                manager = UnifiedPersistenceManager()
                inserted = manager.save_comment_batch(comment_records, target_theme)
                comment_count += inserted

        return JsonResponse({'status': 'success', 'message': f'资产精炼执行完毕，同步视频 {video_count} 条，评论 {comment_count} 条。'})

    except OperationalError:
        # 处理SQLite因并发写引发的写锁错误
        return JsonResponse({"status": "error", "message": "Database is currently busy (Locked). Please try again."}, status=503)
    except Exception as e:
        # 捕获任何因为无效脏数据造成的崩溃点并拒绝保存
        logger.error(f"[Refining] *数据精炼流程异常崩溃: {e}*")
        return JsonResponse({"status": "error", "message": f"Transaction rolled back due to invalid data: {str(e)}"}, status=400)


# 情感分重算接口：强制全量修复 (Retroactive Repair)
@login_required
def recalculate_sentiment_api(request):
    """
    *修复工具：强制重算指定主题的所有评论情感分*
    *1. 遍历所有评论（包括之前被标记为 0.5 的数据）*
    *2. 使用新规则（len >= 1）重新清洗*
    *3. 重新计算 SnowNLP 得分并覆盖入库*
    """
    if request.method != 'POST': return JsonResponse({'status': 'error'})
    theme_label = request.POST.get('theme_label')
    
    # 获取该主题下所有评论
    comments = Comment.objects.filter(theme_label=theme_label)
    total_count = comments.count()
    if total_count == 0: 
        return JsonResponse({'status': 'error', 'message': '当前主题无可用数据'})

    updated_batch = []
    processed_count = 0
    
    # 使用 iterator 避免内存溢出
    for c in comments.iterator():
        # 重置清洗逻辑：先物理去噪，再利用 TF-IDF 提取核心语义特征
        cleaned_raw = clean_text_nuclear(c.content)
        target_text = extract_semantic_features(cleaned_raw)

        # 逻辑修正：放宽长度限制 (>=1)
        if not target_text.strip() or len(target_text.strip()) < 1:
            # 对于确实无效的（空字符串），可以保持 0.5 或标记为无效，这里保持原样但跳过更新
            continue

        score, label = calculate_refined_sentiment(target_text)

        # 更新对象属性
        c.sentiment_score = float(score)
        c.sentiment_label = str(label)
        c.content_clean = target_text
        
        updated_batch.append(c)
        processed_count += 1
        
        # 批量提交
        if len(updated_batch) >= 500:
            Comment.objects.bulk_update(updated_batch, ['sentiment_score', 'sentiment_label', 'content_clean'])
            updated_batch = []
    
    # 提交剩余批次
    if updated_batch: 
        Comment.objects.bulk_update(updated_batch, ['sentiment_score', 'sentiment_label', 'content_clean'])

    return JsonResponse({
        'status': 'success', 
        'message': f'修复完成：已重算 {processed_count}/{total_count} 条评论的情感得分'
    })


# 解除删除封锁：允许物理删除包括“默认主题”在内的任何数据包
@login_required
def delete_theme(request):
    if request.method != 'POST': return JsonResponse({'status': 'error'})
    theme_label = request.POST.get('theme_label')

    # 核心修改：移除对“默认主题”的保护性判断，允许用户进行彻底物理清理
    Video.objects.filter(theme_label=theme_label).delete()
    Comment.objects.filter(theme_label=theme_label).delete()

    # 如果删除的是当前激活的主题，则重置 Session 为默认
    if request.session.get('active_theme') == theme_label:
        request.session['active_theme'] = '默认主题'

    return JsonResponse({'status': 'success', 'message': f'主题【{theme_label}】数据已彻底从物理磁盘中移除'})


# 在 Session 中切换当前工作主题
@login_required
def switch_theme(request):
    if request.method != 'POST': return JsonResponse({'status': 'error'})
    request.session['active_theme'] = request.POST.get('theme_label', '默认主题')
    return JsonResponse({'status': 'success'})


# 主面板：展示核心统计看板
@login_required
def dashboard(request):
    # [NEW] Worker Guard: Ensure ai worker is running
    try:
        start_ai_worker()
    except Exception as e:
        logger.warning(f"[DashboardWorkerGuard] {e}")

    active_theme, themes_list = get_theme_context(request)
    current_videos = Video.objects.filter(theme_label=active_theme)

    # 1. 计算情感平均分，排除中性噪声
    avg_sent =\
        Comment.objects.filter(theme_label=active_theme).exclude(sentiment_score=0.5).aggregate(Avg('sentiment_score'))[
            'sentiment_score__avg'] or 0.0

    # 2. 计算雷达图所需的平均统计数据
    # 常规互动数据 (包含所有视频)
    avg_stats = current_videos.aggregate(
        avg_digg=Avg('digg_count'),
        avg_comment=Avg('comment_count'),
        avg_collect=Avg('collect_count'),
        avg_share=Avg('share_count')
    )

    # AI 多模态特征数据 (【核心修复】仅统计真正被AI成功分析的视频，防止零值稀释)
    ai_videos = current_videos.filter(analysis_status=2)
    ai_stats = ai_videos.aggregate(
        avg_bright=Avg('visual_brightness'),
        avg_sat=Avg('visual_saturation'),
        avg_bpm=Avg('audio_bpm'),
        avg_cut_freq=Avg('cut_frequency')
    )
    # 合并字典供后续统一调用
    avg_stats.update(ai_stats)

    # 3. 计算历史互动总量 (点赞总和 + 评论总和)
    sum_stats = current_videos.aggregate(
        sum_digg=Coalesce(Sum('digg_count'), 0),
        sum_comment=Coalesce(Sum('comment_count'), 0)
    )
    history_interactions = sum_stats['sum_digg'] + sum_stats['sum_comment']

    # 4. 获取最新评论列表 (用于前台舆情穿透展示)
    recent_comments = Comment.objects.filter(
        theme_label=active_theme
    ).exclude(sentiment_score=0.5).order_by('-create_time')[:10]

    # 序列化评论数据供前端 JS 使用
    comments_data = []
    for c in recent_comments:
        comments_data.append({
            "content": c.content,
            "content_clean": c.content_clean,
            "sentiment_score": c.sentiment_score,
            "ip_label": getattr(c, 'ip_label', '未知'),
            "create_time": c.create_time.strftime("%Y-%m-%d %H:%M"),
            "nickname": c.nickname,
            "digg_count": c.digg_count
        })


    # 6. 计算 AI 分析进度百分比 (供前端进度条展示)
    total_video_count = current_videos.count()
    completed_analysis_count = current_videos.filter(analysis_status=2).count()
    analysis_progress = round((completed_analysis_count / max(1, total_video_count)) * 100, 1)

    return render(request, 'dashboard.html', {
        'total_videos': total_video_count,
        'total_comments': Comment.objects.filter(theme_label=active_theme).count(),
        'avg_sentiment': round(avg_sent, 2),
        'active_theme': active_theme,
        'themes_list': themes_list,
        'avg_stats': avg_stats,
        'history_interactions': history_interactions,
        'recent_comments': recent_comments,
        'recent_comments_json': json.dumps(comments_data),
        'analysis_progress': analysis_progress,
        'completed_analysis_count': completed_analysis_count,
    })


# 视频档案分页列表
@login_required
def video_list(request):
    active_theme, themes_list = get_theme_context(request)
    video_qs = Video.objects.filter(theme_label=active_theme).order_by('-create_time')
    return render(request, 'data/video_list.html',
                  {'page_obj': Paginator(video_qs, 10).get_page(request.GET.get('page')), 'active_theme': active_theme,
                   'themes_list': themes_list})


# 评论详情分页列表
@login_required
def comment_list(request):
    active_theme, themes_list = get_theme_context(request)
    comment_qs = Comment.objects.filter(theme_label=active_theme).order_by('-create_time')
    return render(request, 'data/comment_list.html',
                  {'page_obj': Paginator(comment_qs, 15).get_page(request.GET.get('page')),
                   'active_theme': active_theme, 'themes_list': themes_list})


# === 词云停用词常驻内存，提升分词性能 ===
STOP_WORDS = {'的', '了', '和', '是', '就', '都', '而', '及', '与', '着', '或', '一个', '没有', '我们', '你们', '他们', '她', '他', '它', '在', '也', '有', '不', '啊', '吧', '呢', '呀', '哦', '哈', '吗', '这', '那', '我', '你', '很', '太', '还', '没', '这', '去', '说', '看', '要', '把'}

# 用户分布画像分析 (Audience insight 2.0)
@login_required
def chart_user(request):
    active_theme, themes_list = get_theme_context(request)
    
    # 基础查询集
    base_comments = Comment.objects.filter(theme_label=active_theme)
    base_videos = Video.objects.filter(theme_label=active_theme)


    # === 3. NLP Word Cloud ===
    all_clean_text = base_comments.exclude(content_clean='').values_list('content_clean', flat=True)
    word_counter = Counter()
    for text in all_clean_text:
        words = jieba.lcut(text)
        valid_words = [w for w in words if len(w) > 1 and w not in STOP_WORDS]
        word_counter.update(valid_words)
        
    top_50_words = word_counter.most_common(50)
    wordcloud_data = [{"name": w, "value": cnt} for w, cnt in top_50_words]

    # === 4. Regional Sentiment ===
    ip_stats = base_comments.exclude(ip_label='未知').values('ip_label').annotate(
        count=Count('comment_id'),
        avg_sentiment=Avg('sentiment_score')
    ).order_by('-count')[:15]
    
    bubble_data = []
    for row in ip_stats:
        bubble_data.append({
            "ip": row['ip_label'],
            "count": row['count'],
            "sentiment": round(row['avg_sentiment'], 3) if row['avg_sentiment'] else 0.5
        })

    return render(request, 'charts/user_charts.html', {
        'wordcloud_json': json.dumps(wordcloud_data),
        'bubble_data_json': json.dumps(bubble_data),
        'active_theme': active_theme,
        'themes_list': themes_list
    })


# 4维深度视觉分析页面
@login_required
def chart_content(request):
    active_theme, themes_list = get_theme_context(request)
    
    # 过滤条件：限定当前主题且已完成分析的视频，避免被零值稀释
    valid_videos = Video.objects.filter(
        theme_label=active_theme, 
        analysis_status=2
    ).exclude(visual_brightness__isnull=True)

    # [Guard] 空数据状态防御：若无已分析视频，跳过全部聚合逻辑
    if valid_videos.count() == 0:
        return render(request, 'charts/content_charts.html', {
            'active_theme': active_theme,
            'themes_list': themes_list,
            'no_data': True,
            'visual_dist_json': json.dumps({'brightness': {'low': 0, 'mid': 0, 'high': 0}, 'saturation': {'low': 0, 'mid': 0, 'high': 0}}),
            'audio_dist_json': json.dumps({'bpm': {'slow': 0, 'norm': 0, 'fast': 0}, 'cut_freq': {'low': 0, 'mid': 0, 'high': 0}}),
            'engagement_map_json': json.dumps([]),
            'top_dna_json': json.dumps([]),
        })

    visual_dist = {'brightness': {'low': 0, 'mid': 0, 'high': 0}, 'saturation': {'low': 0, 'mid': 0, 'high': 0}}
    audio_dist = {'bpm': {'slow': 0, 'norm': 0, 'fast': 0}, 'cut_freq': {'low': 0, 'mid': 0, 'high': 0}}
    
    engagement_map = []
    top_dna_data = []

    for v in valid_videos:
        b = float(v.visual_brightness or 0)
        s = float(v.visual_saturation or 0)
        bpm = int(v.audio_bpm or 0)
        cut = float(v.cut_frequency or 0)
        digg = int(v.digg_count or 0)

        # 视觉分布 (Brightness)
        if b <= 85: visual_dist['brightness']['low'] += 1
        elif b <= 170: visual_dist['brightness']['mid'] += 1
        else: visual_dist['brightness']['high'] += 1
        
        # 视觉分布 (Saturation)
        if s <= 85: visual_dist['saturation']['low'] += 1
        elif s <= 170: visual_dist['saturation']['mid'] += 1
        else: visual_dist['saturation']['high'] += 1

        # 音频/节奏分布 (BPM)
        if bpm <= 100: audio_dist['bpm']['slow'] += 1
        elif bpm <= 140: audio_dist['bpm']['norm'] += 1
        else: audio_dist['bpm']['fast'] += 1
        
        # 音频/节奏分布 (Cut Frequency)
        if cut <= 0.3: audio_dist['cut_freq']['low'] += 1
        elif cut <= 0.7: audio_dist['cut_freq']['mid'] += 1
        else: audio_dist['cut_freq']['high'] += 1

        # 点赞散点图数据 [Brightness, Likes]
        engagement_map.append([b, digg])
        
    # 获取 Top 10 DNA Profile
    top_videos = valid_videos.order_by('-digg_count')[:10]
    for v in top_videos:
        v_id = str(v.video_id)
        name_label = v.nickname + " - " + (v_id[-4:] if len(v_id) >= 4 else v_id)
        top_dna_data.append({
            'name': name_label,
            'value': [
                float(v.visual_brightness or 0), 
                float(v.visual_saturation or 0), 
                int(v.audio_bpm or 0), 
                float(v.cut_frequency or 0) * 100, 
                min(int(v.digg_count or 0) / 1000.0, 100.0)
            ]
        })

    return render(request, 'charts/content_charts.html', {
        'active_theme': active_theme,
        'themes_list': themes_list,
        'no_data': False,
        'visual_dist_json': json.dumps(visual_dist),
        'audio_dist_json': json.dumps(audio_dist),
        'engagement_map_json': json.dumps(engagement_map),
        'top_dna_json': json.dumps(top_dna_data)
    })


# === 极简情绪温度计分析视图：包含严密的 None 值容错 ===
@login_required
def chart_sentiment(request):
    active_theme, themes_list = get_theme_context(request)
    comments_qs = Comment.objects.filter(theme_label=active_theme)

    # 饼图基础数据
    avg_sent_val = comments_qs.exclude(sentiment_score=0.5).aggregate(Avg('sentiment_score'))[
                       'sentiment_score__avg'] or 0.5
    pos_count = comments_qs.filter(sentiment_score__gt=0.6).count()
    neg_count = comments_qs.filter(sentiment_score__lt=0.4).count()
    neu_count = comments_qs.count() - pos_count - neg_count

    # 获取全量视频节点，防御性排除亮度或节奏为空的异常坏账记录
    video_nodes = Video.objects.filter(theme_label=active_theme).annotate(
        clean_avg=Avg('comments__sentiment_score', filter=~Q(comments__sentiment_score=0.5))
    ).exclude(clean_avg=None).exclude(visual_brightness__isnull=True).exclude(audio_bpm__isnull=True)

    # 1. 视觉维度分桶计算
    v_buckets = {'low': [], 'mid': [], 'high': []}
    for vn in video_nodes:
        # 增加防御性判断，防止 None 比较错误
        if vn.visual_brightness is not None:
            if vn.visual_brightness <= 85:
                v_buckets['low'].append(vn.clean_avg)
            elif vn.visual_brightness <= 170:
                v_buckets['mid'].append(vn.clean_avg)
            else:
                v_buckets['high'].append(vn.clean_avg)

    v_bar_data = [
        round(sum(v_buckets['low']) / len(v_buckets['low']), 3) if v_buckets['low'] else 0.5,
        round(sum(v_buckets['mid']) / len(v_buckets['mid']), 3) if v_buckets['mid'] else 0.5,
        round(sum(v_buckets['high']) / len(v_buckets['high']), 3) if v_buckets['high'] else 0.5,
    ]

    # 2. 节奏维度分桶计算
    a_buckets = {'slow': [], 'norm': [], 'fast': []}
    for vn in video_nodes:
        if vn.audio_bpm is not None:
            if vn.audio_bpm <= 100:
                a_buckets['slow'].append(vn.clean_avg)
            elif vn.audio_bpm <= 140:
                a_buckets['norm'].append(vn.clean_avg)
            else:
                a_buckets['fast'].append(vn.clean_avg)

    a_bar_data = [
        round(sum(a_buckets['slow']) / len(a_buckets['slow']), 3) if a_buckets['slow'] else 0.5,
        round(sum(a_buckets['norm']) / len(a_buckets['norm']), 3) if a_buckets['norm'] else 0.5,
        round(sum(a_buckets['fast']) / len(a_buckets['fast']), 3) if a_buckets['fast'] else 0.5,
    ]

    # 取样最近的有效评论展示
    sample_comments = comments_qs.exclude(sentiment_score=0.5).order_by('-create_time')[:10]

    return render(request, 'charts/sentiment_charts.html', {
        'avg_sentiment': round(avg_sent_val, 2),
        'sentiment_pie_json': json.dumps([pos_count, neu_count, neg_count]),
        'v_bar_json': json.dumps(v_bar_data),
        'a_bar_json': json.dumps(a_bar_data),
        'active_theme': active_theme,
        'themes_list': themes_list,
        'sample_comments': sample_comments
    })


# 爆款预测交互
@login_required
def predict_page(request):
    active_theme, raw_themes = get_theme_context(request)
    
    # [Fix] 强制上下文清洗：避免 lazy object 或非字符串对象导致的模板渲染失败
    themes_list = [str(t) for t in raw_themes if t]
    
    context = {
        'active_theme': str(active_theme),
        'themes_list': themes_list
    }
    return render(request, 'prediction/dashboard.html', context)


# 爆款预测 AI 服务辅助函数
def _build_model_request_payload(hw: dict, follower_count: int, publish_hour: int, theme_name: str) -> dict:
    """只构建模型输入请求体；不混入 display-layer 当前主题基准。"""
    return {
        'visual_brightness': hw.get('visual_brightness', 128),
        'visual_saturation': hw.get('visual_saturation', 100),
        'audio_bpm': hw.get('audio_bpm', 110),
        'cut_frequency': hw.get('cut_frequency', 0.5),
        'duration_sec': hw.get('duration_sec', 15),
        'follower_count': follower_count,
        'publish_hour': publish_hour,
        'avg_sentiment': 0.5,
        'collect_count': 0,
        'comment_count': 0,
        'share_count': 0,
        'script_keywords': hw.get('script_keywords', []),
        'theme_label': theme_name or 'Unknown'
    }


def _load_display_theme_baseline(theme_name: str):
    """
    获取当前主题的 display/explanation baseline。
    该统计仅服务 UI 解释层，不是模型输入预处理真值。
    """
    if not theme_name:
        return None, None

    theme_videos = list(Video.objects.filter(theme_label=theme_name).values('digg_count'))

    # Strategy anchor (docs/ml_data_strategy.md):
    # display baseline belongs to current-theme explanation only.
    # This request-time query path should later move to cached/precomputed UI stats.
    global_videos = list(Video.objects.all().values('digg_count'))
    global_display_baseline = calculate_display_theme_baseline(global_videos)
    display_theme_baseline = calculate_display_theme_baseline(
        theme_videos,
        global_stats=global_display_baseline
    )
    return display_theme_baseline, display_theme_baseline.get('warning')

# 爆款预测 AI 服务接口
@login_required
def predict_api(request):
    """
    *爆款预测 API 核心入口：*
    *1. 接收视频文件与账号基础数据。*
    *2. 触发 RTX 3060 硬件加速特征提取。*
    *3. 调用大模型生成运营诊断书。*
    *4. 返回预测结果 + 4维图表数据。*
    *支持用户主观上的多次循环预测测试。*
    """
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': '仅支持POST请求'})
    
    try:
        video_obj = request.FILES.get('video_file')
        follower_count = int(request.POST.get('follower_count', 10000))
        publish_hour = int(request.POST.get('publish_hour', 18))

        temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp_uploads')
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, f"{uuid.uuid4().hex}.mp4")
        
        with open(temp_path, 'wb+') as f:
            for chunk in video_obj.chunks():
                f.write(chunk)

        try:
            analyzer = VideoContentAnalyzer(temp_path)
            hw = analyzer.run_full_analysis(include_script_keywords=True)
            theme_name = request.POST.get('theme_name')
            model_request_payload = _build_model_request_payload(
                hw=hw,
                follower_count=follower_count,
                publish_hour=publish_hour,
                theme_name=theme_name
            )

            # 将模型预测与 display-layer 当前主题解释显式拆分，避免职责混淆。
            display_theme_baseline, warning_msg = _load_display_theme_baseline(theme_name)

            # 1. 实例化预测服务并执行预测
            # [Singleton] 获取进程级单例实例：若已初始化则直接返回内存中的对象，无欲盘 I/O 开销
            service = predict_service.DiggPredictionService()
            prediction_result = service.predict_digg_count(
                model_request_payload,
                display_theme_baseline=display_theme_baseline
            )

            # 解析预测结果：
            # `predicted_digg` / `predicted_likes` 在当前接口中表示 display-facing 展示值；
            # 模型原始预测保留在服务内部语义中，不在 view 层直接消费。
            display_predicted_likes = prediction_result.get('predicted_digg', 0)
            quality_score = prediction_result.get('quality_score', 0)
            percentile_rank = prediction_result.get('percentile_rank', '')

            # 调用大模型生成专家建议
            # 将质量分和排名信息注入给 LLM
            # [新增] 注入统计学引擎算出的推荐时间与分位数
            theme_p25 = display_theme_baseline.get('p25', 0) if display_theme_baseline else 0
            theme_p50 = display_theme_baseline.get('p50', 0) if display_theme_baseline else 0
            theme_p75 = display_theme_baseline.get('p75', 0) if display_theme_baseline else 0
            optimal_times = (
                display_theme_baseline.get('optimal_publishing_times', ["18:00", "19:00", "20:00"])
                if display_theme_baseline else ["18:00", "19:00", "20:00"]
            )

            advice_context = {
                **model_request_payload,
                'predicted_likes': display_predicted_likes,
                'quality_score': quality_score,
                'percentile_rank': percentile_rank,
                'theme_name': theme_name or "通用赛道",
                'theme_p25': theme_p25,
                'theme_p50': theme_p50,
                'theme_p75': theme_p75,
                'optimal_publishing_times': optimal_times
            }
            
            # [新增] 注入用户自定义API配置
            config = CreatorConfig.objects.filter(user=request.user).first()
            user_key = config.llm_api_key if config else None
            model_name = config.llm_model_name if config else None
            
            try:
                import concurrent.futures
                # 引入更长的超时机制，允许 DeepSeek API 进行深度思考
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(LLMService().generate_advice, advice_context, user_key=user_key, model_name=model_name)
                    # 配置 20 秒超时以允许长文本生成能顺利执行
                    advice = future.result(timeout=50.0)
            except Exception as e:
                # 捕获超时或异常，返回真实的 LLM 报错信息
                logger.warning(f"[Circuit Breaker] LLM 调用失败或超时: {e}")
                advice = f"AI 分析生成异常: {str(e)}"
        except Exception as e:
            logger.error(f"处理视频过程中发生异常: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return JsonResponse({"status": "fallback", "predicted_likes": 0, "quality_score": 0, "message": "Corrupted file handled"}, status=200)

        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)

        # 返回完整数据：这里的 predicted_likes 是最终展示值，不是原始模型输出。
        return JsonResponse({
            'predicted_likes': f"{int(display_predicted_likes):,}",
            'quality_score': quality_score,
            'percentile_rank': percentile_rank,
            'baseline_info': display_theme_baseline,
            'warning': warning_msg,
            'advice': advice,
            'status': 'success'
        })
        
    except Exception as e:
        logger.exception(f"[ERROR] Prediction API failed: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)})


@login_required
def user_logout(request):
    logout(request)
    return redirect('login')


# 全局配置驱动：抽取核心控制变量至全局字典，拒绝硬编码，提升可维护性
GLOBAL_CONFIG = {
    'DEFAULT_CRAWL_QUOTA': 50,
    'SENTIMENT_FILTER_THRESHOLD': 0.6,
    'MAX_AUDIT_LOGS': 6
}

@login_required
def profile_view(request):
    try:
        # 处理用户配置提交
        if request.method == 'POST':
            api_key = request.POST.get('api_key', '').strip()
            model_name = request.POST.get('model_name', 'ernie-4.0-8k').strip()
            
            # API 密钥正则验证，拦截非规范格式
            if api_key and not re.match(r'^sk-[a-zA-Z0-9]{32,64}$', api_key):
                messages.error(request, "Invalid API Key format. Please use the standard 'sk-...' format.")
                return redirect('profile')
            
            # 获取或创建用户配置
            config, _ = CreatorConfig.objects.get_or_create(user=request.user)
            if api_key:
                config.llm_api_key = api_key
            config.llm_model_name = model_name
            config.save()
            messages.success(request, "配置已成功更新")
            return redirect('profile')

        # 获取当前活跃主题
        active_theme, themes_list = get_theme_context(request)
        
        # [个人资产模块] 只展示当前活跃主题下的数据
        theme_videos_count = Video.objects.filter(theme_label=active_theme).count()
        theme_comments_count = Comment.objects.filter(theme_label=active_theme).count()
        theme_ai_processed = Video.objects.filter(theme_label=active_theme, analysis_status=2).count()
        
        # 获取真实API配置并脱敏显示
        config = CreatorConfig.objects.filter(user=request.user).first()
        api_token = '未配置大模型密钥'
        model_name = 'ernie-4.0-8k'
        if config and config.llm_api_key:
            key_len = len(config.llm_api_key)
            if key_len > 8:
                api_token = f"{config.llm_api_key[:4]}...{config.llm_api_key[-4:]}"
            else:
                api_token = "已配置"
            model_name = config.llm_model_name

        # [解决IDE格式化导致的TemplateSyntaxError: 移到后端处理选中状态]
        model_selected = {
            'ernie_4_0': 'selected' if model_name == 'ernie-4.0-8k' else '',
            'ernie_3_5': 'selected' if model_name == 'ernie-3.5-8k' else '',
            'ernie_lite': 'selected' if model_name == 'ernie-lite-8k' else ''
        }

        # [模拟SaaS配额与API集成]
        account_info = {
            'tier': '一个很厉害的分析师',
            'ai_power': '开始你的表演',
            'monthly_quota': 1000,
            'used_quota': theme_ai_processed,
            'api_token': api_token,
            'model_name': model_name,
            'model_selected': model_selected
        }
        
        # [配置管理模块] 核心控制变量
        user_crawl_quota = request.session.get('crawl_quota', GLOBAL_CONFIG['DEFAULT_CRAWL_QUOTA'])
        user_sentiment_threshold = request.session.get('sentiment_threshold', GLOBAL_CONFIG['SENTIMENT_FILTER_THRESHOLD'])
        default_baseline = request.session.get('default_baseline', 1000)

        # [最近动态 (Activity History)]
        footprints = Video.objects.filter(theme_label=active_theme).order_by('-create_time')[:5]

        # 彻底删除了 Python 版本、Django 引擎状态、数据库物理大小、状态机追踪 ID 等管理员专属信息
        context = {
            'user': request.user,
            'active_theme': active_theme,
            'themes_list': themes_list,
            'assets': {
                'total_videos': theme_videos_count,
                'total_comments': theme_comments_count,
            },
            'account_info': account_info,
            'prefs': {
                'quota': user_crawl_quota,
                'threshold': user_sentiment_threshold,
                'default_baseline': default_baseline
            },
            'footprints': footprints,
        }
        
    except Exception as e:
        context = {
            'user': request.user,
            'active_theme': "System Default",
            'themes_list': [],
            'error_msg': str(e),
            'assets': {'total_videos': 0, 'total_comments': 0},
            'account_info': {'tier': 'Error', 'ai_power': 'Error', 'monthly_quota': 0, 'used_quota': 0, 'api_token': 'Error', 'model_name': 'Error'},
            'prefs': {'quota': GLOBAL_CONFIG['DEFAULT_CRAWL_QUOTA'], 'threshold': GLOBAL_CONFIG['SENTIMENT_FILTER_THRESHOLD'], 'default_baseline': 1000},
            'footprints': [],
        }
        
    return render(request, 'users/profile.html', context)



def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            login(request, form.save())
            return redirect('dashboard')
    else:
        form = UserCreationForm()
    return render(request, 'registration/register.html', {'form': form})


# ================================================================
# 增量情感分析触发器 (Incremental Sentiment Trigger)
# ================================================================

def trigger_sentiment_analysis(theme_name):
    """
    *后台异步执行情感分析的钩子函数*
    *1. 仅过滤当前主题下 sentiment_score 在 [0.499, 0.501] 的未处理评论*
    *2. Batch Size = 200 分批更新，防止 OOM*
    *3. 实时更新 Cache 中的 analysis_progress*
    """
    try:
        # 重置初始进度，与前端状态同步
        cache.set('analysis_progress', 0, timeout=3600)
        
        # 1. Incremental Filter (Bug 5): 查出符合条件的数据
        qs = Comment.objects.filter(theme_label=theme_name, sentiment_score__gte=0.499, sentiment_score__lte=0.501)
        total = qs.count()
        if total == 0:
            cache.set('analysis_progress', 100, timeout=3600)
            return

        batch_size = 200
        updated_batch = []
        processed = 0

        # 使用 iterator 防止 OOM
        for c in qs.iterator(chunk_size=batch_size):
            try:
                # 提取清洗文本
                cleaned_raw = clean_text_nuclear(c.content)
                target_text = extract_semantic_features(cleaned_raw)
                
                if not target_text.strip() or len(target_text.strip()) < 1:
                    continue
                
                # 调用底层打分函数分析
                score, label = calculate_refined_sentiment(target_text)
                
                c.sentiment_score = float(score)
                c.sentiment_label = str(label)
                c.content_clean = target_text
                
                updated_batch.append(c)
                processed += 1
                
                # Batching: 分批入库
                if len(updated_batch) >= batch_size:
                    Comment.objects.bulk_update(updated_batch, ['sentiment_score', 'sentiment_label', 'content_clean'])
                    updated_batch = []
                    
                    # Status Sync: 实时同步进度 (Audit 3)
                    progress = int((processed / total) * 100)
                    cache.set('analysis_progress', progress, timeout=3600)
                    
            except Exception as e:
                # Robustness (Audit 4): 报错记录跳过
                logger.error(f"[Sentiment Trigger] *单条评论 {c.comment_id} 处理失败: {e}*")
                continue

        # 处理剩余数据
        if updated_batch:
            Comment.objects.bulk_update(updated_batch, ['sentiment_score', 'sentiment_label', 'content_clean'])
            
        # 完成兜底
        cache.set('analysis_progress', 100, timeout=3600)
        logger.info(f"[Sentiment Trigger] *主题 {theme_name} 的增量分析完成，处理数：{processed}*")

    except Exception as e:
        logger.error(f"[Sentiment Trigger] *执行整体异常: {e}*")


# ================================================================
# 爬虫服务接口 (Spider Service API)
# ================================================================

def _run_spider_background(keywords_list, max_videos, max_comments, theme_name, is_global_limit=False):
    """
    *后台线程执行爬虫任务的内部函数*
    *该函数会在独立线程中运行，避免阻塞 Django 主进程*
    *完整流程：抓取 -> 清洗 -> 入库*
    *支持多关键词批量执行：依次处理关键词队列，实时更新进度状态*
    """
    logger.info(f"[后台线程] 线程已启动! keywords={keywords_list}, max_videos={max_videos}, theme={theme_name}")
    
    try:
        import sys
        data_path = os.path.join(settings.BASE_DIR, 'data')
        sys.path.insert(0, data_path)
        
        from crawler.spyder_unified import run_spider_service, DouyinUnifiedPipeline
        
        # 定义进度回调函数
        def progress_callback(current, total, start_time, message=""):
            try:
                global_status = cache.get('global_pipeline_status', {"status": "idle", "global_phase": "idle", "video": {"c": 0, "t": 0}, "comment": {"c": 0, "t": 0}, "ai": {"c": 0, "t": 0}, "msg": ""})
                global_status["status"] = "running"
                if "视频" in message and "评论" not in message and "详情" not in message:
                    global_status["global_phase"] = "video_crawling"
                    global_status["video"] = {"c": current, "t": total}
                elif "评论" in message or "详情" in message:
                    global_status["global_phase"] = "comment_crawling"
                    global_status["comment"] = {"c": current, "t": total}
                global_status["msg"] = message
                cache.set('global_pipeline_status', global_status, timeout=3600)

                # 防止除零错误
                if total <= 0: total = 1
                
                # ================================================================
                # [Progress Fix] 三阶段平滑算法 (3-Phase Smooth Progression)
                # 解决阶段切换时 current/total 导致进度条回退闪烁的问题
                # ================================================================
                if "采集" in message or "准备" in message:
                    base_pct, max_pct = 0, 15
                elif "下载" in message or "缓存" in message:
                    base_pct, max_pct = 15, 50
                elif "评论" in message or "详情" in message or "跳过" in message:
                    base_pct, max_pct = 50, 95
                else:
                    base_pct, max_pct = 0, 95

                stage_pct = (current / total) * (max_pct - base_pct)
                percent = int(base_pct + stage_pct)
                percent = min(percent, 95)
                
                # 绝对单调递增约束 (Monotonic Constraint)
                old_state = cache.get('spider_progress', {})
                old_percent = old_state.get('percent', 0)
                if old_percent > 0 and percent < old_percent:
                    percent = old_percent
                
                # ETA 算法: (总数 - 当前)  平均耗时
                elapsed = time.time() - start_time
                eta_str = "计算中..."
                avg_duration_str = "0.0s"
                throughput_str = "0.0 vpm"
                
                if current > 0:
                    avg_time = elapsed / current
                    avg_duration_str = f"{avg_time:.1f}s"
                    
                    if elapsed > 0:
                        vpm = (current / elapsed) * 60
                        throughput_str = f"{vpm:.1f} vpm"
                        
                    remaining_sec = int(avg_time * (total - current))
                    if remaining_sec < 60:
                        eta_str = f"{remaining_sec}秒"
                    else:
                        eta_str = f"{remaining_sec // 60}分{remaining_sec % 60}秒"
                
                status_msg = message if message else f"正在挖掘第 {current}/{total} 个视频的评论..."
                
                # 写入缓存 (5分钟过期)
                cache.set('spider_progress', {
                    'percent': percent,
                    'status': status_msg,
                    'eta': eta_str,
                    'current': current,
                    'total': total,
                    'avg_duration': avg_duration_str,
                    'throughput': throughput_str,
                    'current_file': message
                }, timeout=300)
                
            except Exception as e:
                logger.warning(f"[爬虫] 进度回调异常（已忽略）: {e}")


        # === 阶段1：执行爬虫抓取 (支持批量关键词队列) ===
        total_keywords = len(keywords_list)
        
        # === 算法升级：除法取模公平分配 (DivMod / Card Dealing) ===
        # 解决 max_videos < total_keywords 时的负数分配问题
        if is_global_limit:
            # 全局模式：先计算基数和余数
            base_limit = max_videos // total_keywords
            remainder = max_videos % total_keywords
        else:
            # 独立模式：全额分配
            base_limit = max_videos
            remainder = 0
            
        logger.info(f"[爬虫] 批量任务启动: 共 {total_keywords} 个关键词, 主题={theme_name}, 总配额={max_videos}, 模式={'全局限制' if is_global_limit else '独立限制'}")
        
        # [Batch Execution] 遍历关键词队列，依次执行
        accumulated_video_path = None
        accumulated_comment_path = None
        
        for kw_idx, keyword in enumerate(keywords_list):
            # 计算当前关键词的动态配额
            current_limit = 0
            if is_global_limit:
                # 前 remainder 个关键词多分配 1 个 (Card Dealing)
                current_limit = base_limit + (1 if kw_idx < remainder else 0)
                
                # EDGE CASE GUARD: 配额耗尽时跳过 (针对配额极小的情况)
                if current_limit <= 0:
                    logger.info(f"[爬虫] ⏭ 配额耗尽 (0)，跳过关键词: {keyword}")
                    continue
            else:
                current_limit = max_videos
            
            # === 实时阶段报告 (Real-time Stage Reporting) ===
            # 更新 DouyinUnifiedPipeline.instance.progress_stats 以供前端 Navbar Capsule 显示
            stage_label = f"Keyword {kw_idx + 1}/{total_keywords}: [{keyword}]"
            
            try:
                if DouyinUnifiedPipeline.instance:
                    DouyinUnifiedPipeline.instance.progress_stats['stage'] = stage_label
                    DouyinUnifiedPipeline.instance.progress_stats['batch_current'] = kw_idx + 1
                    DouyinUnifiedPipeline.instance.progress_stats['batch_total'] = total_keywords
                    DouyinUnifiedPipeline.instance.progress_stats['current_keyword'] = keyword
            except Exception as stage_err:
                logger.warning(f"[爬虫] *阶段报告更新失败（非致命）: {stage_err}*")
            
            logger.info(f"[爬虫] ▶ 开始处理 {stage_label}, 配额={current_limit}")
            
            # 更新缓存进度
            cache.set('spider_progress', {
                'percent': int((kw_idx / total_keywords) * 90),  # 预留10%给入库
                'status': f"正在处理 {stage_label}...",
                'eta': '计算中...',
                'current': kw_idx,
                'total': total_keywords,
                'stage': stage_label
            }, timeout=300)
            
            # 执行单个关键词的爬虫任务
            video_path, comment_path = run_spider_service(
                keyword, current_limit, max_comments, theme_name, progress_callback
            )
            
            # 保存路径 (用于后续入库)
            if video_path:
                accumulated_video_path = video_path
            if comment_path:
                accumulated_comment_path = comment_path
            
            logger.info(f"[爬虫]  关键词 [{keyword}] 完成: 视频={video_path}, 评论={comment_path}")
        
        # 更新进度为 95% (所有关键词采集完成，准备入库)
        cache.set('spider_progress', {
            'percent': 95, 
            'status': f"全部 {total_keywords} 个关键词抓取完成，正在执行资产入库...", 
            'eta': '0秒',
            'current': total_keywords, 
            'total': total_keywords,
            'stage': 'importing'
        }, timeout=300)
        
        # 使用最后一次采集的路径进行入库
        video_path = accumulated_video_path
        comment_path = accumulated_comment_path
        
        logger.info(f"[爬虫] 批量采集结束: 共处理 {total_keywords} 个关键词")
        
        # === 阶段2：自动入库 ===
        if video_path and comment_path:
            logger.info(f"[入库] 开始自动入库: 主题={theme_name}")
            
            # 调用入库服务函数
            import_result = import_data_service(theme_name, video_path, comment_path)
            
            if import_result['success']:
                logger.info(f"[入库] 自动入库结束: {import_result['message']}")
                
                #  自动化工作流：触发 AI 情感引擎 (Trigger Hook) 
                threading.Thread(target=trigger_sentiment_analysis, args=(theme_name,), daemon=True).start()
                logger.info(f"[触发器] 已启动异步情感分析流水线: {theme_name}")
                # 
            else:
                logger.warning(f"[入库] 自动入库失败: {import_result['message']}")
        else:
            logger.warning(f"[入库] 跳过自动入库: 爬虫未返回有效文件路径")
        
        logger.info(f"[流程] 完整流程结束: 抓取->清洗->入库 主题={theme_name}")
        
        # 任务彻底完成，设置为 100%
        cache.set('spider_progress', {
            'percent': 100, 
            'status': f"所有 {total_keywords} 个关键词的任务已完成！请刷新页面查看。", 
            'eta': '0秒',
            'current': total_keywords, 
            'total': total_keywords,
            'stage': 'completed'
        }, timeout=300)
        
    except Exception as e:
        logger.exception(f"[爬虫] 后台任务异常: {e}")
        
        # 任务失败也更新状态
        cache.set('spider_progress', {
            'percent': 0, 
            'status': f"任务异常中止: {str(e)}", 
            'eta': '--',
            'current': 0, 
            'total': 0
        }, timeout=300)

def _run_comment_only_background(video_csv_filename, max_comments, theme_name):
    """
    *后台线程执行仅评论采集任务的内部函数*
    *该函数会在独立线程中运行,避免阻塞 Django 主进程*
    *完整流程：读取视频CSV -> 抓取评论 -> 入库*
    """
    try:
        import sys
        sys.path.insert(0, os.path.join(settings.BASE_DIR, 'data'))
        from crawler.spyder_unified import run_comment_only_service
        
        # 构建视频CSV的绝对路径
        PROJECT_ROOT = settings.BASE_DIR
        video_csv_path = os.path.join(PROJECT_ROOT, 'data', video_csv_filename)
        
        # 定义进度回调函数
        def progress_callback(current, total, start_time, message=""):
            try:
                global_status = cache.get('global_pipeline_status', {"status": "idle", "global_phase": "idle", "video": {"c": 0, "t": 0}, "comment": {"c": 0, "t": 0}, "ai": {"c": 0, "t": 0}, "msg": ""})
                global_status["status"] = "running"
                global_status["global_phase"] = "comment_crawling"
                global_status["comment"] = {"c": current, "t": total}
                global_status["msg"] = message
                cache.set('global_pipeline_status', global_status, timeout=3600)

                if total <= 0: total = 1
                percent = int((current / total) * 100)
                if percent > 95: percent = 95
                
                elapsed = time.time() - start_time
                eta_str = "计算中..."
                avg_duration_str = "0.0s"
                throughput_str = "0.0 vpm"
                
                if current > 0:
                    avg_time = elapsed / current
                    avg_duration_str = f"{avg_time:.1f}s"
                    
                    if elapsed > 0:
                        vpm = (current / elapsed) * 60
                        throughput_str = f"{vpm:.1f} vpm"
                        
                    remaining_sec = int(avg_time * (total - current))
                    if remaining_sec < 60:
                        eta_str = f"{remaining_sec}秒"
                    else:
                        eta_str = f"{remaining_sec // 60}分{remaining_sec % 60}秒"
                
                status_msg = message if message else f"正在挖掘第 {current}/{total} 个视频的评论..."
                
                cache.set('spider_progress', {
                    'percent': percent,
                    'status': status_msg,
                    'eta': eta_str,
                    'current': current,
                    'total': total,
                    'avg_duration': avg_duration_str,
                    'throughput': throughput_str,
                    'current_file': message
                }, timeout=300)
                
            except Exception as e:
                logger.warning(f"[仅评论] 进度回调异常（已忽略）: {e}")

        # === 阶段1：执行仅评论抓取 ===
        logger.info(f"[仅评论] 后台任务启动: 视频CSV={video_csv_filename}, 主题={theme_name}")
        
        comment_path = run_comment_only_service(video_csv_path, theme_name, max_comments, progress_callback)
        
        cache.set('spider_progress', {
            'percent': 98, 
            'status': "评论抓取完成，正在执行资产入库...", 
            'eta': '0秒',
            'current': 1, 
            'total': 1
        }, timeout=300)
        
        logger.info(f"[仅评论] 评论抓取结束: 评论={comment_path}")
        
        # === 阶段2：自动入库 ===
        if comment_path:
            logger.info(f"[入库] 开始自动入库: 主题={theme_name}")
            import_result = import_data_service(theme_name, video_csv_path, comment_path)
            
            if import_result['success']:
                logger.info(f"[入库] 自动入库结束: {import_result['message']}")
            else:
                logger.warning(f"[入库] 自动入库失败: {import_result['message']}")
        
        logger.info(f"[流程] 完整流程结束: 仅评论采集->入库 主题={theme_name}")
        
        cache.set('spider_progress', {
            'percent': 100, 
            'status': "评论采集任务已完成！请刷新页面查看。", 
            'eta': '0秒',
            'current': 1, 
            'total': 1
        }, timeout=300)
        
    except Exception as e:
        logger.exception(f"[仅评论] 后台任务异常: {e}")
        
        cache.set('spider_progress', {
            'percent': 0, 
            'status': f"任务异常中止: {str(e)}", 
            'eta': '--',
            'current': 0, 
            'total': 0
        }, timeout=300)

@login_required
def get_spider_status_api(request):
    """
    *获取爬虫任务实时进度的 API*
    *前端长轮询此接口以更新进度条*
    """
    global_status = cache.get('global_pipeline_status', {
        "status": "idle", 
        "global_phase": "idle", 
        "video": {"c": 0, "t": 0}, 
        "comment": {"c": 0, "t": 0}, 
        "ai": {"c": 0, "t": 0}, 
        "msg": "等待任务启动..."
    })
    return JsonResponse(global_status)


# === 视频分析状态监控 API ===
@login_required
def get_global_status(request):
    """
    *获取系统全局状态 (The 'Brain' Status)*
    *包含：爬虫运行状态、爬虫进度、AI 待处理队列、当前主题*
    """
    try:
        status_data = {
            "spider_running": False,
            "spider_progress": {"current": 0, "total": 0},
            "ai_queue_count": 0,
            "current_theme": request.session.get('active_theme', 'Default')
        }

        # 1. 获取爬虫状态 (通过直接访问实例)
        # 动态导入以避免循环依赖或路径问题
        spider_instance = None
        try:
            import sys
            if os.path.join(settings.BASE_DIR, 'data') not in sys.path:
                sys.path.insert(0, os.path.join(settings.BASE_DIR, 'data'))
            from crawler.spyder_unified import DouyinUnifiedPipeline
            spider_instance = DouyinUnifiedPipeline.instance
        except ImportError:
            logger.warning("[GlobalStatus] *无法导入 DouyinUnifiedPipeline*")
        except Exception as import_err:
            logger.warning(f"[GlobalStatus] *获取爬虫实例失败: {import_err}*")

        if spider_instance:
            status_data["spider_running"] = True
            # 直接读取实例中的实时进度
            if hasattr(spider_instance, 'progress_stats'):
                status_data["spider_progress"] = spider_instance.progress_stats
            else:
                # Fallback if attribute missing
                status_data["spider_progress"] = {"current": 0, "total": 0, "stage": "unknown"}

        # 2. 获取 AI 队列长度 (高效查询)
        # 使用 only('video_id') 减少数据库 I/O 开销
        try:
            queue_count = Video.objects.filter(
                analysis_status=0,
                local_temp_path__isnull=False
            ).exclude(local_temp_path='').only('video_id').count()
            status_data["ai_queue_count"] = queue_count
        except Exception as db_err:
            logger.error(f"[GlobalStatus] *查询 AI 队列失败: {db_err}*")
            status_data["ai_queue_count"] = -1

        return JsonResponse(status_data)

    except Exception as e:
        logger.error(f"[GlobalStatus] *全局状态查询异常: {e}*")
        # Fail-safe return
        return JsonResponse({
            "spider_running": False,
            "spider_progress": {"current": 0, "total": 0},
            "ai_queue_count": 0,
            "current_theme": "Error",
            "error": str(e)
        })

@login_required
def get_analysis_status_api(request):
    """
    *获取后台 AI 分析任务的统计状态*
    *[重构] unified JSON including is_active, current_theme, scraping_progress, ai_analysis_progress*
    """
    try:
        import time
        from django.db import transaction
        
        # 获取全局 ACTIVE_TASK 和爬虫进度
        active_task = cache.get('ACTIVE_TASK', {})
        spider_progress = cache.get('spider_progress', {})
        
        # ================================================================
        # [Priority Redirection] 优先读取并强制绑定当前活跃主题 (防穿越)
        # ================================================================
        is_active = active_task.get('is_active', False)
        active_theme_name = active_task.get('current_theme')
        
        if is_active and active_theme_name:
            target_theme = active_theme_name
        else:
            target_theme = request.GET.get('theme') or request.session.get('active_theme', '默认主题')
            
        # 按目标主题过滤，防止全局统计混视
        theme_qs = Video.objects.filter(theme_label=target_theme)
        
        ready_pending_qs = theme_qs.filter(
            analysis_status=0,
            local_temp_path__isnull=False
        ).exclude(local_temp_path='')
        blocked_pending_qs = theme_qs.filter(analysis_status=0).filter(
            Q(local_temp_path__isnull=True) | Q(local_temp_path='')
        )

        pending_count = ready_pending_qs.count()
        blocked_count = blocked_pending_qs.count()
        processing_count = theme_qs.filter(analysis_status=1).count()
        completed_count = theme_qs.filter(analysis_status=2).count()
        failed_count = theme_qs.filter(analysis_status=-1).count()
        
        total_ai = pending_count + processing_count + completed_count + failed_count
        
        worker_alive = False
        if _ai_worker_instance is not None:
            worker_alive = _ai_worker_instance.is_alive()
        
        # ================================================================
        # [Health Check] Zombie Task Recovery (绑定主题)
        # ================================================================
        current_time = time.time()
        if pending_count == 0 and processing_count > 0:
            has_zombie = False
            if not worker_alive:
                has_zombie = True
            elif _ai_worker_instance and hasattr(_ai_worker_instance, '_processing_timestamps'):
                for vid, start_t in list(_ai_worker_instance._processing_timestamps.items()):
                    if current_time - start_t > 60:
                        has_zombie = True
                        break
                if len(_ai_worker_instance._processing_timestamps) == 0:
                    has_zombie = True
                    
            if has_zombie:
                with transaction.atomic():
                    theme_qs.filter(analysis_status=1).update(analysis_status=-1)
                processing_count = 0
                failed_count = theme_qs.filter(analysis_status=-1).count()
        
        if pending_count > 0 and not worker_alive:
            try:
                start_ai_worker()
            except Exception:
                pass
        
        # 获取爬虫数据
        scraping_c = spider_progress.get('current', 0)
        scraping_t = spider_progress.get('total', 0)
        scraping_percent = spider_progress.get('percent', 0)
        scraping_msg = spider_progress.get('status', '空闲')
        
        # 获取吞吐量和当前文件数据
        avg_dur = spider_progress.get('avg_duration', '0.0s')
        tp = spider_progress.get('throughput', '0.0 vpm')
        c_file = spider_progress.get('current_file', '等待中...')
        
        # [分母校准] 保证刚启动的新任务 AI 进度条总工作量预映射为爬虫设定配额
        # 防止在入库未完成时分母为0导致页面进度条跳错
        if is_active and active_theme_name == target_theme and scraping_percent < 100 and scraping_t > total_ai:
            total_ai = scraping_t
            
        ai_msg = "等待中..."
        if total_ai > 0 and pending_count > 0:
            ai_msg = f"队列中有 {pending_count} 个视频等待分析"
        if processing_count > 0:
            ai_msg = "AI 正在分析视频内容..."
        if pending_count == 0 and processing_count == 0 and blocked_count > 0:
            if is_active and scraping_percent < 100:
                ai_msg = f"{blocked_count} 个视频等待下载或入库完成"
            else:
                ai_msg = f"{blocked_count} 个视频缺少本地文件，未进入 AI 队列"
        if total_ai > 0 and pending_count == 0 and processing_count == 0 and blocked_count == 0:
            # [修复] 如果爬虫仍在运行(未到100%)，说明新视频还未入库，此时 AI 属于等待前置环节状态
            if is_active and scraping_percent < 100:
                ai_msg = "等待抓取完成并入库..."
            else:
                if failed_count > 0:
                    ai_msg = f"分析完成 (其中 {failed_count} 个视频缺失或损坏)"
                else:
                    ai_msg = "所有视频分析已完成"
            
        # 判定整体活跃度
        if (scraping_t > 0 and scraping_percent < 100) or (total_ai > 0 and (pending_count > 0 or processing_count > 0)):
            is_active = True
        elif is_active:
            is_active = False
            cache.set('ACTIVE_TASK', {'is_active': False, 'current_theme': target_theme}, timeout=86400)
            
        # 计算队列负载
        queue_load = "Empty"
        if pending_count > 50:
            queue_load = "High"
        elif pending_count > 0:
            queue_load = "Normal"
            
        return JsonResponse({
            'is_active': is_active,
            'current_theme': target_theme,
            'scraping_progress': {
                'current': scraping_c,
                'total': scraping_t,
                'percent': scraping_percent,
                'msg': scraping_msg
            },
            'ai_analysis_progress': {
                'current': completed_count + failed_count, # [修复]: 将失败数算入已处理进度，让滚动条达到 100%
                'total': total_ai,
                'msg': ai_msg,
                'pending': pending_count,
                'blocked': blocked_count,
                'processing': processing_count,
                'failed': failed_count
            },
            'avg_duration': avg_dur,
            'throughput': tp,
            'current_file': c_file,
            'queue_load': queue_load
        })
    except Exception as e:
        logger.error(f"[AnalysisStatus] *查询分析状态失败: {e}*")
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@login_required
def launch_spider_api(request):
    """
    *启动爬虫任务的 API 接口*
    *接收 POST 请求，在后台线程中启动爬虫，立即返回响应避免阻塞*
    *安全机制：单例模式防护，防止并发任务冲突*
    """
    if request.method != 'POST':
        return JsonResponse({'success': False, 'status': 'error', 'message': '仅支持 POST 请求'})
    
    try:
        # === 单例防护检查：阻止并发爬虫任务 ===
        # 通过检查 DouyinUnifiedPipeline.instance 判断是否有任务正在运行
        try:
            import sys
            data_path = os.path.join(settings.BASE_DIR, 'data')
            if data_path not in sys.path:
                sys.path.insert(0, data_path)
            from crawler.spyder_unified import DouyinUnifiedPipeline
            
            if DouyinUnifiedPipeline.instance is not None:
                logger.warning("[爬虫] *拒绝启动：检测到已有任务正在运行*")
                return JsonResponse({
                    'success': False,
                    'status': 'error',
                    'message': 'A task is already in progress.'
                })
        except ImportError:
            # 模块未加载时视为无任务运行，允许启动
            pass
        except Exception as check_err:
            logger.warning(f"[爬虫] *单例检查异常（非阻塞）: {check_err}*")
        
        keyword = request.POST.get('keyword', '').strip()
        theme_name = request.POST.get('theme_name', '').strip()
        
        # === Step 5.3: 关键词队列解析 (Keyword Queue Parsing) ===
        # 支持逗号分隔的多关键词输入
        keywords_list = [k.strip() for k in keyword.split(',') if k.strip()]
        
        if not keywords_list:
            return JsonResponse({'success': False, 'status': 'error', 'message': '关键词不能为空'})
        
        # 获取全局/独立限制模式标志 (Global vs Per-Keyword Limit)
        is_global_limit = request.POST.get('is_global_limit', 'false').lower() == 'true'
        
        # 参数协调：优先读取 max_videos，兼容旧版 count
        max_videos_param = request.POST.get('max_videos') or request.POST.get('count')
        max_videos = int(max_videos_param) if max_videos_param else 10
        
        max_comments = int(request.POST.get('max_comments', 50))  # 从前端获取每视频最大评论数

        # 详细日志记录
        logger.info(f"[爬虫] API收到请求: keywords={keywords_list}, theme='{theme_name}', max_videos={max_videos}, is_global_limit={is_global_limit}")

        # 参数校验
        if not theme_name:
            return JsonResponse({'success': False, 'status': 'error', 'message': '主题名称不能为空'})
        if max_videos <= 0 or max_videos > 5000:
            return JsonResponse({'success': False, 'status': 'error', 'message': '抓取数量需在 1-5000 之间'})
        
        # 启动后台线程执行爬虫（daemon=True 确保主进程退出时自动终止）
        cache.set('ACTIVE_TASK', {
            'is_active': True,
            'current_theme': theme_name
        }, timeout=86400)
        
        spider_thread = threading.Thread(
            target=_run_spider_background,
            args=(keywords_list, max_videos, max_comments, theme_name, is_global_limit),
            daemon=True  # 守护线程：不阻塞 Django 服务器退出
        )
        spider_thread.start()
        
        logger.info(f"[爬虫] 批量任务已提交后台: keywords={keywords_list}, theme={theme_name}, max_videos={max_videos}")
        
        # 关键步骤：启动 AI 分析后台工作线程
        # 确保爬虫数据入库后，消费者线程能够即时处理
        try:
            start_ai_worker()
            logger.info("[爬虫] *AI 分析工作线程已同步启动*")
        except Exception as worker_err:
            logger.warning(f"[爬虫] *AI 工作线程启动失败（非致命）: {worker_err}*")
        
        # 构建响应消息
        keywords_display = ', '.join(keywords_list[:3]) + ('...' if len(keywords_list) > 3 else '')
        
        return JsonResponse({
            'success': True,  # Phase 4 集成标志：前端可据此触发 Status Capsule
            'status': 'success',
            'message': f'批量爬虫任务已提交，共 {len(keywords_list)} 个关键词。关键词: [{keywords_display}], 主题: {theme_name}'
        })
        
    except ValueError as e:
        return JsonResponse({'success': False, 'status': 'error', 'message': f'参数格式错误: {e}'})
    except Exception as e:
        logger.error(f"[爬虫] API 异常: {e}")
        return JsonResponse({'success': False, 'status': 'error', 'message': f'服务器内部错误: {e}'})


@login_required
def launch_comment_only_api(request):
    """
    *启动仅评论采集任务的 API 接口*
    *接收 POST 请求,从已有视频CSV中读取视频ID并采集评论*
    *安全机制：单例模式防护，防止并发任务冲突*
    *参数：*
        *- video_csv_filename: 视频CSV文件名 (位于 data/ 目录下)*
        *- theme_name: 主题名称*
        *- max_comments: 每视频最大评论数*
    """
    if request.method != 'POST':
        return JsonResponse({'success': False, 'status': 'error', 'message': '仅支持 POST 请求'})
    
    try:
        # === 单例防护检查：阻止并发爬虫任务 ===
        try:
            import sys
            data_path = os.path.join(settings.BASE_DIR, 'data')
            if data_path not in sys.path:
                sys.path.insert(0, data_path)
            from spyder_unified import DouyinUnifiedPipeline
            
            if DouyinUnifiedPipeline.instance is not None:
                logger.warning("[仅评论] *拒绝启动：检测到已有任务正在运行*")
                return JsonResponse({
                    'success': False,
                    'status': 'error',
                    'message': 'A task is already in progress.'
                })
        except ImportError:
            pass
        except Exception as check_err:
            logger.warning(f"[仅评论] *单例检查异常（非阻塞）: {check_err}*")
        
        video_csv_filename = request.POST.get('video_csv_filename', '').strip()
        theme_name = request.POST.get('theme_name', '').strip()
        max_comments = int(request.POST.get('max_comments', 50))
        
        logger.info(f"[仅评论] API收到请求: video_csv={video_csv_filename}, theme={theme_name}, max_comments={max_comments}")
        
        # 参数校验
        if not video_csv_filename:
            return JsonResponse({'success': False, 'status': 'error', 'message': '视频CSV文件名不能为空'})
        if not theme_name:
            return JsonResponse({'success': False, 'status': 'error', 'message': '主题名称不能为空'})
        if max_comments <= 0 or max_comments > 500:
            return JsonResponse({'success': False, 'status': 'error', 'message': '评论数量需在 1-500 之间'})
        
        # 检查文件是否存在
        video_csv_path = os.path.join(settings.BASE_DIR, 'data', video_csv_filename)
        if not os.path.exists(video_csv_path):
            return JsonResponse({'success': False, 'status': 'error', 'message': f'视频CSV文件不存在: {video_csv_filename}'})
        
        # 启动后台线程执行仅评论采集（daemon=True 确保主进程退出时自动终止）
        comment_thread = threading.Thread(
            target=_run_comment_only_background,
            args=(video_csv_filename, max_comments, theme_name),
            daemon=True  # 守护线程：不阻塞 Django 服务器退出
        )
        comment_thread.start()
        
        logger.info(f"[仅评论] 任务已提交后台: video_csv={video_csv_filename}, theme={theme_name}")
        
        return JsonResponse({
            'success': True,  # Phase 4 集成标志：前端可据此触发 Status Capsule
            'status': 'success',
            'message': f'仅评论采集任务已提交，正在后台运行。主题: {theme_name}, 视频CSV: {video_csv_filename}'
        })
        
    except ValueError as e:
        return JsonResponse({'success': False, 'status': 'error', 'message': f'参数格式错误: {e}'})
    except Exception as e:
        logger.error(f"[仅评论] API 异常: {e}")
        return JsonResponse({'success': False, 'status': 'error', 'message': f'服务器内部错误: {e}'})


# ================================================================
# 模型重训练 API (Retrain Model API)
# ================================================================
@csrf_exempt
@login_required
def retrain_model_api(request):
    """
    *触发模型重训练的 API 接口*
    *在后台线程中执行 train_master_arena.py，完成后热替换 DiggPredictionService 单例*
    """
    if request.method != 'POST':
        return JsonResponse({'success': False, 'message': '仅支持 POST 请求'})

    active_theme = request.session.get('active_theme', '默认主题')
    total_count = Video.objects.filter(theme_label=active_theme).count()
    completed_count = Video.objects.filter(theme_label=active_theme, analysis_status=2).count()
    
    if total_count == 0 or (completed_count / total_count) < 0.9:
        return JsonResponse({'success': False, 'message': '分析进度未达到 90%，无法触发重训练'})

    try:
        import subprocess

        def _run_retrain():
            # 在子进程中执行训练脚本，避免 GPU 上下文冲突
            try:
                script_path = os.path.join(settings.BASE_DIR, 'ml_pipeline', 'train_master_arena.py')
                logger.info(f"[Retrain] *开始执行训练脚本: {script_path}*")

                result = subprocess.run(
                    [sys.executable, script_path],
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10分钟超时
                    cwd=str(settings.BASE_DIR)
                )

                if result.returncode == 0:
                    logger.info("[Retrain] *训练完成，正在热替换模型单例...*")
                    # 热替换: 销毁 DiggPredictionService 单例，下次请求时自动重新加载新模型
                    try:
                        from services.predict_service import DiggPredictionService
                        DiggPredictionService._instance = None
                        DiggPredictionService._initialized = False
                        logger.info("[Retrain] *模型单例已重置，下次推理将加载新模型*")
                    except Exception as swap_err:
                        logger.warning(f"[Retrain] *单例重置失败（非致命）: {swap_err}*")
                else:
                    logger.error(f"[Retrain] *训练脚本执行失败: {result.stderr}*")
            except subprocess.TimeoutExpired:
                logger.error("[Retrain] *训练脚本执行超时 (>600s)*")
            except Exception as e:
                logger.error(f"[Retrain] *后台训练异常: {e}*")

        # 在后台线程中启动训练，避免阻塞 HTTP 响应
        retrain_thread = threading.Thread(target=_run_retrain, daemon=True)
        retrain_thread.start()

        return JsonResponse({
            'success': True,
            'message': '模型重训练任务已提交后台执行，完成后将自动热替换模型。'
        })

    except Exception as e:
        logger.error(f"[Retrain] *API 异常: {e}*")
        return JsonResponse({'success': False, 'message': f'服务器内部错误: {e}'})

