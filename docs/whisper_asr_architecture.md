# VideoContentAnalyzer Architecture - Whisper Logic

## Imports & Dependencies
```python
import cv2
import librosa
import numpy as np
import os
import warnings
import torch
import shutil
import jieba
import jieba.analyse
import whisper
from moviepy.editor import VideoFileClip
import uuid
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import threading
```

## Class Initialization
```python
    def __init__(self, video_path, video_id=None):
        self.video_path = video_path
        self.video_id = video_id or "Unknown_Video"

        # 1. 强制注入 FFmpeg 环境变量 (修复 Web 端 BPM=120 问题)
        self._inject_ffmpeg_path()

        # 2. 自动检测算力设备，优先使用 RTX 3060
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 [Hardware-Init] VideoContentAnalyzer initialized using: {self.device}")

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"未找到视频文件: {video_path}")
```

## The Core Whisper Method
```python
    def _run_with_timeout(self, func, timeout_sec, default_value, stage_name="Unknown"):
        """
        *[NEW] 超时保护包装器：防止单个操作阻塞整个线程*
        *使用 ThreadPoolExecutor 实现跨平台超时控制*
        """
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func)
            try:
                return future.result(timeout=timeout_sec)
            except FuturesTimeoutError:
                print(f"⏰ *[Timeout] {stage_name} 超时 ({timeout_sec}s)，返回默认值: {self.video_id}*")
                return default_value
            except Exception as e:
                print(f"❌ *[Error] {stage_name} 执行失败: {e}*")
                return default_value

    def _extract_audio_keywords(self):
        """
        *核心升级：提取音频 -> Whisper 转录 -> Jieba 提取关键词*
        *V3.2: 修复 VideoFileClip 资源泄漏问题，确保 ffmpeg 进程被正确释放*
        """
        temp_audio_path = f"temp_audio_{uuid.uuid4().hex}.wav"
        video = None
        
        try:
            # 1. *提取音频到临时文件*
            video = VideoFileClip(self.video_path)
            if video.audio is None:
                return []
            
            video.audio.write_audiofile(temp_audio_path, verbose=False, logger=None)

            # 2. *使用全局锁：确保多线程下 GPU 计算与内存分配的原子性*
            with _WHISPER_LOCK:
                # 3. *加载 Whisper 模型并转录 (带缓存)*
                global _CACHED_WHISPER
                if _CACHED_WHISPER is None:
                    print(f"⚡ [加速] 首次加载 Whisper 模型到 {self.device}...")
                    _CACHED_WHISPER = whisper.load_model("base", device=self.device)
                
                # *如果模型不在当前设备（例如缓存的模型在 CPU，但当前检测到 GPU），移动它*
                if _CACHED_WHISPER.device != self.device:
                     _CACHED_WHISPER = _CACHED_WHISPER.to(self.device)

                result = _CACHED_WHISPER.transcribe(temp_audio_path)
            
            text = result.get('text', '')

            if not text:
                return []

            # 4. *Jieba 提取关键词*
            # *topK=10: 只取权重最高的10个词*
            keywords = jieba.analyse.extract_tags(text, topK=10, withWeight=True)
            
            # *格式化返回：[{'name': '词', 'value': 权重}, ...]*
            return [{"name": k, "value": v} for k, v in keywords]

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"🔥 [ASR Error] CUDA OOM during Whisper transcription.")
                torch.cuda.empty_cache()
            return []
        except Exception as e:
            # *捕获所有异常，确保主流程不崩溃*
            print(f"[ASR Error] {e}")
            return []
        
        finally:
            # 4. *【关键修复】无论成功或异常，都必须关闭 VideoFileClip*
            # *这会终止底层 ffmpeg 子进程，释放文件句柄*
            if video is not None:
                try:
                    video.close()
                except Exception as close_err:
                    print(f"[Resource Cleanup Warning] Failed to close VideoFileClip: {close_err}")
            
            # 5. *清理临时音频文件*
            if os.path.exists(temp_audio_path):
                try:
                    os.remove(temp_audio_path)
                except Exception as rm_err:
                    # *文件可能被进程锁定，记录但不阻塞*
                    print(f"[Temp File Warning] Could not remove {temp_audio_path}: {rm_err}")
```

## Audio Extraction Logic
```python
            # 1. *提取音频到临时文件*
            video = VideoFileClip(self.video_path)
            if video.audio is None:
                return []
            
            video.audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
```
