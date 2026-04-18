# video_analyzer.py 增强版 V3.2 (Timeout Protection + RTX 3060 GPU 加速)
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
import subprocess
import uuid
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

# 忽略 librosa 的低电平警告
warnings.filterwarnings('ignore')

# ==========================================
# 核心配置：FFmpeg 路径强行指定
# ==========================================
FFMPEG_MANUAL_PATH = r"C:\ffmpeg\ffmpeg-8.0.1-essentials_build\bin"

# ==========================================
# [NEW] 超时保护配置 (Timeout Protection)
# ==========================================
TIMEOUT_CONFIG = {
    'whisper_asr': 60,        # Whisper 语音识别超时 (秒)
    'bpm_detection': 15,      # BPM 节拍检测超时 (秒)
    'visual_analysis': 45,    # 视觉特征分析超时 (秒)
}

import threading

# 全局模型缓存，避免重复加载
_CACHED_WHISPER = None
# 全局推理锁，保障多线程安全，避免 OOM 和 GPU 状态损坏
_WHISPER_LOCK = threading.Lock()


class VideoContentAnalyzer:
    """
    多模态视频分析器 (Web Service Edition):
    1. 视觉：基于 RTX 3060 (CUDA) 进行 HSV 色彩空间与亮度的并行计算。
    2. 音频：强制注入 FFmpeg 环境，确保 BPM 提取准确。
    3. 资源：分析结束后主动释放显存，保障 Django 服务稳定性。
    """

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

    def _inject_ffmpeg_path(self):
        """Web 端环境注入逻辑：确保 Django 进程能找到 ffmpeg.exe"""
        if os.path.exists(FFMPEG_MANUAL_PATH):
            # 将路径加入系统 PATH 的最前面
            if FFMPEG_MANUAL_PATH not in os.environ["PATH"]:
                os.environ["PATH"] = FFMPEG_MANUAL_PATH + os.pathsep + os.environ["PATH"]
                # 仅在第一次注入时打印，避免日志刷屏
                print(f"[System] FFmpeg 环境已注入 Web 进程: {FFMPEG_MANUAL_PATH}")

    def _run_with_timeout(self, func, timeout_sec, default_value, stage_name="Unknown"):
        """
        *[NEW] 超时保护包装器：防止单个操作阻塞整个线程*
        *使用 ThreadPoolExecutor 实现跨平台超时控制*
        
        Args:
            func: 要执行的 callable
            timeout_sec: 超时秒数
            default_value: 超时后返回的默认值
            stage_name: 阶段名称，用于日志
        
        Returns:
            func 的返回值，或超时后的 default_value
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

    def run_full_analysis(self):
        """执行全量分析并返回物理特征字典"""
        
        # Heartbeat Log: 明确显示当前正在使用哪个硬件进行计算
        print(f"[AIWorker] ⚡ GPU Computing: {self.video_id} on {self.device}")
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return self._get_default_features()

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        brightness_list = []
        saturation_list = []
        cut_points = 0
        prev_hist = None

        # 优化采样步长：Web 端追求响应速度，每秒仅采 1 帧 (极速模式)
        sample_step = max(int(fps), 1)
        
        # === [NEW] 智能损坏检测配置 ===
        consecutive_failures = 0           # 连续读取失败计数
        MAX_CONSECUTIVE_FAILURES = 50      # 连续失败阈值，超过此值视为损坏视频
        successful_frames = 0              # 成功读取的帧数
        MIN_SUCCESSFUL_FRAMES = 5          # 最少需要成功读取的帧数

        try:
            for i in range(0, total_frames, sample_step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                
                if not ret:
                    # 读取失败，累加计数
                    consecutive_failures += 1
                    
                    # === 智能损坏检测：连续失败超过阈值 ===
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        print(f"🔴 *[Corruption Detected] {consecutive_failures} consecutive read failures on: {self.video_id}*")
                        print(f"   *Aborting analysis to prevent FFmpeg infinite loop.*")
                        # 提前跳出循环，使用已收集的数据
                        break
                    continue  # 跳过这一帧，尝试下一帧
                
                # 读取成功，重置失败计数
                consecutive_failures = 0
                successful_frames += 1

                # --- GPU 并行运算加速区 ---
                try:
                    # 将图像矩阵搬运至 RTX 3060 显存
                    t_frame = torch.from_numpy(frame).float().to(self.device)

                    # 维度 1: 亮度 (并行求均值)
                    brightness_list.append(torch.mean(t_frame).item())

                    # 维度 2: 饱和度 (Max-Min 矩阵并行运算，模拟 HSV 的 S 分量)
                    # Web 端为了速度，直接用 RGB 极差近似饱和度，与离线端逻辑对齐
                    sat_tensor = torch.max(t_frame, dim=2)[0] - torch.min(t_frame, dim=2)[0]
                    saturation_list.append(torch.mean(sat_tensor).item())

                    # 维度 3: 转场检测 (保持 Web 端特有的 CPU 优化逻辑，因为 hist 计算 OpenCV CPU 版本更快)
                    # 先缩小图像，大幅降低 CPU 负载
                    small_frame = cv2.resize(frame, (640, 360))
                    # 转换到 HSV 空间检测直方图变化，比 RGB 更准
                    hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
                    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
                    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

                    if prev_hist is not None:
                        corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                        # 相关性低于 0.6 视为镜头切换
                        if corr < 0.6:
                            cut_points += 1
                    prev_hist = hist

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"🔥 [GPU Memory Alert] OOM detected during frame processing. Suggest closing other GPU-heavy apps.")
                        torch.cuda.empty_cache()
                        # 降级策略: 如果 OOM，跳过当前帧或清空缓存后继续
                    pass
                except Exception as e:
                    # 容错：如果 GPU 显存偶尔波动，跳过该帧不影响整体结果
                    pass

        except Exception as e:
            print(f"[Analysis Error] {e}")

        finally:
            cap.release()
            
            # 【关键】Web 服务必须手动清理显存垃圾
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # === [NEW] 损坏视频最终验证 ===
        # 如果成功读取的帧数过少，视为严重损坏
        if successful_frames < MIN_SUCCESSFUL_FRAMES:
            print(f"⚠️ *[Corruption Warning] Only {successful_frames}/{MIN_SUCCESSFUL_FRAMES} frames read successfully: {self.video_id}*")
            # 仍然尝试返回部分数据，不抛出异常

        # 计算核心指标
        visual_brightness_val = round(float(np.mean(brightness_list)), 2) if brightness_list else 0
        visual_saturation_val = round(float(np.mean(saturation_list)), 2) if saturation_list else 0
        cut_frequency_val = round(float(cut_points / duration), 2) if duration > 5 else 0

        # ================================================================
        # [TIMEOUT PROTECTED] 音频分析 - 15秒超时
        # ================================================================
        audio_bpm = self._run_with_timeout(
            func=self._analyze_audio_safe,
            timeout_sec=TIMEOUT_CONFIG['bpm_detection'],
            default_value=120,  # 超时返回默认 BPM
            stage_name="BPM Detection"
        )

        # ================================================================
        # [TIMEOUT PROTECTED] ASR 语音识别 - 60秒超时 (最耗时)
        # ================================================================
        keywords = self._run_with_timeout(
            func=self._extract_audio_keywords,
            timeout_sec=TIMEOUT_CONFIG['whisper_asr'],
            default_value=[],  # 超时返回空列表，触发氛围标签 fallback
            stage_name="Whisper ASR"
        )

        # [Fallback Mechanism] 氛围标签补全
        # 如果 ASR 结果为空（纯音乐/风景），根据视听指标生成氛围标签
        if not keywords:
            fallback_tags = []
            
            # 1. 听觉维度
            if audio_bpm > 120:
                fallback_tags.append({'name': '高燃节奏', 'value': 95})
            elif audio_bpm < 80:
                fallback_tags.append({'name': '舒缓治愈', 'value': 85})
            
            # 2. 视觉维度
            if visual_saturation_val > 70:
                fallback_tags.append({'name': '视觉鲜艳', 'value': 80})
            if visual_brightness_val < 60:
                fallback_tags.append({'name': '深邃氛围', 'value': 75})
            
            # 3. 剪辑维度
            if cut_frequency_val > 0.5:
                fallback_tags.append({'name': '快速剪辑', 'value': 90})
            
            # 4. 兜底默认
            if not fallback_tags:
                fallback_tags.append({'name': '纯享版', 'value': 60})
            
            keywords = fallback_tags

        return {
            "duration_sec": round(duration, 2),
            "visual_brightness": visual_brightness_val,
            "visual_saturation": visual_saturation_val,
            "cut_frequency": cut_frequency_val,
            "audio_bpm": audio_bpm,
            "script_keywords": keywords
        }

    def _extract_audio_keywords(self):
        """
        *核心升级：提取音频 -> Whisper 转录 -> Jieba 提取关键词*
        """
        temp_audio_path = f"temp_audio_{uuid.uuid4().hex}.wav"

        try:
            # 1. 使用 ffmpeg 仅提取前 60 秒音频
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    self.video_path,
                    "-t",
                    "60",
                    "-vn",
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    temp_audio_path,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )

            # 2. 使用全局锁：确保多线程下 GPU 计算与内存分配的原子性
            with _WHISPER_LOCK:
                global _CACHED_WHISPER
                if "_CACHED_WHISPER" not in globals() or _CACHED_WHISPER is None:
                    print(f"⚡ [加速] 首次加载 Whisper 模型到 {self.device}...")
                    _CACHED_WHISPER = whisper.load_model("base", device=self.device)

                if _CACHED_WHISPER.device != self.device:
                    _CACHED_WHISPER = _CACHED_WHISPER.to(self.device)

                result = _CACHED_WHISPER.transcribe(
                    temp_audio_path,
                    fp16=(self.device.type == 'cuda')
                )

            text = result.get('text', '')

            if not text:
                return []

            # 3. Jieba 提取关键词
            keywords = jieba.analyse.extract_tags(text, topK=10, withWeight=True)

            return [{"name": k, "value": v} for k, v in keywords]

        except subprocess.CalledProcessError:
            return []
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"🔥 [ASR Error] CUDA OOM during Whisper transcription.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return []
        except Exception as e:
            print(f"[ASR Error] {e}")
            return []

        finally:
            try:
                os.remove(temp_audio_path)
            except FileNotFoundError:
                pass
            except Exception:
                pass

    def _analyze_audio_safe(self):
        """鲁棒的音频分析，集成 FFmpeg 支持"""
        try:
            # duration=20: 只读取前 20 秒，提升 Web 响应速度
            y, sr = librosa.load(self.video_path, sr=22050, duration=20, mono=True)

            # 静音检测：如果信号太弱或太短，直接返回默认值
            if y is None or len(y) < 100:
                return 120

            # 使用 tempo 检测 (比 beat_track 更适合短视频 BGM)
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)

            # 兼容性处理：处理 numpy array 返回值
            if np.ndim(tempo) > 0:
                return int(tempo[0])
            return int(tempo)

        except Exception as e:
            # 仅在严重错误时打印，防止日志刷屏
            # print(f"[Audio Error] {e}")
            return 120

    def _get_default_features(self):
        """兜底返回默认值"""
        return {
            "duration_sec": 0,
            "visual_brightness": 0,
            "visual_saturation": 0,
            "cut_frequency": 0,
            "audio_bpm": 120
        }
