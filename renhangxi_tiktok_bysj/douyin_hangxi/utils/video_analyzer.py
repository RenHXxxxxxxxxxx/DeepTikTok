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
    'audio_extract': 20,      # 音频抽取超时 (秒)
    'whisper_asr': 60,        # Whisper 语音识别超时 (秒)
    'bpm_detection': 15,      # BPM 节拍检测超时 (秒)
    'visual_analysis': 45,    # 视觉特征分析超时 (秒)
}

import threading

# 全局模型缓存，避免重复加载
_CACHED_WHISPER = None
# 全局推理锁，保障多线程安全，避免 OOM 和 GPU 状态损坏
_WHISPER_LOCK = threading.Lock()
_FFMPEG_PATH_LOCK = threading.Lock()
_FFMPEG_PATH_READY = False
_ANALYZER_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        self._gpu_cleanup_needed = False
        self._visual_batch_size = 8

        # 1. 强制注入 FFmpeg 环境变量 (修复 Web 端 BPM=120 问题)
        self._inject_ffmpeg_path()

        # 2. 自动检测算力设备，优先使用 RTX 3060
        self.device = _ANALYZER_DEVICE
        print(f" [Hardware-Init] VideoContentAnalyzer initialized using: {self.device}")

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"未找到视频文件: {video_path}")

    def _inject_ffmpeg_path(self):
        """Web 端环境注入逻辑：确保 Django 进程能找到 ffmpeg.exe"""
        global _FFMPEG_PATH_READY
        if _FFMPEG_PATH_READY or not os.path.exists(FFMPEG_MANUAL_PATH):
            return

        with _FFMPEG_PATH_LOCK:
            if _FFMPEG_PATH_READY:
                return

            path_value = os.environ.get("PATH", "")
            path_entries = path_value.split(os.pathsep) if path_value else []

            if FFMPEG_MANUAL_PATH not in path_entries:
                os.environ["PATH"] = FFMPEG_MANUAL_PATH + os.pathsep + path_value
                print(f"[System] FFmpeg 环境已注入 Web 进程: {FFMPEG_MANUAL_PATH}")

            _FFMPEG_PATH_READY = True

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
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(func)
        try:
            return future.result(timeout=timeout_sec)
        except FuturesTimeoutError:
            print(f"[Timeout] {stage_name} 超时 ({timeout_sec}s)，返回默认值: {self.video_id}")
            future.cancel()
            return default_value
        except Exception as e:
            print(f"[Error] {stage_name} 执行失败: {e}")
            return default_value
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _process_visual_batch(self, frame_batch):
        """小批量处理视觉张量，减少逐帧 CPU<->GPU 同步开销。"""
        if not frame_batch:
            return [], []

        try:
            np_batch = np.stack(frame_batch).astype(np.float32, copy=False)
            t_batch = torch.from_numpy(np_batch)
            if self.device.type == 'cuda':
                t_batch = t_batch.to(self.device, non_blocking=True)

            brightness_vals = t_batch.mean(dim=(1, 2, 3))
            saturation_vals = (
                t_batch.max(dim=3).values - t_batch.min(dim=3).values
            ).mean(dim=(1, 2))

            return (
                brightness_vals.detach().cpu().tolist(),
                saturation_vals.detach().cpu().tolist()
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self._gpu_cleanup_needed = True
                print(" [GPU Memory Alert] OOM detected during visual batch processing.")
            return [], []
        except Exception:
            return [], []

    def _extract_audio_segment(self, max_duration=60):
        """统一抽取一份音频临时文件，供 BPM 与 ASR 复用。"""
        temp_audio_path = f"temp_audio_{uuid.uuid4().hex}.wav"
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    self.video_path,
                    "-t",
                    str(max_duration),
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
                timeout=TIMEOUT_CONFIG['audio_extract'],
            )
            return temp_audio_path
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            try:
                os.remove(temp_audio_path)
            except FileNotFoundError:
                pass
            except Exception:
                pass
            return None

    def run_full_analysis(self, include_script_keywords=True):
        """执行全量分析并返回物理特征字典"""
        
        # Heartbeat Log: 明确显示当前正在使用哪个硬件进行计算
        print(f"[AIWorker]  GPU Computing: {self.video_id} on {self.device}")
        
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
        sampled_frames = []

        # 优化采样步长：Web 端追求响应速度，每秒仅采 1 帧 (极速模式)
        sample_step = max(int(fps), 1)
        
        # === [NEW] 智能损坏检测配置 ===
        consecutive_failures = 0           # 连续读取失败计数
        MAX_CONSECUTIVE_FAILURES = 50      # 连续失败阈值，超过此值视为损坏视频
        successful_frames = 0              # 成功读取的帧数
        MIN_SUCCESSFUL_FRAMES = 5          # 最少需要成功读取的帧数

        try:
            frame_idx = 0
            while True:
                if total_frames > 0 and frame_idx >= total_frames:
                    break

                should_sample = (frame_idx % sample_step) == 0
                if should_sample:
                    ret, frame = cap.read()
                else:
                    ret = cap.grab()
                    frame = None
                
                if not ret:
                    # 读取失败，累加计数
                    consecutive_failures += 1
                    
                    # === 智能损坏检测：连续失败超过阈值 ===
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        print(f" [Corruption Detected] {consecutive_failures} consecutive read failures on: {self.video_id}")
                        print("   Aborting analysis to prevent FFmpeg infinite loop.")
                        # 提前跳出循环，使用已收集的数据
                        break

                    if total_frames <= 0 or frame_idx >= max(total_frames - 1, 0):
                        break

                    frame_idx += 1
                    continue  # 跳过这一帧，尝试下一帧
                
                # 读取成功，重置失败计数
                consecutive_failures = 0

                if frame is not None:
                    successful_frames += 1
                    sampled_frames.append(frame)

                    try:
                        # 维度 3: 转场检测 (保持 CPU 路径，避免引入高风险视觉重构)
                        small_frame = cv2.resize(frame, (640, 360))
                        hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
                        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
                        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

                        if prev_hist is not None:
                            corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                            if corr < 0.6:
                                cut_points += 1
                        prev_hist = hist
                    except Exception:
                        pass

                    if len(sampled_frames) >= self._visual_batch_size:
                        batch_brightness, batch_saturation = self._process_visual_batch(sampled_frames)
                        brightness_list.extend(batch_brightness)
                        saturation_list.extend(batch_saturation)
                        sampled_frames.clear()

                frame_idx += 1

        except Exception as e:
            print(f"[Analysis Error] {e}")

        finally:
            cap.release()

        if sampled_frames:
            batch_brightness, batch_saturation = self._process_visual_batch(sampled_frames)
            brightness_list.extend(batch_brightness)
            saturation_list.extend(batch_saturation)

        # === [NEW] 损坏视频最终验证 ===
        # 如果成功读取的帧数过少，视为严重损坏
        if successful_frames < MIN_SUCCESSFUL_FRAMES:
            print(f" [Corruption Warning] Only {successful_frames}/{MIN_SUCCESSFUL_FRAMES} frames read successfully: {self.video_id}")
            # 仍然尝试返回部分数据，不抛出异常

        # 计算核心指标
        visual_brightness_val = round(float(np.mean(brightness_list)), 2) if brightness_list else 0
        visual_saturation_val = round(float(np.mean(saturation_list)), 2) if saturation_list else 0
        cut_frequency_val = round(float(cut_points / duration), 2) if duration > 5 else 0

        # ================================================================
        # [TIMEOUT PROTECTED] 音频分析 - 15秒超时
        # ================================================================
        keywords = []
        audio_bpm = 120
        shared_audio_path = None

        try:
            if include_script_keywords:
                shared_audio_path = self._extract_audio_segment(max_duration=60)

                if shared_audio_path:
                    audio_bpm = self._run_with_timeout(
                        func=lambda: self._analyze_audio_safe(shared_audio_path),
                        timeout_sec=TIMEOUT_CONFIG['bpm_detection'],
                        default_value=120,
                        stage_name="BPM Detection"
                    )
                    keywords = self._run_with_timeout(
                        func=lambda: self._extract_audio_keywords(shared_audio_path),
                        timeout_sec=TIMEOUT_CONFIG['whisper_asr'],
                        default_value=[],
                        stage_name="Whisper ASR"
                    )
                else:
                    audio_bpm = self._run_with_timeout(
                        func=self._analyze_audio_safe,
                        timeout_sec=TIMEOUT_CONFIG['bpm_detection'],
                        default_value=120,
                        stage_name="BPM Detection"
                    )
            else:
                audio_bpm = self._run_with_timeout(
                    func=self._analyze_audio_safe,
                    timeout_sec=TIMEOUT_CONFIG['bpm_detection'],
                    default_value=120,
                    stage_name="BPM Detection"
                )
        finally:
            if shared_audio_path:
                try:
                    os.remove(shared_audio_path)
                except FileNotFoundError:
                    pass
                except Exception:
                    pass

        # [Fallback Mechanism] 氛围标签补全
        # 如果 ASR 结果为空（纯音乐/风景），根据视听指标生成氛围标签
        if include_script_keywords and not keywords:
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

        if self.device.type == 'cuda' and self._gpu_cleanup_needed:
            torch.cuda.empty_cache()
            self._gpu_cleanup_needed = False

        return {
            "duration_sec": round(duration, 2),
            "visual_brightness": visual_brightness_val,
            "visual_saturation": visual_saturation_val,
            "cut_frequency": cut_frequency_val,
            "audio_bpm": audio_bpm,
            "script_keywords": keywords
        }

    def _extract_audio_keywords(self, audio_path=None):
        """
        *核心升级：提取音频 -> Whisper 转录 -> Jieba 提取关键词*
        """
        temp_audio_path = audio_path
        owns_temp_audio = False

        try:
            if not temp_audio_path:
                temp_audio_path = self._extract_audio_segment(max_duration=60)
                owns_temp_audio = True

            if not temp_audio_path or not os.path.exists(temp_audio_path):
                return []

            # 2. 使用全局锁：确保多线程下 GPU 计算与内存分配的原子性
            with _WHISPER_LOCK:
                global _CACHED_WHISPER
                if "_CACHED_WHISPER" not in globals() or _CACHED_WHISPER is None:
                    print(f" [加速] 首次加载 Whisper 模型到 {self.device}...")
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
                print(f" [ASR Error] CUDA OOM during Whisper transcription.")
                self._gpu_cleanup_needed = True
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return []
        except Exception as e:
            print(f"[ASR Error] {e}")
            return []

        finally:
            if owns_temp_audio and temp_audio_path:
                try:
                    os.remove(temp_audio_path)
                except FileNotFoundError:
                    pass
                except Exception:
                    pass

    def _analyze_audio_safe(self, audio_source_path=None):
        """鲁棒的音频分析，集成 FFmpeg 支持"""
        try:
            # duration=20: 只读取前 20 秒，提升 Web 响应速度
            y, sr = librosa.load(audio_source_path or self.video_path, sr=22050, duration=20, mono=True)

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
        if self.device.type == 'cuda' and self._gpu_cleanup_needed:
            torch.cuda.empty_cache()
            self._gpu_cleanup_needed = False

        return {
            "duration_sec": 0,
            "visual_brightness": 0,
            "visual_saturation": 0,
            "cut_frequency": 0,
            "audio_bpm": 120,
            "script_keywords": []
        }

