import pandas as pd
import cv2
import torch
import librosa
import numpy as np
import os
import time
import shutil  # 用于检测 ffmpeg 是否存在
import traceback  # 用于打印详细的错误堆栈
from tqdm import tqdm

# ==========================================
# 用户配置区 (User Configuration Zone)
# ==========================================
# 1. 你的项目根目录
ROOT_DIR = r"D:\renhangxi_tiktok_bysj"

# 2. 原始输入文件 (请确认文件名是否对应当前主题)
INPUT_CSV = r"D:\renhangxi_tiktok_bysj\data\douyin_video_sikachi.csv"

# 3. 特征工厂产出的最终文件
OUTPUT_CSV = r"D:\renhangxi_tiktok_bysj\multy_video_data\douyin_data_sikachi_with_gpu_features.csv"

# 4. 显卡配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 5. 【⭐ 关键修复】手动指定 FFmpeg 的 bin 路径
# 请将下面的路径修改为你电脑上实际存放 ffmpeg.exe 的文件夹路径
# 注意：路径前面加 r，并且不要指向 .exe 文件，而是指向 ...\bin 文件夹
FFMPEG_MANUAL_PATH = r"C:\ffmpeg\ffmpeg-8.0.1-essentials_build\bin"


# ↑↑↑↑↑ 请务必修改这里！如果你的路径不一样，请替换它！ ↑↑↑↑↑

# ==========================================


class MultiModalFeatureFactory:
    """
    特征工厂：集成 3060 算力提取视频视觉、音频特征，并与原始文本数据整合
    [V5 Update]: 强制注入环境变量，解决 IDE 缓存导致的 NoBackendError
    """

    def __init__(self):
        print("=" * 60)
        print("🚀 特征工厂启动 (V5 Force-Link Mode)...")
        print(f"📍 当前根目录: {ROOT_DIR}")
        print(f"💻 算力设备: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

        # --- 强制环境注入 (Force Environment Injection) ---
        self._inject_ffmpeg_path()

        # --- 环境自检 ---
        self.check_environment()
        print("=" * 60)

    def _inject_ffmpeg_path(self):
        """
        核心修复逻辑：将用户指定的 FFmpeg 路径强行加入 Python 运行时的 PATH 变量中
        """
        if os.path.exists(FFMPEG_MANUAL_PATH):
            # 将路径加入系统 PATH 的最前面，确保优先读取
            os.environ["PATH"] = FFMPEG_MANUAL_PATH + os.pathsep + os.environ["PATH"]
            print(f"💉 [系统注入] 已强制添加 FFmpeg 路径: {FFMPEG_MANUAL_PATH}")
        else:
            print(f"⚠️ [路径警告] 你填写的 FFMPEG_MANUAL_PATH 不存在: {FFMPEG_MANUAL_PATH}")
            print("   请检查代码第 28 行是否填写正确！")

    def check_environment(self):
        """
        检查系统依赖
        """
        print("🔍 正在检查系统环境依赖...")

        # shutil.which 现在应该能通过 injected path 找到 ffmpeg 了
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            print(f"   ✅ FFmpeg 已就绪: {ffmpeg_path}")
            print("      (音频解码引擎已激活，Librosa 将正常工作)")
        else:
            print("   ❌ [致命警告] 依然未检测到 FFmpeg！")
            print("      原因可能为：")
            print("      1. FFMPEG_MANUAL_PATH 填写的路径不对（必须是包含 ffmpeg.exe 的 bin 文件夹）")
            print("      2. 文件夹权限问题")

    def extract_visual_on_gpu(self, video_path):
        """
        利用 3060 显卡进行像素级分析
        """
        if not os.path.exists(video_path):
            return 0.0, 0.0, 0.0

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0.0, 0.0, 0.0

        brightness_list = []
        saturation_list = []
        cut_count = 0
        prev_hist = None
        total_processed_frames = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or total_processed_frames > 500:
                break

            if total_processed_frames % 5 == 0:
                try:
                    t_frame = torch.from_numpy(frame).float().to(DEVICE)
                    brightness_list.append(torch.mean(t_frame).item())
                    sat = torch.max(t_frame, dim=2)[0] - torch.min(t_frame, dim=2)[0]
                    saturation_list.append(torch.mean(sat).item())

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                    if prev_hist is not None:
                        diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR)
                        if diff > 10000:
                            cut_count += 1
                    prev_hist = hist
                except Exception:
                    pass

            total_processed_frames += 1

        cap.release()
        duration_sampled = total_processed_frames / 30.0
        cut_freq = cut_count / duration_sampled if duration_sampled > 0 else 0
        avg_bright = np.mean(brightness_list) if brightness_list else 0.0
        avg_sat = np.mean(saturation_list) if saturation_list else 0.0
        return avg_bright, avg_sat, cut_freq

    def extract_audio_bpm(self, video_path):
        """
        利用音频库提取音乐节奏 BPM
        """
        try:
            if not os.path.exists(video_path):
                print(f"   ❌ [音频错误] 文件不存在: {video_path}")
                return 120.0

            # 这里的 duration=30 意味着只读前30秒
            y, sr = librosa.load(video_path, sr=None, duration=30)

            if y is None or len(y) == 0:
                print("   ⚠️ [音频为空] 无法提取特征")
                return 120.0

            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

            if np.ndim(tempo) > 0:
                tempo = tempo[0]

            return float(tempo)

        except Exception as e:
            # 此时如果 FFmpeg 注入成功，这里应该不会再进来了
            file_name = os.path.basename(video_path)
            print(f"\n   🔴 [音频解析崩溃] 文件: {file_name}")
            print(f"   🔧 错误摘要: {e}")
            # 如果是 NoBackendError，说明注入还是失败了
            if "NoBackendError" in str(e) or "NoBackendError" in str(type(e)):
                print("   👉 提示：FFmpeg 路径可能依然没对，请检查 config 区的路径拼写！")
            else:
                traceback.print_exc()

            return 120.0

    def start_factory_line(self):
        if not os.path.exists(INPUT_CSV):
            print(f"❌ 错误：找不到输入文件 {INPUT_CSV}")
            return

        try:
            df = pd.read_csv(INPUT_CSV)
            print(f"📋 已加载原始数据，共计 {len(df)} 条视频记录。")
        except Exception as e:
            print(f"❌ 读取 CSV 失败: {e}")
            return

        new_features = []

        for index, row in df.iterrows():
            video_id = row['视频ID']
            raw_rel_path = str(row['本地路径'])
            rel_path = raw_rel_path.replace('\\', os.sep).replace('/', os.sep)

            if rel_path.startswith(ROOT_DIR):
                full_path = rel_path
            else:
                if rel_path.startswith(os.sep):
                    rel_path = rel_path.lstrip(os.sep)
                full_path = os.path.join(ROOT_DIR, rel_path)

            print(f"\n[{index + 1}/{len(df)}] 正在加工视频 ID: {video_id}")
            display_path = os.sep.join(full_path.split(os.sep)[-3:])
            print(f"   📂 目标路径: ...{os.sep}{display_path}")

            if not os.path.exists(full_path):
                print(f"   ⚠️  [跳过] 找不到物理文件: {full_path}")
                new_features.append([0.0, 0.0, 0.0, 110.0])
                continue

            start_t = time.time()
            print(f"   ⚡ 步骤 1: 3060 显卡正在分析画面像素...", end="\r")
            bright, sat, cut_f = self.extract_visual_on_gpu(full_path)

            print(f"   🎵 步骤 2: 正在解析背景音乐节奏...", end="\r")
            bpm = self.extract_audio_bpm(full_path)

            cost_t = time.time() - start_t
            print(f"   ✅ 加工完成！耗时: {cost_t:.2f}s | 亮度: {bright:.1f} | 切镜: {cut_f:.2f} | BPM: {bpm:.1f}")

            new_features.append([round(bright, 2), round(sat, 2), round(cut_f, 2), round(bpm, 2)])

        feature_cols = ['visual_brightness', 'visual_saturation', 'cut_frequency', 'audio_bpm']
        features_df = pd.DataFrame(new_features, columns=feature_cols)

        # 重置索引以防对齐错误
        df.reset_index(drop=True, inplace=True)
        features_df.reset_index(drop=True, inplace=True)
        final_df = pd.concat([df, features_df], axis=1)

        output_dir = os.path.dirname(OUTPUT_CSV)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        final_df.to_csv(OUTPUT_CSV, index=False, encoding='utf_8_sig')

        print("\n" + "=" * 60)
        print("🎉 特征工厂任务结束！")
        print(f"📁 整合后的完整多模态数据集已保存至: {OUTPUT_CSV}")
        print("=" * 60)


if __name__ == "__main__":
    factory = MultiModalFeatureFactory()
    factory.start_factory_line()