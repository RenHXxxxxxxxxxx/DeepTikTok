import os
from openai import OpenAI
import time
from dotenv import load_dotenv

# 显式加载环境变量
load_dotenv()

# 全局配置：配置驱动架构
GLOBAL_CONFIG = {
    "BASE_URL": "https://api.deepseek.com",
    "DEFAULT_MODEL": "deepseek-chat",
    "MAX_RETRIES": 3,
    "INITIAL_BACKOFF": 1.0,
    "BACKOFF_FACTOR": 2.0
}

class LLMService:
    _instance = None
    _client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMService, cls).__new__(cls)
            cls._instance._init_client()
        return cls._instance

    def _init_client(self):
        # 全局开发密钥后退机制已被彻底移除，实现严格的用户级密钥隔离
        self.base_url = GLOBAL_CONFIG["BASE_URL"]

    def generate_advice(self, data, user_key=None, model_name=None):
        # 生成运营建议
        if not user_key or not str(user_key).strip():
            raise PermissionError("API Key not configured. Please set your credentials in the Profile page.")
            
        # [核心修复]：无视外界传来的 model_name，强制使用全局配置的 deepseek-chat
        actual_model = GLOBAL_CONFIG["DEFAULT_MODEL"]
        
        try:
            client = OpenAI(api_key=str(user_key).strip(), base_url=self.base_url)
        except Exception:
            return "*AI 诊断模块暂时不可用。*"

        # 提取数据并设置默认值
        brightness_val = data.get('visual_brightness', 0)
        saturation_val = data.get('visual_saturation', 0)
        bpm_val = data.get('audio_bpm', 0)
        cut_freq_val = data.get('cut_frequency', 0)
        pred_likes = data.get('predicted_likes', '未知')
        theme_p25 = data.get('theme_p25', 0)
        theme_p50 = data.get('theme_p50', 0)
        theme_p75 = data.get('theme_p75', 0)
        follower_count = data.get('follower_count', 0)
        publish_hour = data.get('publish_hour', '未知')
        quality_score = data.get('quality_score', 0)
        percentile_rank = data.get('percentile_rank', '未知')
        optimal_times_str = ", ".join(data.get('optimal_publishing_times', []))

        # 计算基础状态
        brightness_status = "偏暗" if brightness_val < 60 else "正常"
        bpm_status = "舒缓" if bpm_val < 100 else "动感"
        cut_status = "节奏快" if cut_freq_val > 0.8 else "节奏稳健"

        # 构建具备 Chain-of-Thought 特性的 Few-Shot 提示词
        prompt = f"""
# 角色设定：资深算法工程师 + 抖音爆款内容操盘手

请基于以下提供的多模态特征数据，为这支“高燃剪辑”方向的视频出具一份结构清晰的诊断报告。
请严格按照以下三个部分输出，合理使用 Emoji 和 Markdown 语法（请务必加粗关键数据）：

## 第一部分：专业指标体检单 
（本部分负责客观呈现系统传入的数据，无需过多解释）
- **核心预测**：预测点赞量 **{pred_likes}**（赛道 P50基准为 {theme_p50}，P75爆款基准为 {theme_p75}），综合质量分 **{quality_score}**，超越 **{percentile_rank}** 同类视频。
- **视觉与节奏**：亮度 **{brightness_val}**（{brightness_status}），饱和度 **{saturation_val}**。BPM **{bpm_val}**（{bpm_status}），转场频率 **{cut_freq_val}次/秒**（{cut_status}）。
- **环境与流量**：当前粉丝量 **{follower_count}**，KDE密度估计最佳发布窗口严格限制在：**[{optimal_times_str}]**。

## 第二部分：大白话深度解读 
（本部分负责将上述冷冰冰的数据，翻译成创作者能听懂的“大实话”）
- 任务：根据预测点赞与 P75 基准的差距，结合当前的 BPM、转场频率和亮度状态，用极度口语化、接地气、甚至有些犀利的语言，指出该视频在观众眼中的“实际观感体验”和“核心痛点”。
- 语气要求：就像一位资深导师在看片室直接指导新人。比如：“说句大实话，你的视频目前节奏确实够快（BPM达到 {bpm_val}），但画面偏暗，而且预计点赞离爆款线还差一口气，观众看了可能会觉得‘燃了，但没完全燃透’...”

## 第三部分：实战爆款建议（招招致命） 
（本部分负责给出马上就能上手修改的具体操作方案）
1. **情绪钩子与视听（Hook）**：结合高燃剪辑特性，给出1到2个具体到“建议在第X秒做什么”的强效修改建议（例如：利用音乐气口做切分、高亮反差、关键帧定格等）。严禁“优化前三秒”这种废话。
2. **画面与节奏微调**：针对当前的 {cut_status} 和 {brightness_status} 状态，给出明确的剪辑手法或调色指令。
3. **流量承接策略**：结合现有粉丝数，并严格参照黄金推荐窗口 **[{optimal_times_str}]**，给出具体的发布踩点策略及标题互动建议。

# 输出约束
- 必须严格从“## 第一部分：专业指标体检单 ”开始输出，绝对不要生成任何格式化的开场白或自我介绍。
- 痛点找准，话糙理不糙；建议够具体，能直接落地。
"""

        retries = 0
        backoff = GLOBAL_CONFIG["INITIAL_BACKOFF"]
        
        while retries <= GLOBAL_CONFIG["MAX_RETRIES"]:
            try:
                response = client.chat.completions.create(
                    model=actual_model,
                    messages=[{"role": "user", "content": prompt}]
                )
                advice_text = response.choices[0].message.content
                return advice_text.strip()
            except Exception as e:
                error_str = str(e).lower()
                # HTTP 429 或 503 错误，执行指数退避重试
                if "429" in error_str or "503" in error_str or "rate limit" in error_str or "overload" in error_str:
                    if retries < GLOBAL_CONFIG["MAX_RETRIES"]:
                        time.sleep(backoff)
                        retries += 1
                        backoff *= GLOBAL_CONFIG["BACKOFF_FACTOR"]
                        continue
                return f"*建议生成失败: {str(e)}*"
        
        return "*建议生成失败: 达到最大重试次数*"

