# tasks/t2_ai_prediction.py
import io
import os
import time
import random
import logging
from locust import TaskSet, task

logger = logging.getLogger(__name__)

# 全局配置项字典，避免硬编码
GLOBAL_CONFIG = {
    "topics": ["JYP", "Tech", "Gaming", "LifeStyle", "None"],
    "normal_video_timeout": 30.0,
    "corrupt_video_timeout": 15.0,
    "corrupt_video_theme": "Error_Test",
    "endpoint": "/predict/api/",
    "referer_key": "Referer",
    "csrf_header": "X-CSRFToken",
    "csrf_cookie": "csrftoken",
    "normal_task_weight": 4,
    "corrupt_task_weight": 1
}

class AIPredictionTasks(TaskSet):
    """
    *场景 2：多模态 AI 预测与特征提取 (计算与 GPU 密集型)*
    *对应基准测试：*
    *- 3.3 (多模态提取、异常头部兜底、OOM 恢复)*
    *- 3.4 (预估服务静默回滚)*
    *- 4.0 (LLM 429 限流退避)*
    """

    def on_start(self):
        """
        *初始化虚拟用户的测试数据。*
        *为了避免高并发下本地磁盘 I/O 成为瓶颈（掩盖了真实的 GPU/CPU 瓶颈），*
        *我们在内存中动态生成伪造的 MP4 字节流。*
        """
        # 生成一个包含合法 MP4 文件头 (ftyp) 的伪造视频流 (约 500KB)
        mp4_header = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00isommp42"
        self.valid_video_bytes = mp4_header + os.urandom(500 * 1024)
        
        # 生成一个完全损坏的字节流，用于测试 CV2 的 MAX_CONSECUTIVE_FAILURES 机制
        self.corrupt_video_bytes = os.urandom(100 * 1024)
        
        # 预设测试主题池从配置获取
        self.topics = GLOBAL_CONFIG["topics"]

    @task(GLOBAL_CONFIG["normal_task_weight"])
    def predict_normal_video(self):
        """
        *核心高压测试：正常视频特征提取与预测 (权重 4)*
        *模拟用户上传视频并等待多模态处理（耗时较长）。*
        """
        # 每次请求重置字节流指针
        video_file = io.BytesIO(self.valid_video_bytes)
        topic = random.choice(self.topics)
        
        # 根据文档要求从 Cookie 提取 CSRF Token 并构造包含 Referer 的 Header
        csrftoken = self.client.cookies.get(GLOBAL_CONFIG["csrf_cookie"], "")
        headers = {
            GLOBAL_CONFIG["csrf_header"]: csrftoken, 
            GLOBAL_CONFIG["referer_key"]: self.user.host
        }

        files = {
            'video_file': ('load_test_valid.mp4', video_file, 'video/mp4')
        }
        data = {
            'theme_name': topic,
            'follower_count': random.randint(1000, 100000),
            'publish_hour': random.randint(0, 23)
        }

        # 预测接口通常耗时较长，设置超时防止 Locust 线程假死
        start_time = time.time()
        
        # 进行网络请求操作，使用 try-except 捕获可能的网络异常保证稳定性
        try:
            with self.client.post(GLOBAL_CONFIG["endpoint"], 
                                  files=files, 
                                  data=data, 
                                  headers=headers,
                                  name="API_写_AI多模态预测(正常)", 
                                  catch_response=True, 
                                  timeout=GLOBAL_CONFIG["normal_video_timeout"]) as response:
                
                process_time = time.time() - start_time

                # 1. 正常处理完毕
                if response.status_code == 200:
                    try:
                        res_json = response.json()
                        # 严格断言返回结构是否包含预测结果
                        if "predicted_likes" in res_json and "quality_score" in res_json:
                            response.success()
                            if process_time > 10.0:
                                logger.warning(f"⚠️ 预测接口响应迟缓 ({process_time:.2f}s)，可能触发了 _WHISPER_LOCK 排队或 GPU 显存紧张。")
                        else:
                            response.failure("JSON 缺失核心预测字段 (Silent Rollback 可能失败)")
                    except ValueError:
                        response.failure("服务器返回了非 JSON 格式的错误页")

                # 2. 触发了 LLM 的限流 (HTTP 429) 或 GPU 满载 (HTTP 503)
                elif response.status_code in [429, 503]:
                    logger.warning(f"🚫 触发系统限流或排队 (Status {response.status_code})。执行指数退避休眠...")
                    # 在压测逻辑中，正确触发限流也是一种“预期内的成功拦截”
                    response.success() 
                    # 模拟 Exponential Backoff
                    time.sleep(random.uniform(2.0, 5.0)) 

                # 3. 内部服务器错误 (如 OOM 导致进程崩溃)
                elif response.status_code >= 500:
                    logger.error(f"💥 预测接口崩溃 (HTTP {response.status_code})！极大概率发生 GPU OOM 或多线程死锁。")
                    response.failure(f"Server Error {response.status_code}")
                    
                else:
                    response.failure(f"Unexpected HTTP {response.status_code}")
                    
        except Exception as e:
            # 捕获单次请求失效，防止整个测试意外崩溃
            logger.error(f"网络请求引发异常: {str(e)}")

    @task(GLOBAL_CONFIG["corrupt_task_weight"])
    def predict_corrupted_video(self):
        """
        *容错与兜底测试：上传损坏的视频流 (权重 1)*
        *用于验证架构中 3.3 的异常 Fallback 机制。如果系统健壮，应返回 200 并携带默认分值，或返回规范的 400 错误，而不是 500。*
        """
        video_file = io.BytesIO(self.corrupt_video_bytes)
        
        # 根据文档要求从 Cookie 提取 CSRF Token 并构造包含 Referer 的 Header
        csrftoken = self.client.cookies.get(GLOBAL_CONFIG["csrf_cookie"], "")
        headers = {
            GLOBAL_CONFIG["csrf_header"]: csrftoken, 
            GLOBAL_CONFIG["referer_key"]: self.user.host
        }

        files = {
            'video_file': ('load_test_corrupt.mp4', video_file, 'video/mp4')
        }
        data = {'theme_name': GLOBAL_CONFIG["corrupt_video_theme"]}

        # 进行网络请求操作，使用 try-except 捕获可能的网络异常保证稳定性
        try:
            with self.client.post(GLOBAL_CONFIG["endpoint"], 
                                  files=files, 
                                  data=data, 
                                  headers=headers,
                                  name="API_写_AI多模态预测(坏文件)", 
                                  catch_response=True, 
                                  timeout=GLOBAL_CONFIG["corrupt_video_timeout"]) as response:
                
                if response.status_code == 400:
                    # 预期结果 A：系统成功识别了坏文件并拒绝
                    response.success()
                elif response.status_code == 200:
                    try:
                        res_json = response.json()
                        if res_json.get("status") == "fallback":
                            # 预期结果 B：触发了静默退回和异常补偿机制
                            response.success()
                        else:
                            response.failure("坏文件未触发 fallback 状态标志")
                    except ValueError:
                        response.failure("解析异常补偿 JSON 失败")
                elif response.status_code == 500:
                    logger.critical("🔥 严重漏洞：坏文件导致 cv2 或后端逻辑发生 500 崩溃，未被正确 try-except 捕获！")
                    response.failure("Bad file caused HTTP 500")
                    
        except Exception as e:
            # 捕获单次请求失效，防止整个测试意外崩溃
            logger.error(f"网络请求引发异常: {str(e)}")