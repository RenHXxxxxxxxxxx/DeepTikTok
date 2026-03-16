# locustfile.py
import logging
import time
from locust import HttpUser, between, events, tag
from common.auth import login_user

# 导入即将编写的任务场景（先写在这，稍后我们去实现它们）
from tasks.t1_dashboard_polling import DashboardPollingTasks
from tasks.t2_ai_prediction import AIPredictionTasks
from tasks.t3_data_import import DataImportTasks

# 配置日志：不仅看控制台，还要区分级别
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# 1. 全局监控与预检钩子 (Environment Hooks)
# ==========================================
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """测试启动前的预检，比如检查 Django 后端是否存活"""
    logger.info("🚀 [PRE-FLIGHT] 正在检测本地后端 http://127.0.0.1:8000 是否可达...")
    try:
        # 这里可以使用 requests 做一个简单的探活
        pass 
    except Exception:
        logger.error("❌ 后端服务未启动，压测终止！")
        environment.runner.quit()

# ==========================================
# 2. 仿真用户基类 (Base Analytics User)
# ==========================================
class BaseDouyinUser(HttpUser):
    """
    抽象基类：处理公共的鉴权逻辑和异常保护
    """
    abstract = True  # 声明为抽象类，Locust 不会直接实例化它
    wait_time = between(1.5, 5.0)  # 更加仿真的随机思考时间

    def on_start(self):
        """每个用户孵化时，必须先通过身份门禁"""
        self.is_authenticated = login_user(self, username="renhangxi", password="bang0531")
        if not self.is_authenticated:
            logger.critical(f"用户 {self} 登录失败，该线程将持续空转。")

# ==========================================
# 3. 细分业务场景类 (Differentiated Workloads)
# ==========================================

class ViewerUser(BaseDouyinUser):
    """
    场景 A：普通观察者（高频、轻负载）
    权重：10
    行为：主要在 Dashboard 闲逛，查看实时统计。
    """
    weight = 10 
    tasks = [DashboardPollingTasks]

class HeavyAnalyzerUser(BaseDouyinUser):
    """
    场景 B：资深分析师（低频、高负载、计算密集）
    权重：2
    行为：上传视频、调用 XGBoost 预测、触发 DeepSeek AI 诊断。
    """
    weight = 2
    tasks = [AIPredictionTasks]

class AdminDataMaintainer(BaseDouyinUser):
    """
    场景 C：系统管理员（极低频、I/O 密集、锁表风险高）
    权重：1
    行为：执行大批量 CSV 导入，触发数据库备份。
    """
    weight = 1
    tasks = [DataImportTasks]

# ==========================================
# 4. 自定义性能指标监控 (Custom Metrics)
# ==========================================
@events.request.add_listener
def my_request_handler(request_type, name, response_time, response_length, response, exception, **kwargs):
    """
    全局请求监听：
    如果某个接口（如 AI 预测）响应时间超过 5 秒，自动记录为性能瓶颈。
    """
    if response_time > 5000:
        logger.warning(f"⚠️ 性能预警：接口 {name} 耗时过长 ({response_time}ms)")

    if exception:
        logger.error(f"💥 异常捕获：接口 {name} 发生错误 -> {exception}")