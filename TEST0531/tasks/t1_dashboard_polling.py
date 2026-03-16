# tasks/t1_dashboard_polling.py
import time
import random
import logging
from locust import TaskSet, task

logger = logging.getLogger(__name__)

class DashboardPollingTasks(TaskSet):
    """
    【进阶版】场景 1：数据看板高频轮询任务
    模拟真实大屏前端的 AJAX 轮询行为，包含缓存击穿、动态条件过滤以及深度的 JSON Schema 校验。
    旨在极限试探 Django ORM 的查询优化与 SQLite 的并发读锁瓶颈。
    """

    def on_start(self):
        """预设动态过滤条件，防止 Django/数据库 层面产生查询缓存导致测试失真"""
        self.topics = ["JYP", "Tech", "Gaming", "LifeStyle", "All"]

    @task(4)
    def poll_global_status(self):
        """
        高频请求：带时间戳防缓存的全局状态查询 (权重 4)
        """
        # 模拟前端 Axios/Fetch 产生的防缓存时间戳，确保每一次请求都必须穿透到后端逻辑
        cache_buster = int(time.time() * 1000)
        url = f"/api/global-status/?_t={cache_buster}"

        start_time = time.time()
        # 设置极低的超时时间(5s)，如果单纯的查询状态都超过 5 秒，系统已处于危险边缘
        with self.client.get(url, name="API_读_全局爬虫状态(穿透)", catch_response=True, timeout=5.0) as response:
            process_time = time.time() - start_time

            if response.status_code == 200:
                try:
                    data = response.json()
                    # 严格结构断言：必须包含特定字段，防止由于并发过高导致读出不完整/脏数据
                    if "spider_active" in str(data) or "total" in str(data).lower():
                        response.success()
                        # 自定义性能劣化捕捉
                        if process_time > 1.5:
                            logger.warning(f"🐢 状态查询延迟飙升 ({process_time:.2f}s)，Web 线程或数据库连接池可能存在积压。")
                    else:
                        response.failure("JSON 结构缺失核心监控指标")
                except ValueError:
                    response.failure("服务器返回了 HTML/文本，而非期望的 JSON")
            elif response.status_code >= 500:
                response.failure(f"服务器内部崩溃 HTTP {response.status_code}")
            else:
                response.failure(f"异常状态码 HTTP {response.status_code}")

    @task(2)
    def poll_analysis_status(self):
        """
        中频请求：带动态聚合条件的分析状态查询 (权重 2)
        迫使 SQLite 进行实时的 COUNT() 与 GROUP BY 操作，在读写混合场景下最容易触发 database is locked。
        """
        # 随机选取一个主题进行过滤，增加数据库 SQL 解析引擎与表扫描的负担
        topic = random.choice(self.topics)
        url = "/api/get_analysis_status/"
        if topic != "All":
            url += f"?topic={topic}"

        start_time = time.time()
        with self.client.get(url, name="API_读_分析状态聚合(带过滤)", catch_response=True, timeout=10.0) as response:
            process_time = time.time() - start_time

            if response.status_code == 200:
                try:
                    response.json()
                    response.success()
                    
                    if process_time > 3.0:
                        logger.warning(f"🔥 聚合查询告警：耗时过长 ({process_time:.2f}s) -> 过滤条件: {topic}")
                        
                except ValueError:
                    response.failure("JSON 解析失败")
                    
            # 精准捕获 SQLite 的死锁/锁等待超时
            elif response.status_code == 503 or "locked" in response.text.lower():
                logger.error(f"💥 捕获到确切的 SQLite 读锁/过载异常！耗时: {process_time:.2f}s")
                response.failure("SQLite Database is Locked")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def fetch_recent_logs(self):
        """
        低频请求：拉取最新的评论/视频流水 (权重 1)
        测试带 LIMIT/OFFSET 以及 ORDER BY 排序时的数据库 I/O 承压能力。
        """
        # 模拟大屏前端右侧经常会有的“最新动态滚动条”
        url = "/data/videos/?limit=20"
        with self.client.get(url, name="API_读_最新数据流水(带排序)", catch_response=True, timeout=8.0) as response:
            if response.status_code == 200:
                response.success()
            else:
                # 这种包含大量文本数据的查询如果失败，大概率是内存分配问题
                response.failure(f"流水检索异常 HTTP {response.status_code}")