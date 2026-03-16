# tasks/t3_data_import.py
import io
import csv
import uuid
import time
import random
import logging
from locust import TaskSet, task

logger = logging.getLogger(__name__)

class DataImportTasks(TaskSet):
    """
    【灾难级】场景 3：数据仓库批量导入与 DB 事务高压测试
    针对 SQLite 的 Write Lock、Django 事务回滚以及大文件 I/O 内存消耗进行极限施压。
    """

    def on_start(self):
        """
        初始化：准备一个固定的 video_id，用于后续的幂等性（重复覆盖）测试
        """
        self.idempotent_vid = f"7200000000{random.randint(1000, 9999)}"

    def _get_csrf_headers(self):
        csrftoken = self.client.cookies.get("csrftoken", "")
        return {
            "X-CSRFToken": csrftoken,
            "Referer": self.user.host
        }

    def _generate_dynamic_csv(self, num_rows=500, inject_poison=False, use_fixed_id=False):
        """
        核心造数引擎：在内存中极速生成逼真的抖音 CSV 数据流
        """
        output = io.StringIO()
        writer = csv.writer(output)
        # 写入标准表头
        writer.writerow(['video_id', 'url', 'desc', 'nickname', 'digg_count', 'comment_count'])

        for i in range(num_rows):
            # 是否使用固定的 ID 用于测试 update_or_create
            vid = self.idempotent_vid if use_fixed_id else str(uuid.uuid4().int)[:18]
            
            # 【毒药注入机制】在第 50 行注入致命的数据库完整性错误（把必需的整型改成超长字符串或空）
            if inject_poison and i == 50:
                writer.writerow([vid, "http://poison.url", "Poison Row", "Hacker", "NOT_A_NUMBER", ""])
                continue
                
            writer.writerow([
                vid,
                f"https://www.douyin.com/video/{vid}",
                f"Auto generated load test description {i}",
                f"Creator_{i}",
                random.randint(100, 100000),
                random.randint(10, 5000)
            ])
            
        return output.getvalue().encode('utf-8')

    @task(3)
    def bulk_import_normal(self):
        """
        高压写请求：正常大批量 CSV 导入 (权重 3)
        测试 UnifiedPersistenceManager 的 bulk_create 性能及 SQLite 锁表阈值。
        """
        # 动态生成 1000 行 CSV 字节流，约占用几百 KB 内存
        csv_bytes = self._generate_dynamic_csv(num_rows=1000)
        files = {'file': ('massive_import.csv', io.BytesIO(csv_bytes), 'text/csv')}
        headers = self._get_csrf_headers()

        start_time = time.time()
        # 写入操作极容易锁库，设置较长的超时时间
        with self.client.post("/api/import_data/", 
                              files=files, 
                              headers=headers,
                              name="API_写_CSV全量导入(1000行)", 
                              catch_response=True, 
                              timeout=45.0) as response:
            
            process_time = time.time() - start_time

            if response.status_code == 200:
                response.success()
                if process_time > 15.0:
                    logger.warning(f"⚠️ 批量插入耗时严重 ({process_time:.2f}s)。SQLite 正在死扛，可能触发了 DatabaseBackupService 备份。")
            elif response.status_code == 503 or "locked" in response.text.lower():
                logger.error(f"💥 SQLite 写锁崩溃 (Write Lock)! 并发写入互相踩踏，耗时 {process_time:.2f}s")
                response.failure("SQLite Database is Locked (Write)")
            elif response.status_code == 429:
                response.success() # 如果做了导入限流，算正常防御
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def test_idempotency(self):
        """
        逻辑校验请求：幂等性测试 (权重 1)
        故意发送完全一样的 video_id，验证底层是否正确使用了 update_or_create，而不是报 500 IntegrityError。
        """
        csv_bytes = self._generate_dynamic_csv(num_rows=50, use_fixed_id=True)
        files = {'file': ('idempotent_test.csv', io.BytesIO(csv_bytes), 'text/csv')}
        headers = self._get_csrf_headers()

        with self.client.post("/api/import_data/", 
                              files=files, 
                              headers=headers,
                              name="API_写_幂等性覆盖测试", 
                              catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 500 and "UNIQUE constraint failed" in response.text:
                logger.critical("🔥 严重漏洞：幂等性失效！由于重复 video_id 导致数据库报 UNIQUE constraint failed 错误。")
                response.failure("Idempotency Failed: Unique Constraint")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def test_transaction_rollback(self):
        """
        破坏性请求：事务回滚与脏数据防范测试 (权重 1)
        包含毒药数据的 CSV。预期 Django 能够捕获异常，并且执行 Rollback，返回 400 或 200(带错误提示)，绝不应该是 500。
        """
        csv_bytes = self._generate_dynamic_csv(num_rows=200, inject_poison=True)
        files = {'file': ('poison_test.csv', io.BytesIO(csv_bytes), 'text/csv')}
        headers = self._get_csrf_headers()

        with self.client.post("/api/import_data/", 
                              files=files, 
                              headers=headers,
                              name="API_写_毒药事务回滚测试", 
                              catch_response=True) as response:
            
            # 完美的系统应该拦截毒药并返回 400 Bad Request
            if response.status_code == 400:
                response.success()
            # 或者系统宽容处理，忽略毒药行，返回 200
            elif response.status_code == 200:
                logger.info("系统吸收了毒药数据（可能是跳过了错误行，也可能是整体事务回滚，需要校验数据库）。")
                response.success()
            # 如果是 500，说明 `bulk_create` 报错没有被 try-except 包裹，直接炸到了前端
            elif response.status_code == 500:
                logger.critical("🔥 严重漏洞：毒药数据击穿了事务防护，引发 HTTP 500 崩溃！可能产生脏数据。")
                response.failure("Transaction Rollback Failed (HTTP 500)")
            else:
                response.failure(f"Unexpected HTTP {response.status_code}")