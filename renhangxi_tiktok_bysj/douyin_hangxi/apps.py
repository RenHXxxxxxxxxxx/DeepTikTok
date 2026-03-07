import os
from django.apps import AppConfig

class DouyinHangxiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'renhangxi_tiktok_bysj.douyin_hangxi'
    verbose_name = '抖音多模态分析系统'

    def ready(self):
        # *0. 注册 SQLite WAL 信号处理器 (所有进程均需)*
        try:
            import renhangxi_tiktok_bysj.db_signals  # noqa: F401
        except Exception:
            pass

        # *1. 检查是否在主进程中运行（避免在 runserver 重载进程中重复启动）*
        if os.environ.get('RUN_MAIN') == 'true':
            # *2. 延迟导入以避免 AppRegistryNotReady 错误*
            from .views import start_ai_worker
            if os.environ.get('ML_TRAINING_MODE') != '1':
                try:
                    # *3. 启动后台 AI 分析工作线程*
                    start_ai_worker()
                    print("✅ [System Init] AI Analysis Worker started automatically.")
                except Exception as e:
                    print(f"❌ [System Init] Failed to start worker: {e}")

