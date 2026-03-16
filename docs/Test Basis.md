# 测试基准 (Test Basis) - 抖音多模态分析系统

# *1. 项目概述与架构体系*
# *本项目为基于 Django (5.2.9) 及前端可视化库 (ECharts) 构建的抖音全链路多模态分析平台。*
# *核心架构涵盖以下五层：*
# *数据采集层：基于 DrissionPage (无头浏览器抗反爬) 实现视频与评论异步爬取。*
# *数据持久层：基于 UnifiedPersistenceManager 单例封装 sqlite3，提供批量写入及数据库原子级备份。*
# *多模态解析层：利用 OpenCV、Librosa、Torch/Whisper 及 jieba 实现视觉(HSV、亮度)、听觉(BPM、分贝)及文本(ASR 转录)的维度提取，支持 RTX GPU 加速。*
# *AI 数据与爆款预测层：通过 XGBoost/LightGBM 等模型及贝叶斯统计平滑 (ThemeBaselineCalculator) 预测点赞数据，包含基于 LLM (DeepSeek) 的专家运营建议生成。*
# *展示与调度层：Django Views 和 Background Threads (`AIAnalysisWorker`)，支撑 Dashboard 实时聚合监控与接口轮询。*

# *2. 核心数据模型 (Models)*
# *视频元数据表 (Video)：*
# *包含基础信息 (video_id, url, desc, create_time, nickname...)*
# *包含流量指标 (digg_count, comment_count, share_count...)*
# *包含多模态特征 (visual_brightness, visual_saturation, audio_bpm, cut_frequency...)*
# *状态流转机 (analysis_status: 0待分析, 1分析中, 2已完成, -1失败/异常)*
# *评论流水表 (Comment)：*
# *记录评论内容 (content)、清洗文本 (content_clean)*
# *记录外键关联 (video_id)*
# *包含 SnowNLP/Jieba 分析结果 (sentiment_score, sentiment_label)*
# *AI 及用户配置 (AIModelConfig, CreatorConfig)：*
# *管理热更新的模型权重文件。*
# *针对用户级别的 API_KEY（如大模型 Token）进行软隔离。*

# *3. 重点高频测试链路 (Testing P0/P1 Flows)*

## *3.1 爬虫模块测试 (Spider/Crawler Testing)*
# *验证并发下的 DrissionPage 超时降级 (网络阻塞、页面未拉起情况下的 recovery)*
# *验证 `spyder_unified.py` 中的 `.part` 原子文件写入，中断爬虫后不产生脏数据。*
# *验证对于 SPA (单页应用) 路由漂移、验证码等拦截的容错重试。*
# *验证生产者-消费者队列 (`queue.Queue`) 最大容量及内存水位是否可控。*

## *3.2 数据库及事务测试 (Persistence & Transaction Testing)*
# *验证 `data_manager.py` 的事务级批量读写 (bulk_create/bulk_update) 是否在失败时正确回滚。*
# *验证 sqlite3 备份机制 (`DatabaseBackupService`) 在十万级以上查询插入时的表现、以及锁库超时应对。*
# *验证针对重名、相同 ID 的 Idempotency Check (幂等处理)，`update_or_create` 是否防脏数据。*

## *3.3 AI 多模态特征提取测试 (Multimodal Feature Extraction Testing)*
# *验证带有异常/损坏头部的 MP4 是否不会让 `cv2.VideoCapture` 陷入死循环 (`MAX_CONSECUTIVE_FAILURES` 机制)。*
# *验证音频不存在、或者低于阈值时的 Fallback 机制。*
# *验证内存管理，视频流过长或者并发处理时显存耗尽 (OOM) 后的自恢复 (torch.cuda.empty_cache())。*
# *验证并发执行时的多线程全局大模型锁 (`_WHISPER_LOCK`)。*

## *3.4 模型预估及统计验证 (Predict Service Testing)*
# *验证点赞预估服务 (`predict_service.py`) 的 Hot-Reload 机制，在替换 json 结构或者 pkl 文件时服务是否不会挂掉。*
# *验证小样本量 (< 5) 时通过 `ThemeBaselineCalculator` 配置的 Bayesian Smoothing 算法是否正确向 Global Mean 收敛。*
# *验证预测报错情况下的 "Silent Rollback" (静默退回老版本)。*

## *3.5 AI 情感增量计算与后台队列 (Background Worker Testing)*
# *验证常驻进程 `start_ai_worker()` 的 `_ai_worker_instance.is_alive()` 心跳状态。*
# *测试“僵尸任务”回收清理机制：由于宕机导致 `analysis_status=1` 卡死超过 60 秒的能否正确退役并标记 `-1`。*
# *验证大数据量下的增量情感触发器 Chunked Iterator 处理是否会引发 OOM。*

# *4. 第三方服务及环境配置验证*
# *LLM Service 对接：由于依赖外界（API key 鉴权），验证限流(HTTP 429/503)情况下的指数退避重试 (Exponential Backoff)。*
# *CUDA 及 FFmpeg 安装兼容性：依赖外部 C 库或者环境变量的组件缺失时的 Fallback。*

# *5. 前端 API 测试覆盖面*
# *GET `/douyin/api/status/global/` (长轮询，查询爬虫活跃状态的接口幂等与数据一致)。*
# *GET `/douyin/api/status/analysis/` (查询基于 Video 的 0/1/2/-1 状态百分率)。*
# *POST `/douyin/api/predict/` (预测接口及文件上传稳定性验证)。*
# *POST `/douyin/warehouse/import_data/` (批量加载CSV并重置 `analysis_status` 为 0)。*

# *6. 其他注意事项*
# *部署环境：基于 Docker 及 NVIDIA GPU Container Toolkit (docker-compose: deploy.resources.reservations.driver=nvidia)。*
# *依赖核心包：Django(5.2.9), drissionpage, openai, librosa, moviepy, torch。*
