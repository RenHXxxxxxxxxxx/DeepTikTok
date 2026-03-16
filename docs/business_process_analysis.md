| 页面/功能真实名称 (UI/urls.name) | 对应代码函数 (View/Task) | 所属模块 | 操作角色 | 流程位置 (起点/处理/终点) |
| :--- | :--- | :--- | :--- | :--- |
| 启动综合爬虫任务 (`launch_spider_api`) | `views.launch_spider_api` / `_run_spider_background` | 市场调研模块 | 创作者 | 流程起点 |
| 启动单条评论爬虫 (`launch_comment_only_api`) | `views.launch_comment_only_api` | 市场调研模块 | 创作者 | 流程起点 |
| 数据精炼与统一入库 (`import_data_api`) | `views.run_clean_data_api` / `import_data_service` | 市场调研模块 | 系统后台 | 处理 |
| 主题数据强制重算修复 (`recalculate_sentiment_api`) | `views.recalculate_sentiment_api` / `trigger_sentiment_analysis` | 市场调研模块 | 管理员 | 处理 |
| 爬虫队列状态监控 (`get_spider_status_api`) | `views.get_spider_status_api` | 市场调研模块 | 系统后台 | 处理 |
| AI分析后台异步处理线程 | `views.AIAnalysisWorker.run` / `_process_single_video` | 视频诊断模块 | 系统后台 | 处理 |
| 获取AI分析实时状态 (`analysis_status_api`) | `views.get_analysis_status_api` | 视频诊断模块 | 系统后台 | 处理 |
| 爆款预测交互界面 (`predict_page`) | `views.predict_page` | 视频诊断模块 | 创作者 | 流程起点 |
| 赛道基准对比预测特征提取 (`predict_api`) | `views.predict_api` / `DiggPredictionService.predict_digg_count` | 视频诊断模块 | 系统后台 | 处理 |
| DeepSeek 专家运营建议生成 (`predict_api`系统内置调用) | `LLMService.generate_advice` (嵌于 `views.predict_api`) | AI 建议生成模块 | 系统后台 | 终点 |
| 大模型API Token与偏好接入 (`profile`) | `views.profile_view` | AI 建议生成模块 | 创作者 | 流程起点 |
| 系统全链路运行看门狗 (`global_status`) | `views.get_global_status` | 后台运维模块 | 系统后台 | 监控 |
| 物理数据彻底擦除 (`delete_theme`) | `views.delete_theme` | 后台运维模块 | 管理员 | 终点 |
| 预测模型热更新重训 (`retrain_model_api`) | `views.retrain_model_api` | 后台运维模块 | 管理员 | 处理 |
| 数据资产大盘宏观指标 (`dashboard`) | `views.dashboard` | 市场调研模块 | 创作者 | 终点 |
| 视觉化四维剖析图表 (`chart_user`, `chart_content`, `chart_sentiment`) | `views.chart_user`, `views.chart_content`, `views.chart_sentiment` | 视频诊断模块 | 创作者 | 终点 |
| 资产数据分页流 (`video_list`, `comment_list`) | `views.video_list`, `views.comment_list` | 市场调研模块 | 创作者 | 终点 |
| 数据仓库物理文件视图 (`data_warehouse`) | `views.data_warehouse` | 后台运维模块 | 管理员/创作者 | 终点 |
