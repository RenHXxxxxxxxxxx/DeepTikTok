# 抖音数据采集与智能分析系统 (TikTok Data Scraping & Analysis System)

> *基于 Django 与 DrissionPage 的抖音全链路数据采集、转存与 AI 智能分析可视化平台。*

## 核心功能特性

- **强干扰绕过爬虫**: 基于 `DrissionPage` 封装的浏览器沙盒操作引擎，实现底层请求穿透，完美绕过抖音 WAF 302 重定向并支持批量数据抓取。
- **AI 大模型流式处理**: 接入 DeepSeek 大语言模型 (`openai` 标准库)，对视频标题、互动标签进行深度语义理解与二次加工，支持高并发异常捕获。
- **实时监控仪表盘 (Dashboard)**: 完全基于 Django 构建，实时呈现 AI 工作流吞吐量、页面数据抓取速率及 Celery/Redis 后端排队状态。
- **离线安全导入**: 配置驱动型设计 (Configuration-Driven)，支持安全地将本地预存的 JSON/CSV 数据清洗后导入至数据库处理链路。
- **隔离的运行沙盒**: 彻底分离运行时缓存、敏感依赖密钥、视频切片及爬虫原始日志 (已通过 [.gitignore](cci:7://file:///d:/renhangxi_tiktok_bysj/.gitignore:0:0-0:0) 规则屏蔽上传)，保证开源库轻量与纯净。

## 关键技术栈

- **Web 架构体系**: Django 5.2, Django SimpleUI
- **异步与缓存**: Celery 5.3, Redis 5.0
- **浏览器与爬虫**: DrissionPage 4.1.1, DownloadKit 2.0
- **人工智能与大模型**: OpenAI SDK (DeepSeek Provider), PyTorch 2.5, LightGBM
- **数据分析与展现**: Pandas 2.3, Pyecharts 2.0, matplotlib

## 代码库拓扑结构

```text
.
├── crawler/                # *基于 DrissionPage 的爬虫核心调度模块*
├── docs/                   # *平台接口说明及操作指引库*
├── ml_pipeline/            # *核心机器学习及数据清洗加工流水线*
├── renhangxi_tiktok_bysj/  # *Django 主配置目录 (包含 settings.py 等)*
├── scripts/                # *辅助分析与脏数据打捞运维脚本*
├── services/               # *全局抽象服务层 (如 LLMService 调用封装)*
├── static/                 # *前端静态依赖文件目录*
├── templates/              # *用户管理界面及大屏视图模板*
├── tests/                  # *系统压力测试与异常覆盖用例*
├── manage.py               # *Django 服务启动网关执行文件*
└── requirements.txt        # *全量环境依赖绑定文件*
