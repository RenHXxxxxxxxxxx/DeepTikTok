# 视频分析系统数据设计审计报告

## 第一部分：实体词典

| 实体名称 | 代码类名 | 功能描述 | 核心职责 |
| :--- | :--- | :--- | :--- |
| **视频数据实体** | `Video` | 记录抓取的视频基础元数据、多模态特征以及分析状态。 | 整个视频处理流程的核心枢纽，连接外部爬虫数据与内部特征加工流水线。 |
| **评论舆情实体** | `Comment` | 依附于视频记录单条评论及情感分析结果。 | 为视频挖掘深度用户共鸣度和情感极性提供结构化舆情数据。 |
| **预测模型配置实体** | `AIModelConfig` | 高维数据预测模型（如 XGBoost）及其预处理器的版本控制。 | 将核心机器学习资产实现热拔插与回溯，避免硬编码引入。 |
| **创作者配置实体** | `CreatorConfig` | 记录终端用户的业务配置（如大语言模型凭证）。 | 满足大模型 API 动态调用需求，并提供一对一用户定制化能力。 |
| **系统用户实体** | `User` (Django built-in) | Django 内置的用户及权限管理底座。 | 提供核心的 RBAC 权限控制与身份校验基石。 |

## 第二部分：核心表结构明细表

### 1. 视频数据表 (`tb_video`)
*存储系统中的基础多模态数据与流水线处理状态*

| 字段名称 | 物理类型 | 映射组件 / 含义 | 备注 / 约束条件 |
| :--- | :--- | :--- | :--- |
| `video_id` | `CharField(50)` | 视频唯一编号 | **主键** (Primary Key) |
| `theme_label` | `CharField(100)` | 归属主题分类 (Partition) | 默认='默认主题', 存在 DB Index 加速检索 |
| `nickname` | `CharField(100)` | 作者昵称 | `null=False` |
| `desc` | `TextField` | 视频文字描述 | 可为空 (`blank=True, null=True`) |
| `duration` | `CharField(20)` | 视频时长字面量 | 后期通过清洗引擎转化为秒级浮点数 |
| `create_time` | `DateTimeField` | 抖音视频发布时间 | - |
| `follower_count` | `IntegerField` | 创作者粉丝数 | 默认=0 |
| `digg_count` | `IntegerField` | 点赞基数 | 默认=0 (后用作回归模型 Target) |
| `comment_count` | `IntegerField` | 评论总数 | 默认=0 |
| `collect_count` | `IntegerField` | 收藏数 | 默认=0 |
| `share_count` | `IntegerField` | 分享数 | 默认=0 |
| `download_count` | `IntegerField` | 下载数 | 默认=0 |
| `visual_brightness` | `FloatField` | [多模态] 画面基础亮度 | Web 3060 CUDA 并行提取特征 |
| `visual_saturation` | `FloatField` | [多模态] 画面极大极小极差 (HSV 模拟) | Web 3060 CUDA 像素级特征 |
| `audio_bpm` | `IntegerField` | [多模态] 音频节拍 (BPM) | librosa 强制依赖 FFmpeg 提取 |
| `cut_frequency` | `FloatField` | [多模态] OpenCV 直方图转场频率 | 结合 `duration` 转化为均频系数 |
| `video_file` | `FileField` | [多模态] 原始物理素材路径 | 挂载于 `videos/%Y/%m/%d/` |
| `predicted_digg_count` | `IntegerField` | 服务端算法推演分（爆款潜质） | - |
| `actual_vs_predicted_error` | `FloatField` | 系统推断阻力系数 (误差率) | 评估泛化能力的后处理探针 |
| `analysis_status` | `IntegerField` | 后台异步队列处理状态位 | **Index=True**. 详见状态机审计章节 |
| `local_temp_path` | `CharField(255)` | OOM 容灾或隔离存储挂载点 | 可为空 |

### 2. 评论舆情表 (`tb_comment`)
*围绕具体的视频构建的二级属性结构*

| 字段名称 | 物理类型 | 映射组件 / 含义 | 备注 / 约束条件 |
| :--- | :--- | :--- | :--- |
| `comment_id` | `CharField(50)` | 评论节点 ID | **主键** (Primary Key) |
| `video` | `ForeignKey` | 挂载依附的源视频 | **外键**关联 `tb_video(video_id)`, `on_delete=CASCADE` |
| `theme_label` | `CharField(100)` | 数据隔离域 | 冗余设计以支持跨域宽表计算 |
| `nickname` | `CharField(100)` | 评论发布者标识 | - |
| `content` | `TextField` | 原文 (Raw Corpus) | - |
| `content_clean` | `TextField` | 脱敏与清洗后文本 | 剔除超链接及非语义符号 |
| `create_time` | `DateTimeField` | 时间轴锚点 | - |
| `ip_label` | `CharField(50)` | LBS 坐标锚点 | - |
| `digg_count` | `IntegerField` | 反馈验证强度 | - |
| `sentiment_score` | `FloatField` | 语义模型正负判别分 | 默认 0.5 (中性) |
| `sentiment_label` | `CharField(10)` | 极性归档（如：积极/消极等） | 字符串离散表示 |
| `hour` | `IntegerField` | 时间窗口 (0-23) | 高频数据特征 |
| `text_len` | `IntegerField` | 信息熵厚度 (文本占量) | 预计算结果缓冲 |

### 3. AI模型配置管理表 (`tb_ai_model_config`)
*控制 XGBoost 与 Scaler 热加载逻辑的软状态机*

| 字段名称 | 物理类型 | 含义 | 备注 / 约束条件 |
| :--- | :--- | :--- | :--- |
| `id` | `AutoField` | 架构自增 ID | **主键** (默认隐藏字段) |
| `version_name` | `CharField(50)` | 版本指纹 | `null=False` |
| `model_file` | `FileField` | pkl 文件介质指针 | 用于动态实例重启 |
| `scaler_file` | `FileField` | 数据归一化介质指针 | `StandardScaler` 对象持久化 |
| `is_active` | `BooleanField` | 多路复用流量开关 | 是否投入主干推断核心 (`default=False`) |
| `description` | `TextField` | 版本调优审计日志 | 业务可读 |
| `create_time` | `DateTimeField` | 模型定型封版锚点 | `auto_now_add=True` |

### 4. 创作者配置表 (`tb_creator_config`)
*负责用户的扩展配置，如大模型 API 信息*

| 字段名称 | 物理类型 | 含义 | 备注 / 约束条件 |
| :--- | :--- | :--- | :--- |
| `id` | `AutoField` | 配置 ID | **主键** |
| `user` | `OneToOneField` | 所属操作者 | **外键** (1对1) -> Django `User`, `CASCADE` |
| `llm_api_key` | `CharField(255)` | 外部 API 交互凭证 | *强烈建议生产环境进行强加密* |
| `llm_model_name` | `CharField(50)` | 文本语义生成底座引擎 | 默认="ernie-4.0-8k" |

## 第三部分：逻辑关系矩阵

| 主体实体 | 关系类型 | 依赖实体 | 关联约束定义 | 业务数据一致性保障机制 |
| :--- | :--- | :--- | :--- | --- |
| `Video` | **1 : N** | `Comment` | `video_id` -> `tb_comment.video_id` | **级联删除 (`on_delete=CASCADE`)**。父视频实例毁灭，归属的评论流进行硬清理，保证无孤儿死锁点。相关代码中利用 `bulk_create` 与 `transaction.atomic()` 进行原子注入。 |
| `User` | **1 : 1** | `CreatorConfig` | `id` -> `tb_creator_config.user_id` | **级联删除 (`on_delete=CASCADE`)**。创作者账号生命周期完结则回收专属的软配置及密保。 |
| `Video` | **1 : 1** (逻辑挂载) | 外部预测服务 `predict_service` | 依赖实体无表，与特征空间存在软限制。 | 依靠 `predict_service.py` 动态构建 29 维度输入矩阵（由硬编码基线字段与动态 One-Hot `theme_label` 并行组合）。 |

## 第四部分：状态机与多模态审计专项

### 1. 多模态存储专项 (29维量化特征对齐)

在表结构 `Video` 层，系统存在针对多模态物理存储的静态特征字段（如 `visual_brightness`, `visual_saturation`, `audio_bpm`, `cut_frequency`），但最终向模型交托的输入层为动态合成的 29 维矩阵，这种结构在代码底层执行了映射与对齐方案（详情追踪自 `predict_service.py` 及流水线脚本）：

*   **13 维静态基础集（含衍生特征）：**
    通过 `Video` 表直接抽取并在 `predict_service.py` 中构造衍生列池：
    *   源维度：`follower_count`, `publish_hour`, `duration_sec`, `avg_sentiment` 以及 4 大多模态属性 (`visual_brightness`, `visual_saturation`, `cut_frequency`, `audio_bpm`)。
    *   推演衍生：`visual_impact` (亮度*饱和度), `sensory_pace` (心跳*剪频), `sentiment_intensity` (绝对极化距), `audio_visual_energy` (动测), `content_density` (频率浓度)。
*   **16 维动态 Theme One-Hot 集合：**
    系统在特征交汇时，读取表属的 `theme_label`，将已知的主题池向量化铺平（OHE 化）。
*   **最终对齐（Physical Consistency Matrix）：**
    利用缓存版 `Scaler` 持有的 `feature_names_in_` 及 XGBoost Booster 的特征序。系统根据静态模型 `Video` 数据、计算得来的衍生特征字典和 `theme_label` 打平的向量层，通过 `X_final = pd.DataFrame(0.0, index=[0], columns=ordered_feature_names)` 执行最终物理缝合并输入推测节点。

### 2. 状态机审计 (Task Status Audit)

异步流水线状态被硬编码储存在 `Video` 表中的 `analysis_status` 构建的精细状态机（Integer 标识类型）。在 `data_manager.py` 与执行循环间维持运转：

*   `0`: **Pending (待处理)**。任务进入队尾初始栈结构，等待调度循环的 `get_pending_videos` API 抽取并发放至计算核心。
*   `1`: **Processing (处理中)**。任务正占用硬件资源，已被工作节点使用 `update_video_analysis_status` 原子锁定。
*   `2`: **Completed (验证完毕)**。多模态推演链顺畅完成，回写至 SQLite 层，生命周期结束。
*   `-1`: **Failed (异常或损坏)**。如果并发队列出现不可挽回系统阻断点（如 OOM, CUDA 显存溢出，Timeout 捕获，或视频流彻底损坏），任务标记退缩降级不进行无效的无尽循环。

## 第五部分：E-R 图绘制建议

依据审计结果，绘制 E-R (Entity-Relationship) 图时敬请参照以下规范以达到严格的学术绘图准则：

1.  **实体形状：** 将 `Video`（视频数据）、`Comment`（评论舆情）、`AIModelConfig`（AI 配置）、`CreatorConfig`（创作者配置）与 `User` 设为矩形核心主实体。
2.  **属性结构：** 在每个实体下用椭圆形引出主要字段，并对被定义的 `Primary Key` （如 `video_id`, `comment_id` 等）加下划线标注。
3.  **连接线关系：**
    *   用菱形表述 "拥有/挂载/包含" 逻辑。
    *   连接 `Video` 与 `Comment`，引出 **(1, N)** 基数标志（一个视频衍生多路评论流分析）。
    *   连接 `User` 与 `CreatorConfig`，引出 **(1, 1)** 核心基数标志。
4.  **隐形弱实体提示：** 请将多模态特征集合 (如 `visual_brightness`, `visual_saturation` 等) 用高亮标出作为 `Video` 的紧密原子属性层，不能单独拆除关联表。可于下方或附录处注明此基底最后经由管道变形为多维推演矩阵输入引擎的过程。
