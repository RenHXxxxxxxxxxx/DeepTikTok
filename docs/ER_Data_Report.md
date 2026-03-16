# 数据库实体关系 (ER) 与数据拓扑架构主报告

**项目名称**: 多模态短视频智能分析与预测系统 (DeepTikTok)
**文档版本**: V2.0 (High-Fidelity)
**核心受众**: 系统架构设计与 CS 毕业设计 ER 图重构

本报告提供全工作区深度的数据库图谱分析，不仅包含标准的 Django ORM 物理实体，还涵盖了机器学习管线产生的实体、异步调度作业的隐式状态，以及动态计算的主题基准节点。

---

## 1. 核心领域实体 (Core Domain Entities)

### 1.1 视频主库 (`tb_video`)
**业务定位**: 系统的全局事实表 (Fact Table)。承载爬虫抓取的原始数据、用户互动指标、AI后台任务的状态机，以及由 RTX 3060 硬件加速提取的多模态底层特征。

| 字段名称 (Attribute) | 数据类型 (Type) | 约束/属性 (Constraints) | 业务描述与多模态标记 (Description) |
| :--- | :--- | :--- | :--- |
| `video_id` | `CharField(50)` | **PK** | 视频的全局唯一标识（业务主键，来源于抖音）。 |
| `theme_label` | `CharField(100)` | `db_index=True`, `default='默认主题'` | 数据包主题标签，作为微批处理的数据集隔离边界。 |
| `nickname` | `CharField(100)` | - | 视频作者/UP主显示昵称。 |
| `desc` | `TextField` | `null=True`, `blank=True` | 视频原始描述文案。 |
| `create_time` | `DateTimeField` | `null=True`, `blank=True` | 视频在外部平台发布的原始时间戳。 |
| `duration` | `CharField(20)` | `null=True`, `blank=True` | 视频原始时长文本。 |
| `follower_count` | `IntegerField` | `default=0` | 抓取节点时UP主的粉丝存量度量。 |
| `digg_count` | `IntegerField` | `default=0` | 原始点赞量（算法预测及评分模型的核心标的 Target）。 |
| `comment_count` | `IntegerField` | `default=0` | 原始评论量。 |
| `collect_count` | `IntegerField` | `default=0` | 原始收藏量。 |
| `share_count` | `IntegerField` | `default=0` | 原始分享量。 |
| `download_count` | `IntegerField` | `default=0` | 原始下载量。 |
| `video_file` | `FileField` | `null=True`, `blank=True` | 物理素材路由映射，用于定位暂存的 MP4 资产。 |
| **`visual_brightness`** | `FloatField` | `null=True`, `blank=True` | **[Multimodal Feature]** 基于 OpenCV 时序均值计算的全局像素亮度标量。 |
| **`visual_saturation`** | `FloatField` | `null=True`, `blank=True` | **[Multimodal Feature]** 画面色彩饱和度张量均值。 |
| **`audio_bpm`** | `IntegerField` | `null=True`, `blank=True` | **[Multimodal Feature]** Librosa 解析的音频心率基准节拍 (Beats Per Minute)。 |
| **`cut_frequency`** | `FloatField` | `null=True`, `blank=True` | **[Multimodal Feature]** 帧间色谱欧氏距离突变率（剪辑频次）。 |
| `predicted_digg_count`| `IntegerField` | `null=True`, `blank=True` | 竞技场冠军模型输出的预期互动极值。 |
| `actual_vs_predicted_error` | `FloatField` | `null=True`, `blank=True` | 残差偏离度 (Actual vs Prediction Error)。 |
| **`analysis_status`** | `IntegerField` | `default=0`, `db_index=True` | **状态机追踪键**: 0=Pending, 1=Processing, 2=Completed, -1=Failed。 |
| `local_temp_path` | `CharField(255)` | `null=True`, `blank=True` | 异步调度中的挂载临时管道，在处理结束后由 Worker 自动销毁以防 OOM。 |


### 1.2 评论舆情极点 (`tb_comment`)
**业务定位**: 从属视频的维度扩展节点 (Dimension Node)。提供 NLP 自然语言处理的基础语料，构成群体共识与情感拓扑结构。

| 字段名称 (Attribute) | 数据类型 (Type) | 约束/属性 (Constraints) | 业务描述与多模态标记 (Description) |
| :--- | :--- | :--- | :--- |
| `comment_id` | `CharField(50)` | **PK** | 单条评论全局唯一键。 |
| `video_id` | `ForeignKey` | `FK to tb_video`, `on_delete=CASCADE` | 强制维系到特定的底层事实实体。删除主视频物理湮灭下属所有评论。 |
| `theme_label` | `CharField(100)` | `default='默认主题'` | 冗余设计的主题标识（规避部分宽表 Join 开销）。 |
| `nickname` | `CharField(100)` | - | 评论用户的马赛克态昵称。 |
| `content` | `TextField` | - | 带有脏数据的原始网民回复池。 |
| **`content_clean`** | `TextField` | - | 经正则除噪匹配，结合 TF-IDF 抽提后的高密度语义结晶文本。 |
| `create_time` | `DateTimeField` | - | 时间截流发生点，用于生成热度衰变曲线。 |
| `ip_label` | `CharField(50)` | - | 地理围栏分布分析 (Geo-Fencing Analysis)。 |
| `digg_count` | `IntegerField` | `default=0` | 此评论的受认可度权值（NLP权重因子）。 |
| **`sentiment_score`** | `FloatField` | `default=0.5` | **[Multimodal Feature]** SNOWNLP 引擎核算的浮点型情感极性 [0, 1]。 |
| **`sentiment_label`** | `CharField(10)` | - | **[Multimodal Feature]** 基于自适应阈值的文本情绪掩码（非常积极...非常消极）。 |
| `hour` | `IntegerField` | `default=0` | 特征工程抽取的发布时辰切片。 |
| `text_len` | `IntegerField` | `default=0` | 内容密度的物理尺寸标记。 |


---

## 2. 系统管理与 AI 架构实体 (System & AI Operations)

### 2.1 机器学习竞技场主簿 (`tb_ai_model_config`)
**业务定位**: AI 模型的资产管理库，实现热重载 (Hot-Reload) 架构和多权重隔离的版本注册中心。它驱动预测服务对 `version_manifest.json` 进行校验对齐。

| 字段名称 (Attribute) | 数据类型 (Type) | 约束/属性 (Constraints) | 业务描述与多模态标记 (Description) |
| :--- | :--- | :--- | :--- |
| `id` | `AutoField` | **PK** | 系统内部自增标识。 |
| `version_name` | `CharField(50)` | - | 模型演进的语义化标签版本号（例: `v20260306_130252`）。 |
| `model_file` | `FileField` | `upload_to='ai_models/'` | 序列化之后的二进制模型黑盒 (e.g. `LightGBM/XGBoost .pkl`)。 |
| `scaler_file` | `FileField` | `upload_to='ai_models/'` | 与当前特征拓扑锁定的数据标准化矩阵 (StandardScaler)。 |
| `is_active` | `BooleanField` | `default=False` | 单一互斥互斥量 (Mutex Flag)：系统永远只有一把激活的冠军钥匙。 |
| `description` | `TextField` | `blank=True` | 人工投毒防护与备注审计信息。 |
| `create_time` | `DateTimeField` | `auto_now_add=True` | 资产入库与打标时间。 |


### 2.2 创作者全局环境变量 (`tb_creator_config`)
**业务定位**: 独立于核心事实表的创作者偏好沙盒，承接 LLM 对外连接件的独立加密鉴权。

| 字段名称 (Attribute) | 数据类型 (Type) | 约束/属性 (Constraints) | 业务描述与多模态标记 (Description) |
| :--- | :--- | :--- | :--- |
| `id` | `AutoField` | **PK** | Django 默认隐式 ID。 |
| `user_id` | `OneToOneField` | `FK to auth.User`, `on_delete=CASCADE` | 系统租户绑定。级联销毁，生命周期完全隶属于平台管理员。 |
| `llm_api_key` | `CharField(255)` | `blank=True`, `null=True` | 深网/基座模型（如 ErniesBot/DeepSeek）的 API Bearer Token 安全载荷。 |
| `llm_model_name` | `CharField(50)` | `default="ernie-4.0-8k"` | LLM 底座路由分配符。 |

---

## 3. 隐式操作实体 (Implicit Operational Entities)

*(这些实体虽然不在 Django models.py 直观建表，但在代码运行、任务调度、多模态流转中占据了核心 ER 节点地位，必须录入高级 ER 图)*

### 3.1 全局爬虫与异步流水线状态 (Cache Node)
* **主键标识**: Redis/Memcached `global_pipeline_status` Key
* **架构作用**: 用于系统多进程防抖、防互斥锁定的神经中枢。
* **数据结构**: JSON String (`status`: idle/running, `global_phase`, `video: {c, t}`, `ai: {c, t}`)
* **关联属性**: 与 `Video` 表的状态字段互为**隐性控制关系**，指挥 `AIAnalysisWorker` 的消费游标，拦截全局并发的破坏性指令。

### 3.2 竞技场元数据清单 (`version_manifest.json`)
* **主键标识**: 文件系统文件挂载 (Artifacts Directory)
* **架构作用**: 热驱动推理服务的路由网关，ML Pipeline 输送模型的信标器。
* **包含特征拓扑 (Feature Defenses)**: 存放 `current_version`, 跌落防御机制指针 `previous_version`, 参数评估体系 `test_rmse/r2`, 以及从模型系数动态释放出包含衍生特征的 `feature_importances` 有序特征群（例如: `visual_impact`, `sensory_pace`, `content_density` 等）。
* **关联属性**: 为 `tb_ai_model_config` 镜像备份和真实载荷点。`PredictService` 依靠比对该记录校验机制进行热切分。

---

## 4. 实体间连通性与数据依赖度 (Inter-Entity Connectivity)

### 4.1 事实根与舆情延展 (`tb_video` ━ 1:N ━ `tb_comment`)
* **基数映射**: 一对多。一台内容发生器（视频）辐射 N 名多极化受众（评论）。
* **业务连通性**: **强聚合级联 (Strong Aggregation)**。评论是依附于视频的绝对从属衍生品（基于 `on_delete=CASCADE` 机制）。
* **多模态流转**: 
  依赖 Django ORM 视图级的反向计算查询（如 `Avg('comments__sentiment_score')`），它动态形成一张虚拟连结桥梁，将大量非结构化自然语言节点聚合成为短视频多模态体系下的结构化评分。

### 4.2 鉴权实体与工作空间 (`auth.User` ━ 1:1 ━ `tb_creator_config`)
* **基数映射**: 一对一。
* **业务连通性**: **垂直扩展 (Vertical Extension)**。独立切分用户授权信息。保障框架核心鉴权态不因为第三方的 API 修改而出现 Schema 裂变。若外部环境崩坏或用户生命周期终止，它会被完全熔断和级联回收。

### 4.3 推理管道与模型清单依赖 (M:1 Shadow Link)
* **连通性说明**: **影子参照 (Shadow Reference) / 时序冻结**。
  每一次预测行为爆发时，其构建的 DataFrame 骨架直接强依赖于 `version_manifest.json` 所指定的特征列表定义（如 `theme_` 独热编码分支）。如果传入样本（基于 `Video`）缺乏指定的分类维，推理引擎通过其自动注入均值以实现兜底拦截（防止出现缩放器游标漂移错误）。

### 4.4 线程状态与物理资产调度回路
* **连通性说明**: 数据实体 `Video` 通过持有 `local_temp_path` 和 `analysis_status` 位点，与磁盘上的临时文件以及操作系统后台进程紧密捆绑。如果一个视频处在处理状态中途因为异常中断，系统将会检测超时或文件稳定度，并主动调整其事务位，使其在下一次微批扫描重新获得 I/O 锁具，展示了高并发调度里的韧性结构。

---
*End of Report. Ready for ER Diagram generation and implementation planning.*
