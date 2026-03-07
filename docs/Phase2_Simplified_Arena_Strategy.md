# Phase 2: Simplified "3+1" Champion-Challenger Arena Strategy

## 1. 预设权重的作用（"出厂默认设置"）
即使系统具备根据最新抓取数据进行高频重新训练的能力，保留一套高质量的“预设模型权重”（Factory Default）依然是架构设计的基石。
- **解决“冷启动”问题（Cold Start）：** 在系统首次部署或处于全新环境中缺乏足够有效历史数据时，预设模型能够立即提供基准的推理能力，确保业务流程不中断。
- **灾难性重训的“安全回退”（Safety Fallback）：** 爬虫获取的数据往往伴随不可控的噪声、异常值甚至数据漂移。如果最新一次的训练因为脏数据而导致模型性能崩塌，预设权重和验证机制可以作为最后一道防线，拒绝劣质模型的上线，并在极端异常时安全回滚到稳定的出厂预设状态，防止“灾难性遗忘”（Catastrophic Forgetting）。

## 2. "3+1" 核心执行流 (取代原本臃肿的6模型循环)
为了避免无限循环比较和基准测试疲劳，`train_master_arena.py` 的训练流水线将精简为高效的“3（内部挑战者）+ 1（现任卫冕冠军）”逻辑：

- **Step 1 (内部选拔 - Internal Selection):**
  在最新聚合的训练数据集上，同时并行训练 3 种核心算法（Random Forest, LightGBM, XGBoost）。训练完成后，通过交叉验证（Cross-Validation）或保留集（Hold-out）对三者进行严格评分与排名。
- **Step 2 (加冕挑战者 - Crowning the Challenger):**
  遵循内存与磁盘的极致优化原则，仅保留 3 个模型中综合指标（如 RMSE, R2）排名第一（Top 1）的模型作为唯一的“挑战者（Challenger）”。立即在内存中销毁并丢弃落败的 2 个模型。
- **Step 3 (守门人对决 - The Gatekeeper Showdown):**
  解析模型控制清单 `version_manifest.json`，从磁盘中加载当前的“卫冕冠军（Champion）”模型。如果是系统首次运行，此模型即为前述引入的“出厂预设权重”。
- **Step 4 (实战部署 - Deployment):**
  在独立、未经污染的全局验证集上，让“挑战者”与“卫冕冠军”进行正面对决。如果“挑战者”的性能指标（例如误差降低幅度）超越预设的阈值（Margin），则执行原子化操作（Atomic Overwrite）更新清单文件，成功上位；若挑战者败北，则直接终止部署流程，继续由现任冠军服役。

## 3. 清单与遥测优化 (Manifest & Telemetry Optimization)
基于此简化策略，模型元数据清单（如 `arena_master_manifest.json`）将得到实质性的精简和重构。
- **单一真理来源：** 该文件只需追踪记录唯一在役的 Reigning Champion 及其核心元数据，无需维护其余被淘汰模型的冗余信息，大幅降低了版本控制的复杂度。
- **赋能推理服务：** 这一优化直接使得下游推理层（`predict_service.py`）的加载逻辑变成单线、确定性操作。不仅缩短了模型热加载的耗时，更能确保线上的“零宕机回滚机制（Zero-Downtime Rollback）”随时精准回退到记录的唯一且最安全的底座模型。
