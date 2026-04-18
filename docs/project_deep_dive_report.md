# 1. Executive Summary
This project is a real, code-backed Django system for Douyin data crawling, data warehousing, multimodal feature extraction, sentiment analytics, virality prediction, and LLM-based diagnosis. It is not only interface code: key business logic is implemented in backend modules such as `renhangxi_tiktok_bysj/douyin_hangxi/views.py`, `crawler/spyder_unified.py`, `services/predict_service.py`, `services/data_manager.py`, and `ml_pipeline/train_master_arena.py`.

At runtime, the architecture is a Django monolith plus in-process asynchronous threads (not Celery runtime in active path):
- crawler background threads launched by APIs
- an `AIAnalysisWorker` daemon that consumes pending video analysis tasks
- cache-based progress/status communication to frontend
- model inference and LLM diagnosis on prediction requests

As of local inspection on 2026-03-22 (DB evidence):
- `Video`: 2147 rows
- `Comment`: 23499 rows
- 20 themes
- `analysis_status`: 929 completed (`2`), 1218 failed (`-1`)

Defense conclusion: implementation depth is real and demonstrable, but there are material risks around consistency (doc/runtime mismatch, mixed legacy paths, and partial data quality/analysis completion across themes). A strong defense strategy should acknowledge these clearly and explain your engineering trade-offs.

# 2. Repository Structure Overview
## 2.1 High-value directories and actual roles
| Path | Actual role in this project |
|---|---|
| `renhangxi_tiktok_bysj/settings.py` | Django runtime config (SQLite, timezone, app registration) |
| `renhangxi_tiktok_bysj/urls.py` | Root router, includes app routes |
| `renhangxi_tiktok_bysj/douyin_hangxi/urls.py` | Main page/API routing (dashboard, crawler, import, prediction, retrain, status) |
| `renhangxi_tiktok_bysj/douyin_hangxi/models.py` | Core entities: `Video`, `Comment`, `AIModelConfig`, `CreatorConfig` |
| `renhangxi_tiktok_bysj/douyin_hangxi/views.py` | Main business orchestration (very large, central control file) |
| `renhangxi_tiktok_bysj/douyin_hangxi/utils/video_analyzer.py` | Multimodal extraction (OpenCV + Librosa + Whisper + Torch) |
| `renhangxi_tiktok_bysj/douyin_hangxi/utils/llm_service.py` | DeepSeek API calls via OpenAI SDK |
| `services/predict_service.py` | Model loading, feature building, prediction, fallback, baseline correction |
| `services/data_manager.py` | Unified persistence manager with idempotent write patterns |
| `crawler/spyder_unified.py` | Browser crawler + downloader + comment crawler + async DB queue |
| `ml_pipeline/train_master_arena.py` | Active retrain script (champion-challenger) |
| `ml_pipeline/theme_baseline_engine.py` | Bayesian-smoothed theme baseline statistics |
| `templates/*` | UI pages connected to real APIs |
| `TEST0531/*` | Locust pressure/concurrency test scripts |
| `tests/*` | Basic unit/integration-lite tests |
| `artifacts/` | Active model artifacts and version manifest used by runtime inference |

## 2.2 Important structural facts
- Active Django app namespace is `renhangxi_tiktok_bysj.douyin_hangxi` (`settings.py:24-33`, `apps.py:6`).
- Core runtime logic is split across app-level views and root-level `services/`, `crawler/`, `ml_pipeline/`.
- There are duplicate/legacy-style files and paths (for example root-level `douyin_hangxi/utils/video_analyzer.py`, and auxiliary scripts importing `data.spyder_unified`), which increases maintenance and defense questioning risk.

# 3. Real System Architecture
## 3.1 Runtime organization (what actually runs)
1. Django web process handles page/API requests (`views.py`).
2. Crawling tasks are started by API and executed in background threads (`launch_spider_api` -> `_run_spider_background`, `launch_comment_only_api` -> `_run_comment_only_background`).
3. Crawling engine (`crawler/spyder_unified.py`) performs video/comment acquisition and pushes persistence tasks.
4. Persistence layer (`services/data_manager.py`) writes videos/comments with idempotency and FK checks.
5. AI analysis worker thread (`AIAnalysisWorker` in `views.py`) continuously consumes videos with `analysis_status=0` and fills multimodal features.
6. Prediction endpoint (`predict_api`) runs on-demand extraction + model inference + LLM advice.
7. Retrain endpoint (`retrain_model_api`) triggers `ml_pipeline/train_master_arena.py` in subprocess and hot-resets inference singleton.

## 3.2 Frontend-backend coupling
- Warehouse page (`templates/data/warehouse.html`) calls import/crawler/switch/delete APIs.
- Dashboard page (`templates/dashboard.html`) polls `/api/get_analysis_status/` and can trigger `/api/retrain_model/`.
- Prediction page (`templates/prediction/dashboard.html`) uploads video to `/predict/api/` and renders result + advice.

## 3.3 Database and async roles
- Database is SQLite (`settings.py:65-73`) with WAL enabled on connection (`db_signals.py:3-14`).
- Async strategy is native threading + cache keys (not full queue middleware in active path).
- Status/state machine core uses `Video.analysis_status` (`models.py:42`) and cache keys like `global_pipeline_status`, `spider_progress`, `ACTIVE_TASK` (`views.py` around lines 1497+, 1538+, 1900+).

## 3.4 External services
- Douyin website/API interaction via DrissionPage + requests (`crawler/spyder_unified.py`).
- DeepSeek via OpenAI client base URL `https://api.deepseek.com` (`llm_service.py:11-13`, `llm_service.py:41`).

# 4. End-to-End Data Flow
## 4.1 Crawl-to-dashboard pipeline
1. User starts crawl from UI -> `/api/launch_spider/` (`urls.py:13`, `views.py:2026`).
2. Backend parses keywords and starts background thread (`views.py:2092-2097`).
3. Thread calls `run_spider_service()` (`views.py:1616-1618`, `crawler/spyder_unified.py:1437`).
4. Crawler collects video metadata, downloads video to `media/pending_videos`, writes CSV, writes DB video records (`spyder_unified.py:369-517`, `601+`, `863-865`).
5. Crawler traverses video pages for comments, writes CSV, asynchronously pushes comment batches (`spyder_unified.py:1139+`, `1295-1308`).
6. Back in `_run_spider_background`, imported CSVs are persisted via `import_data_service` (`views.py:1651-1653`, `456+`).
7. `trigger_sentiment_analysis(theme)` is fired in thread to convert comment `sentiment_score` from default 0.5 to computed values (`views.py:1659`, `1400+`).
8. In parallel, `AIAnalysisWorker` consumes `analysis_status=0` videos and fills `visual_brightness`, `visual_saturation`, `audio_bpm`, `cut_frequency` (`views.py:66+, 162+, 256-262`).
9. Dashboard/charts query DB and render aggregated analytics (`dashboard`, `chart_user`, `chart_content`, `chart_sentiment`).

## 4.2 Predict-on-upload pipeline
1. User uploads video to `/predict/api/` (`views.py:1113`).
2. File stored temporarily in `media/temp_uploads` (`views.py:1131-1137`).
3. `VideoContentAnalyzer.run_full_analysis()` extracts physical features (`views.py:1140-1149`, `video_analyzer.py:98+`).
4. `calculate_theme_stats` computes theme baseline + Bayesian smoothing using DB digg data (`views.py:1175-1187`, `theme_baseline_engine.py:130+`).
5. `DiggPredictionService.predict_digg_count()` builds features -> scales -> predicts -> baseline-corrects -> optional fallback model (`views.py:1193-1196`, `predict_service.py:227+`).
6. LLM advice generated by `LLMService.generate_advice(...)` with timeout protection (`views.py:1230-1233`, `llm_service.py:32+`).
7. JSON response returns predicted likes, quality score, percentile info, baseline, advice (`views.py:1253-1261`).

# 5. Core Functional Modules
## 5.1 Data collection / crawling / import
- Problem solved: obtain themed Douyin videos/comments and persist them for analytics.
- User-visible entry: warehouse modal and global launch controls (`templates/base.html`, `templates/data/warehouse.html`).
- Backend path:
  - `/api/launch_spider/` -> `launch_spider_api` -> `_run_spider_background` -> `crawler.run_spider_service`
  - `/api/launch_comment_only/` -> `launch_comment_only_api` -> `_run_comment_only_background` -> `crawler.run_comment_only_service`
- Key files/functions:
  - `renhangxi_tiktok_bysj/douyin_hangxi/views.py:2026, 1472, 2126, 1698`
  - `crawler/spyder_unified.py:87, 601, 1139, 1437, 1462`
  - `services/data_manager.py:155, 270`
- Input: keyword/theme/max limits or existing video CSV.
- Output: CSV files in `data/`, DB rows in `tb_video` and `tb_comment`, cache status updates.
- Async: yes, multi-thread + crawler internal async DB queue.
- Defense value: proves full pipeline from acquisition to structured storage exists in code.

## 5.2 Data cleaning / refinement / warehousing
- Problem solved: normalize imported CSVs, ensure idempotent writes, and attach comments to existing videos.
- User-visible entry: "数据精炼" in warehouse.
- Backend path:
  - `/api/import_data/` -> `run_clean_data_api`
  - crawler pipeline -> `import_data_service`
- Key functions:
  - `views.py:600 (run_clean_data_api)`
  - `views.py:456 (import_data_service)`
  - `data_manager.py:155 (save_video_record), 270 (save_comment_batch)`
- Important note: there are two import logic paths (one direct ORM update_or_create in `run_clean_data_api`, one manager-based in `import_data_service`), which is functionally useful but increases behavior divergence risk.

## 5.3 Multimodal feature extraction
- Problem solved: convert raw video to numeric signals for model and analytics.
- User-visible effects: content charts and prediction feature quality.
- Backend path:
  - Offline/background: `AIAnalysisWorker` -> `VideoContentAnalyzer.run_full_analysis()`
  - Online/predict: `predict_api` -> same analyzer
- Key files/functions:
  - `views.py:66-329`, `1113-1159`
  - `video_analyzer.py:42, 98, 262, 337`
- Input: local MP4 path.
- Output: duration, brightness, saturation, cut frequency, BPM, script keywords.
- Async: background worker loop for dataset; synchronous for prediction API.
- Defense value: demonstrates CV/audio/NLP multimodal stack is actually executed, not mocked.

## 5.4 Fan portrait / audience analysis
- Problem solved: expose comment semantics and regional feedback structure.
- User-visible page: `/charts/user/`.
- Backend path: `chart_user` (`views.py:902`).
- Key logic:
  - word cloud from `content_clean` + jieba token filtering
  - IP-region sentiment bubble data aggregation
- Output: `wordcloud_json`, `bubble_data_json` for ECharts.
- Defense value: shows comment-level analytics beyond simple counts.

## 5.5 Visual analytics / chart generation
- Problem solved: link multimedia features with engagement and sentiment.
- Key endpoints:
  - `chart_content` (`views.py:945`) for brightness/saturation/BPM/cut distributions + top DNA profile
  - `chart_sentiment` (`views.py:1031`) for sentiment pie and bucketed relation with visual/audio dimensions
- Input: DB features/comments by theme.
- Output: multiple chart JSON payloads.
- Defense value: demonstrates actual analysis transformations and threshold logic.

## 5.6 Sentiment analysis / feedback analysis
- Problem solved: transform raw comments into cleaned text and sentiment labels.
- Core logic:
  - text cleaning: `clean_text_nuclear`, `_clean_text_service`
  - TF-IDF keyword extraction: `extract_semantic_features`
  - SnowNLP scoring and label mapping: `calculate_refined_sentiment`
  - batch reprocessing: `recalculate_sentiment_api`
  - incremental trigger: `trigger_sentiment_analysis`
- Key files/functions:
  - `views.py:367, 380, 395, 714, 1400`
  - `utils/comment_refiner.py` (offline refiner script)
- Defense value: can explain both online pipeline and offline refinement utilities.

## 5.7 Virality / heat prediction
- Problem solved: estimate likes and relative quality under theme context.
- Backend path: `predict_api` -> `DiggPredictionService.predict_digg_count`.
- Key files/functions:
  - `views.py:1193-1201`
  - `services/predict_service.py:35, 156, 227, 273, 288`
  - `ml_pipeline/theme_baseline_engine.py:130, 217`
- Input: multimodal features + account/time + theme baseline.
- Output: `predicted_digg`, `quality_score`, percentile text, baseline ref.
- Defense value: can justify why prediction is not only single regression output (has baseline correction and percentile framing).

## 5.8 AI diagnosis / recommendation generation
- Problem solved: convert numeric prediction context into actionable creator advice.
- Backend path: `predict_api` -> `LLMService.generate_advice`.
- Key files/functions:
  - `views.py:1210-1237`
  - `llm_service.py:11-13, 32+`
- Input: prediction, multimodal metrics, theme quantiles, optimal publish times, follower count.
- Output: structured markdown advice text.
- Dependencies: user API key (`CreatorConfig`) required.
- Defense value: integrates deterministic ML + generative explanation.

## 5.9 Admin-side management functions
- User/admin capabilities:
  - Django admin for videos/comments/model configs (`admin.py`)
  - theme switch/delete and profile model-key config (`views.py:771, 788, 1284`)
  - retraining trigger (`views.py:2207`)
- Defense value: shows ops/management considerations, not only model code.

# 6. Core Implementation Highlights
1. `analysis_status` state machine plus worker recovery.
- `Video.analysis_status` encodes pending/processing/completed/failed (`models.py:42`).
- Startup dead-letter recovery resets orphan processing tasks (`views.py:336-344`).
- Worker handles missing/corrupt files and marks failures (`views.py:186-213`, `296-307`).

2. Producer-consumer separation in crawler and persistence.
- Crawler downloads/collects; DB writes are decoupled by queue worker for comments (`spyder_unified.py:201-239`, `1300-1308`).
- Atomic `.part` download strategy and pre-commit OpenCV validation reduce bad file ingestion (`spyder_unified.py:369-517`).

3. Inference robustness with hot reload and fallback.
- Model assets loaded from manifest; automatic reload on mtime change (`predict_service.py:86-97`).
- On inference failure, fallback to previous model version (`predict_service.py:273-286`).

4. Theme-aware statistical correction.
- Prediction uses Bayesian-smoothed theme baseline and percentile context (`theme_baseline_engine.py:162-189`, `predict_service.py:288-299`).

5. Training deployment safety mechanism.
- `train_master_arena.py` trains challenger set (RF/LGBM/XGB), compares with champion, deploys only when RMSE wins by margin (`train_master_arena.py:599-608`).
- Deployment writes new artifact files and atomically updates root manifest (`train_master_arena.py:220-266`).

# 7. Model and Analysis Pipeline
## 7.1 Feature extraction and engineering pipeline
- Online extraction (`predict_api`): `VideoContentAnalyzer` yields brightness/saturation/BPM/cut/duration/keywords.
- Prediction feature builder (`predict_service._build_features`) creates:
  - base features: `follower_count_log`, `publish_hour`, `duration_sec`, `avg_sentiment`, visual/audio metrics
  - engineered features: `visual_impact`, `sensory_pace`, `sentiment_intensity`, `audio_visual_energy`, `content_density`
  - theme representation: Bayesian `theme_encoded` + one-hot theme columns if present in model topology
- Missing/unknown feature handling:
  - scaler mean fill for uninjected non-theme columns (`predict_service.py:214-221`)
  - `theme_Unknown` fallback mapping (`predict_service.py:211-213`)

## 7.2 Model loading and preprocessing
- Active inference model source is root `artifacts/version_manifest.json` via `services/predict_service.py`.
- Current active manifest indicates `best_model = RandomForest`, with `current_version = v20260317_091925`.
- Preprocessing uses persisted scaler + aligned feature order (`predict_service.py:239-241`).

## 7.3 Training pipeline actually wired to system
- Retrain API executes `ml_pipeline/train_master_arena.py` (`views.py:2228-2238`).
- Training script:
  - loads DB data from `Video` + comment sentiment aggregate (`train_master_arena.py:280-291`)
  - performs leakage-aware splitting/imputation and derived features (`352+`, `383+`, `393+`)
  - trains RF/LGBM/XGB challengers (`427-449`, `455+`)
  - showdown with champion and conditional deployment (`521+`, `599+`)

## 7.4 AI diagnosis logic
- LLM prompt integrates predicted likes, quantiles, quality score, percentile, and optimal publish windows (`llm_service.py` prompt body).
- Retry/backoff for 429/503 is implemented (`llm_service.py` loop near bottom).

## 7.5 Active vs candidate vs legacy logic
| Category | Evidence |
|---|---|
| Active runtime inference | `services/predict_service.py` + root `artifacts/version_manifest.json` |
| Active retrain path | `views.retrain_model_api` -> `ml_pipeline/train_master_arena.py` |
| Candidate models within active retrain | RF/LGBM/XGB challengers in `train_master_arena.py` |
| Legacy/alternative training scripts | `ml_pipeline/train_model_arena.py`, `train_universal_model.py`, `ml_pipeline/trainers/build_*.py` |
| Legacy/offline feature/data scripts | `ml_pipeline/feature_factory_v2.py`, `data_fusion_and_cleaning.py`, `export_db_to_csv.py` |
| Potentially stale artifacts | `ml_pipeline/artifacts/*` (separate from runtime root `artifacts/*`) |

# 8. Testing, Validation, and Stability
## 8.1 Evidence that system works
1. Unit and integration-lite tests exist and run under venv.
- Command result: `python -m pytest tests` -> 6 passed.
- Coverage includes:
  - Django DB read/write smoke (`tests/test_django_db.py`)
  - Bayesian smoothing correctness (`tests/test_topic_aware.py`)
  - prediction service API-shape integration test (`tests/test_topic_aware.py`)

2. Load/performance test scripts exist.
- Locust scenarios in `TEST0531` cover:
  - dashboard polling (`tasks/t1_dashboard_polling.py`)
  - prediction endpoint stress including corrupted input (`tasks/t2_ai_prediction.py`)
  - data import stress/idempotency/rollback scenarios (`tasks/t3_data_import.py`)

3. Runtime resilience mechanisms are present in code.
- crawler retries + rate-limit backoff (`spyder_unified.py:400+`)
- video corruption checks before commit (`spyder_unified.py:459+`)
- AI worker timeout/stuck handling (`views.py:106-121`)
- prediction fallback to previous model (`predict_service.py:273+`)
- LLM retry/backoff (`llm_service.py` retry loop)

## 8.2 Weak evidence / gaps
1. Full `pytest` discovery is fragile.
- `scripts/test_spider_connection.py` exits at import (`sys.exit(1)`), causing collection failure if running full tree.

2. Test depth mismatch.
- Existing tests do not fully validate end-to-end crawl -> import -> analyze -> charts workflow.
- No automated assertions for UI polling path consistency.

3. Data quality and completion gap (local DB evidence).
- 1218/2147 videos are currently `analysis_status=-1`.
- Only 7488/23499 comments have non-default sentiment score (`!=0.5`).

4. Multi-process stability caveat.
- Cache status is used heavily, but no explicit Redis cache backend in settings; default local-memory cache can be process-local.

# 9. What Teachers Are Most Likely to Ask
1. "Is this truly implemented backend logic, or mostly front-end pages?"
- Why they ask: many graduation projects over-focus on UI.
- Code evidence: crawler, persistence manager, worker, prediction service, training script are all concrete backend implementations.

2. "Is your async architecture really Celery-based as described in docs?"
- Why they ask: README/docs mention Celery/Redis.
- Code evidence: active runtime path is thread-based (`threading.Thread`, `AIAnalysisWorker`), not Celery queue workers in application path.

3. "How does raw data become model input step by step?"
- Why they ask: data pipeline authenticity.
- Code evidence: `run_spider_service` -> CSV + DB -> `analysis_status` worker -> feature fields -> prediction service.

4. "How do you prove model selection is not arbitrary?"
- Why they ask: model credibility.
- Code evidence: `train_master_arena.py` has challenger training + champion showdown + margin-based deployment.

5. "How do you handle bad files and external API instability?"
- Why they ask: robustness.
- Code evidence: download validation, worker file-size/stability checks, fallback model, LLM retry/backoff.

6. "Are there architectural inconsistencies or technical debt?"
- Why they ask: engineering maturity.
- Likely points:
  - monolithic `views.py`
  - multiple legacy training scripts and artifact locations
  - import/path inconsistency in some scripts
  - doc/runtime mismatch.

7. "Is retraining scope aligned with your active theme and analyzed data quality?"
- Why they ask: data validity.
- Code evidence: retrain trigger checks active theme 90%, but `train_master_arena.py` trains on global `Video` data and does not filter `analysis_status=2`.

# 10. Suggested Defense Answers
1. For "backend depth":
- "Core业务并不在模板里，而是在 `views.py` + `crawler/spyder_unified.py` + `services/predict_service.py`。例如 `/api/launch_spider/` 会启动后台线程执行抓取、入库、情感分析和AI特征队列，最终由看板读取数据库聚合结果。"

2. For "Celery mismatch":
- "当前可运行主路径采用 Django 进程内线程实现异步（`AIAnalysisWorker`、爬虫后台线程），这是我为了单机环境稳定和调试效率做的选择。仓库里保留了 Celery/Redis 相关配置，属于预留和历史演进，不是当前答辩演示主路径。"

3. For "data pipeline真实性":
- "可以按代码路径说明：`launch_spider_api` -> `_run_spider_background` -> `run_spider_service` -> CSV与ORM入库 -> `analysis_status`工作线程提取视觉/音频特征 -> 图表与预测页面消费这些字段。"

4. For "prediction meaning":
- "预测不是裸回归值：先做主题基准统计与贝叶斯平滑，再把模型质量分映射到主题分位区间，输出同类超越百分位信息，便于解释业务意义。"

5. For "robustness":
- "系统有多层兜底：下载阶段 `.part` 原子提交和 OpenCV 验证，分析阶段文件稳定性检查和超时回收，推理阶段 previous_version 回滚，LLM阶段 429/503 重试退避。"

6. For "known limitations":
- "我会明确承认并解释：目前仍有历史脚本与文档不一致、`views.py`耦合偏高、测试覆盖偏向单元层。我已经将这些纳入最终版本改进计划。"

# 11. Strongest Parts of the Project
1. End-to-end engineering path is complete and runnable.
- Not only inference demo: includes crawling, warehousing, feature extraction, analytics pages, retraining trigger, and user profile configuration.

2. Practical resilience design is visible in code.
- File corruption checks, retries, fallback model, timeout controls, cache-based progress telemetry.

3. Model pipeline has explicit deployment governance.
- Champion-challenger process with measurable deployment criterion (`win_margin`).

4. Defense-friendly observability.
- Multiple status APIs (`get_global_status`, `get_analysis_status_api`) and frontend monitoring widgets provide explainable runtime behavior.

# 12. Weakest Parts / Risk Points
1. Doc/runtime inconsistency risk.
- README and some docs mention Celery/Redis active runtime, but implementation path is threading-based.

2. Monolithic orchestration file.
- `views.py` carries too many responsibilities (ingestion, crawler orchestration, prediction, sentiment, status, retrain), which may be challenged as maintainability weakness.

3. Multi-path/legacy confusion risk.
- Coexistence of active and legacy training scripts plus two artifact directories (`artifacts/` vs `ml_pipeline/artifacts/`) can confuse "what is actually used".

4. Data quality and completion risk.
- High failure ratio in `analysis_status` and large neutral-default comment share may weaken claims of uniformly complete analysis across themes.

5. Specific implementation inconsistencies to prepare for:
- `LLMService.generate_advice` ignores incoming `model_name` and enforces default model.
- `delete_theme` response text claims physical disk removal, but function only deletes DB records.
- `launch_comment_only_api` singleton-guard import path differs from primary crawler import path.
- `run_clean_data_api` and `import_data_service` implement different import semantics.

# 13. Mid-Term Defense Focus
Mid-term defense should emphasize "real progress and implemented workload".

Recommended emphasis:
1. Show concrete completed modules with API paths.
- Crawl launch, import, dashboard, content/sentiment charts, prediction endpoint.

2. Demonstrate asynchronous pipeline actually running.
- Show `analysis_status` queue and status polling behavior.

3. Present measurable repository and DB evidence.
- Number of themes/videos/comments and module-level code paths.

4. Be explicit about in-progress risks.
- Say which parts are complete, which are stabilization/consistency tasks before final defense.

# 14. Final Defense Focus
Final defense should emphasize "architectural clarity and technical depth".

Recommended emphasis:
1. Explain architecture by data flow, not by page list.
- Request trigger -> backend function -> DB/cache/model -> UI output.

2. Deeply explain model + feature pipeline.
- Multimodal extraction, feature engineering, baseline correction, champion-challenger retraining.

3. Provide robust evidence and realistic limitations.
- Show passing tests and load-test scripts, then openly state unresolved debts and planned fixes.

4. Defend trade-offs.
- Why threading now, and how to evolve to stronger distributed async if needed.

# 15. Recommended Oral Narrative
## 15.1 3 minutes
1. Project goal and one-sentence architecture.
2. End-to-end pipeline in 5 steps: crawl -> import -> analyze -> predict -> diagnose.
3. One strongest technical point: model fallback/hot reload or champion deployment.
4. One honest limitation and one planned fix.

## 15.2 5 minutes
1. Business problem and why this system is needed.
2. Runtime architecture with key modules and APIs.
3. Multimodal + model pipeline explanation.
4. Testing/stability evidence.
5. Risks and improvement roadmap.

## 15.3 8 minutes
1. Full architecture walk-through with file-level evidence.
2. Detailed data flow (crawler path and predict path).
3. Core module deep dive (crawler, AI worker, predict service, retrain arena).
4. Model governance (champion/challenger, manifest, rollback).
5. Validation and performance testing strategy.
6. Limitations, mismatch handling, and final improvement plan.

# 16. Final Verdict
## 16.1 Implementation completeness
- Verdict: Medium-High.
- Reason: all major business chains are implemented in code and connected to UI/APIs, but not all data themes are equally complete in analysis outcomes.

## 16.2 Technical depth
- Verdict: High for graduation-project scope.
- Reason: includes crawler robustness, async workers, multimodal extraction, feature engineering, model deployment strategy, and LLM integration.

## 16.3 Defense readiness
- Verdict: Ready with conditions.
- Condition 1: clearly distinguish active runtime path from legacy/docs claims.
- Condition 2: proactively explain known risks (module coupling, path inconsistencies, incomplete analysis ratio).
- Condition 3: answer with code evidence (specific functions/files) rather than abstract descriptions.

Overall realistic assessment:
- This is a substantive engineering project with real implementation depth.
- It can defend well in both mid-term and final sessions if you present it as "implemented system + explicit trade-offs + clear improvement plan" instead of claiming perfect completeness.
