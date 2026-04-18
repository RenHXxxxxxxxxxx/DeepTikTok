# Project Comprehensive Analysis Report

## 1. Executive Summary
As of **2026-03-21**, the project is a working **Django-based Douyin/TikTok data acquisition and analysis system** with a real end-to-end path from crawling/import to multimodal feature extraction, sentiment aggregation, prediction, and dashboard display.

Evidence indicates that substantial implementation already exists:
- Core entities and persistence are implemented in `Video`, `Comment`, `AIModelConfig`, and `CreatorConfig` (`renhangxi_tiktok_bysj/douyin_hangxi/models.py`).
- Crawling and comment supplementation are implemented (`crawler/spyder_unified.py`, `crawler/crawl_comments_only.py`) and exposed by API endpoints (`renhangxi_tiktok_bysj/douyin_hangxi/urls.py`, `views.py`).
- A background AI worker consumes pending videos and writes extracted multimodal features (`AIAnalysisWorker` in `views.py`).
- Prediction and model hot reload/versioning are implemented (`services/predict_service.py`, `artifacts/version_manifest.json`).
- Training/retraining scripts with champion-challenger logic are implemented (`ml_pipeline/train_master_arena.py`, `views.retrain_model_api`).
- Frontend pages for dashboard, data warehouse, charts, and prediction are implemented (`templates/*`).

The project workload is demonstrably non-trivial:
- Database currently contains **2147 videos** and **23499 comments** (`db.sqlite3`, queried on 2026-03-21).
- There are many theme-scoped datasets (`data/douyin_video_*.csv`, `data/douyin_comment_*.csv`).
- Multiple model artifacts and versions exist (`artifacts/model_*.pkl`, `scaler_*.pkl`, `version_manifest.json`).

Main gaps before defense:
- Significant **doc/code inconsistency** (MySQL/Celery/Django-Q claims vs current SQLite + threading runtime).
- High failed-analysis proportion in DB (`analysis_status=-1` has 1218 rows vs 929 completed).
- Security and governance gaps (debug settings, plaintext secrets, coarse permissions, CSRF exemptions on sensitive endpoints).
- Testing exists but is limited and environment-dependent.

## 2. Project Positioning and Goal
**Project identity (from code/UI/docs):**
- "抖音数据采集与智能分析系统" / "多模态爆款预测工作台" (`README.md`, `templates/prediction/dashboard.html`).

**Core purpose:**
- Acquire short-video data (videos + comments), process multimodal signals (visual/audio/text), and support creator decision-making with analytics + popularity prediction + AI advice.

**Problem solved:**
- Transforms fragmented, noisy social-video data into structured metrics and decision signals (theme-level baselines, predicted likes, sentiment trends, content feature distributions).

**Intended users:**
- Content creators (theme management, data warehouse, charts, prediction page).
- Admin/maintainer users via Django admin and maintenance scripts.

**Real-world use scenario:**
- A creator selects a theme, crawls/imports datasets, monitors analysis progress, views audience/content/sentiment insights, uploads a candidate video for predicted performance, and receives strategy advice.

**Project type assessment:**
- **Hybrid (system-oriented + applied research)**.
- System-oriented: multi-module web platform with data pipeline, worker orchestration, management, and visualization.
- Research-oriented: feature engineering, model comparison, retraining pipeline, theme baseline statistics.

## 3. Repository / Directory Structure
### 3.1 High-level structure
Key top-level directories/files and practical roles:
- `renhangxi_tiktok_bysj/`: Django project config and active app package.
- `renhangxi_tiktok_bysj/douyin_hangxi/`: main app code (models/views/urls/admin/utils/migrations).
- `crawler/`: Douyin crawler and comment-only crawling.
- `services/`: persistence and prediction service layer.
- `ml_pipeline/`: training and baseline/statistics scripts.
- `templates/`: frontend pages (dashboard, warehouse, charts, prediction, auth, profile).
- `static/` and `renhangxi_tiktok_bysj/douyin_hangxi/static/`: frontend static assets.
- `data/`: raw/intermediate CSV datasets.
- `artifacts/`: deployed model/scaler/version manifest.
- `media/`: temporary uploads and pending videos.
- `docs/`: architecture/audit/thesis-support Markdown docs.
- `tests/` and `TEST0531/`: unit-style tests and Locust performance test scripts.

### 3.2 Notable structure observations
- There is a duplicate analyzer path: `douyin_hangxi/utils/video_analyzer.py` (top-level) and `renhangxi_tiktok_bysj/douyin_hangxi/utils/video_analyzer.py` are byte-identical. This is maintainability risk (duplicate authoritative source).
- Many markdown audit documents exist (`docs/*.md`, 35 files), but code is the actual source of truth for implementation state.
- Root also contains many maintenance/debug scripts and logs; some are clearly active, some are legacy/stale.

### 3.3 Asset category mapping
- Source code: `renhangxi_tiktok_bysj/douyin_hangxi`, `crawler`, `services`, `ml_pipeline`.
- Config/env: `requirements.txt`, `docker-compose.yml`, `.env`, `settings.py`.
- Data: `data/*.csv`, `db.sqlite3`.
- Models/artifacts: `artifacts/*.pkl`, `artifacts/version_manifest.json`, `media/ai_models/*`.
- Documentation: `README.md`, `docs/*.md`, one `.docx` template file.
- Tests: `tests/*.py`, `TEST0531/*`.

## 4. Technology Stack Analysis
### 4.1 Confirmed actual stack in code
- Languages: Python, HTML, JavaScript, SQL (SQLite).
- Backend framework: Django 5.2 (`requirements.txt`, `settings.py`).
- Admin UI: `django-simpleui` (`INSTALLED_APPS`, `requirements.txt`).
- Database runtime: SQLite (`settings.py` -> `django.db.backends.sqlite3`, `db.sqlite3`).
- Crawler: DrissionPage (`crawler/spyder_unified.py`, `requirements.txt`).
- Async/task execution (runtime): in-process threads (`AIAnalysisWorker`, background thread launch in `views.py`), not task queue workers.
- Data processing: pandas, numpy (`views.py`, `ml_pipeline/*`).
- ML models: RandomForest, LightGBM, XGBoost, scaler/joblib (`ml_pipeline/train_master_arena.py`, `services/predict_service.py`).
- NLP/sentiment: SnowNLP + jieba (`views.py`, `utils/comment_refiner.py`, `video_analyzer.py`).
- Multimodal extraction: OpenCV + librosa + Whisper + PyTorch (`utils/video_analyzer.py`).
- Visualization: ECharts in templates (`templates/charts/*.html`, CDN includes).
- LLM integration: OpenAI SDK with DeepSeek base URL (`utils/llm_service.py`).
- Testing: unittest-style tests + Locust scripts (`tests`, `TEST0531`).
- Deployment artifacts: Dockerfile and docker-compose (present but consistency issues below).

### 4.2 Internal consistency check
Consistent with code:
- Django + SQLite + thread-based async + artifact-based model loading are internally coherent.

Inconsistencies found:
- `README.md` and several docs claim **Celery/Redis task queue as active runtime**, but core execution path uses threading in Django process (`AIAnalysisWorker`, `launch_spider_api`, `retrain_model_api`).
- Multiple docs claim **MySQL** architecture, while active settings use SQLite.
- `docs/whisper_asr_architecture.md` shows `moviepy` extraction logic, but current analyzer uses `ffmpeg` via `subprocess`.
- Profile logic stores selectable LLM model (`CreatorConfig.llm_model_name`), but `LLMService.generate_advice` forces `deepseek-chat` and ignores `model_name`.
- Docker worker command uses `python -m celery -A core worker`, but there is no clear Celery app module `core` in repo.

## 5. Functional Module Analysis
### 5.1 Authentication and account management
Status: **Completed (basic), partial (role governance)**
- Implemented login/register/logout/profile flows (`urls.py`, `views.register`, `views.user_logout`, `views.profile_view`, `templates/registration/*`, `templates/users/profile.html`).
- Supports per-user API key/model preference storage (`CreatorConfig`).
- Limitation: role/permission granularity is weak; critical operations are mostly `@login_required` only, with no explicit admin-only guard checks (`is_staff`, `is_superuser`, `has_perm` not found in project code).

### 5.2 Data collection / crawler
Status: **Implemented (major)**
- Main crawler pipeline class: `DouyinUnifiedPipeline` (`crawler/spyder_unified.py`).
- Supports video collection, comment collection, progress callbacks, status handling, local pending video path usage.
- API launch endpoints exist (`launch_spider_api`, `launch_comment_only_api`).
- Evidence of anti-instability handling and queueing exists in code comments and logic.
- Limitation: some helper scripts are stale (`crawler/run_comment_crawler.py` import path points to `data.spyder_unified`).

### 5.3 Data import / supplement collection
Status: **Implemented**
- `run_clean_data_api` supports importing local video/comment CSV by theme.
- `import_data_service` + `UnifiedPersistenceManager` handle persistence and idempotent updates.
- Comment supplement flow exists via comment-only crawler + auto import.

### 5.4 Topic/theme data management
Status: **Implemented (functional), with design caveats**
- Theme switching/deletion endpoints (`switch_theme`, `delete_theme`) and UI controls in `templates/data/warehouse.html` and `base.html`.
- Theme context retrieval via session and `Video.theme_label` distinct values.
- Caveat: theme is a denormalized string field; manager may append comma-separated labels in some paths, which weakens relational consistency.

### 5.5 Data cleaning and sentiment analysis
Status: **Implemented**
- Text cleaning + semantic extraction + SnowNLP scoring (`clean_text_nuclear`, `extract_semantic_features`, `calculate_refined_sentiment`).
- Full-theme recalculation endpoint (`recalculate_sentiment_api`).
- Incremental trigger function exists (`trigger_sentiment_analysis`).

### 5.6 Multimodal feature extraction
Status: **Implemented (core module)**
- `VideoContentAnalyzer` extracts brightness, saturation, cut frequency, audio BPM, and ASR keywords.
- Includes timeout wrappers, Whisper lock/cache, GPU detection, fallback tags, and temp file cleanup.
- AI worker writes these features into `Video` table and advances status.

### 5.7 Audience/fan portrait and feedback analysis
Status: **Implemented (analytics level)**
- `chart_user` computes word cloud and regional sentiment bubbles from comments.
- `chart_sentiment` computes sentiment distributions and visual/audio bucket sentiment response.
- `chart_content` relates multimodal features to engagement distributions/top DNA.

### 5.8 Popularity prediction
Status: **Implemented (strong)**
- Prediction endpoint `predict_api` accepts uploaded video + account metadata.
- Calls `VideoContentAnalyzer`, builds feature dict, uses `DiggPredictionService` for inference.
- Returns predicted likes, quality score, percentile rank, baseline info.

### 5.9 AI diagnosis / recommendation generation
Status: **Implemented (dependent on user key and external API)**
- `LLMService.generate_advice` called from `predict_api` with timeout guard.
- Uses OpenAI SDK and DeepSeek-compatible endpoint.
- Limitation: user-selected model field is not effectively honored in current service logic.

### 5.10 Admin-side management
Status: **Partially implemented**
- Django admin model management for `Video`, `Comment`, `AIModelConfig` is implemented (`admin.py`).
- App-level management pages exist for data warehouse and model retraining trigger.
- Missing stronger admin-only authorization boundaries and operation auditing model.

### 5.11 Model management and retraining
Status: **Implemented (core), partial integration with UI config model**
- Training and deployment path implemented in `train_master_arena.py` with manifest-based versioning.
- Retraining API starts script asynchronously and resets prediction singleton.
- `AIModelConfig` exists in DB/admin but runtime prediction service primarily uses `artifacts/version_manifest.json`, not `tb_ai_model_config` active pointer.

### 5.12 Error logging and task monitoring
Status: **Implemented (basic/medium)**
- Logging and status polling endpoints exist (`get_global_status`, `get_analysis_status_api`, `global_tracker.js`).
- Worker and crawler contain extensive try/except and progress cache writes.
- Limitation: logging is mixed with prints; no centralized structured observability stack.
## 6. System Workflow Reconstruction
### 6.1 Main end-to-end pipeline (implemented)
1. User starts collection/import from Data Warehouse (`templates/data/warehouse.html`).
2. For online crawl, `launch_spider_api` starts background thread and crawler service.
3. Crawler writes CSV and/or temporary pending video files; metadata/comments are persisted.
4. Imported or crawled videos are inserted/updated in `tb_video` with `analysis_status=0` pending.
5. `AIAnalysisWorker` polls pending videos, marks processing, validates file readiness, runs `VideoContentAnalyzer`, writes multimodal features, sets status complete or failed.
6. Dashboard and global tracker poll status APIs and show scraping/AI progress and throughput.
7. Charts aggregate theme-scoped records for user/content/sentiment views.
8. Prediction page uploads a candidate video, extracts multimodal features, computes theme baseline stats, executes model prediction, optionally invokes LLM advice, and returns frontend results/PDF export.
9. Optional retraining can be triggered when analysis progress reaches threshold (>=90% in active theme), updating artifact manifest and hot-reloading inference service.

### 6.2 Admin/maintenance side flow
- Django admin manages data rows and AIModelConfig entries.
- Scripts (`scripts/fix_anything.py`, `scripts/verify_status.py`) provide maintenance utilities.
- Caveat: some maintenance paths are stale or path-dependent.

## 7. Data and Storage Analysis
### 7.1 Data model and relationships
- `Video` (`tb_video`) is the central entity with multimodal features + engagement + status fields.
- `Comment` (`tb_comment`) has FK to `Video` and stores cleaned text and sentiment.
- `CreatorConfig` (`tb_creator_config`) links one-to-one with Django `User` for LLM credentials.
- `AIModelConfig` (`tb_ai_model_config`) stores upload-based model assets and active flag.

### 7.2 Schema fitness for business logic
Strengths:
- `analysis_status` and `local_temp_path` directly support asynchronous AI consumer workflow.
- Comment FK supports sentiment and audience aggregation linked to video/theme.
- Indexes on `theme_label` and `analysis_status` help common filter paths.

Inconsistencies/weaknesses:
- Theme is a string in multiple tables rather than normalized topic table; potentially inconsistent when appending multi-theme labels.
- `AIModelConfig` and artifact-manifest management are two parallel model management channels with unclear single source of truth.
- User API key is stored directly in DB char field (no encryption at rest).

### 7.3 Runtime data evidence (2026-03-21)
- `tb_video`: 2147 rows.
- `tb_comment`: 23499 rows.
- `analysis_status`: `-1` failed 1218, `2` completed 929.
- Top themes by video count include `高燃混剪`, `美食探店`, `一路生花`.
- `tb_ai_model_config`: 1 row.
- `tb_creator_config`: 1 row.

### 7.4 File storage
- `data/*.csv` contains many theme datasets (video/comment sources).
- `artifacts/*` contains model/scaler versions and manifest.
- `media/pending_videos` currently empty at inspection time (consistent with cleanup-after-processing design).

## 8. Model / Algorithm Analysis
### 8.1 Training strategy
Implemented in `ml_pipeline/train_master_arena.py`:
- Loads data from ORM with aggregated `avg_sentiment`.
- Feature engineering includes duration, publish hour, log follower count, Bayesian theme encoding, one-hot theme dummies, and composite multimodal features.
- Uses train/test split, outlier handling, imputation, scaling.
- Compares RandomForest / LightGBM / XGBoost with parameter grids and CV.
- Champion-challenger showdown logic decides deployment by RMSE margin.
- Deploys model/scaler and writes `version_manifest.json` with metadata.

### 8.2 Inference strategy
Implemented in `services/predict_service.py`:
- Singleton service with manifest mtime hot reload.
- Builds feature frame dynamically using manifest topology (`feature_names_in` preferred).
- Calculates theme Bayesian encoding using baseline stats + manifest global mean.
- Executes prediction with scaler, score mapping, baseline correction.
- Supports fallback to `previous_version` model on inference exception.

### 8.3 Baseline/statistics layer
`ml_pipeline/theme_baseline_engine.py` provides:
- Theme baseline stats (mean/std/percentiles).
- Bayesian smoothing against global stats for low-sample themes.
- Optimal publishing time recommendation via weighted hour-density logic.

### 8.4 Evidence of versioning/replacement
- `artifacts/version_manifest.json` has `current_version`, `previous_version`, `best_model`, metrics, feature metadata.
- `retrain_model_api` triggers retraining and resets service singleton for hot load on next inference.

### 8.5 Practical issues observed
- Runtime environment mismatch can break fallback path (missing dependencies for older model serialization types).
- `AIModelConfig` table is not the primary live selector in current predictor path.

## 9. UI, Pages, and Demonstrable Results
Major user-facing pages and what they prove:
- `templates/dashboard.html`: overall progress, AI queue, throughput, retrain trigger, key KPIs.
- `templates/data/warehouse.html`: CSV import, online crawl launch, comment supplement launch, theme activation/deletion.
- `templates/data/video_list.html` and `templates/data/comment_list.html`: persisted asset details and pagination.
- `templates/charts/user_charts.html`: word cloud + region-sentiment bubble.
- `templates/charts/content_charts.html`: multimodal feature distributions and engagement map.
- `templates/charts/sentiment_charts.html`: sentiment pie + visual/audio sentiment bars + theme operations.
- `templates/prediction/dashboard.html`: upload/predict/quality score/percentile/AI advice/PDF export.
- `templates/users/profile.html`: user API key management.

High-value demonstrable outputs for mid-term:
- Live queue progression and status transitions in dashboard.
- Chart pages showing transformed data products (not raw CSV only).
- Prediction result card with baseline-relative rank and generated advice.
- Model artifacts and manifest showing real model lifecycle.

## 10. Admin-Side Capability Analysis
What is actually implemented:
- Django admin model management (`VideoAdmin`, `CommentAdmin`, `AIModelConfigAdmin`) with custom list displays and edit filters.
- App-level management actions: theme switching/deletion, import, crawl launch, retraining API.

What is not strongly implemented:
- No explicit fine-grained role-based permission checks in custom views (mostly `@login_required`).
- No dedicated operation log table for admin actions.
- No explicit permission-controlled model switch workflow connected to active inference path.

Assessment:
- Admin capability is **usable but not strongly governed**.
- Suitable for prototype/mid-term demonstration, but weaker for production-grade governance claims.
## 11. Testing, Robustness, and Security
### 11.1 Testing
Present:
- `tests/test_basic.py` trivial sanity test.
- `tests/test_django_db.py` model CRUD connectivity intent.
- `tests/test_topic_aware.py` Bayesian smoothing and prediction-service integration tests.
- `TEST0531/*` Locust scripts for dashboard/prediction/import load scenarios.

Execution evidence (2026-03-21):
- `python -m unittest discover -s tests -v` ran 6 tests with 1 import error due missing Django in active interpreter environment.
- Topic-aware tests ran and showed expected smoothing behavior.
- Predictor test returned safe fallback output under dependency/runtime constraints.

Conclusion:
- Testing assets exist, but reproducible automated quality gate is not fully solid in current environment.

### 11.2 Robustness mechanisms in code
- Worker timeout/stuck detection and dead-letter recovery (`AIAnalysisWorker`, `start_ai_worker`).
- File integrity and stable-write checks before analysis.
- Temp file cleanup after processing.
- Crawler-side progress callbacks and exception handling.
- Prediction service hot reload and fallback mechanisms.
- SQLite WAL activation (`renhangxi_tiktok_bysj/db_signals.py`).

### 11.3 Security and operational risks
- `DEBUG=True`, `ALLOWED_HOSTS=[]` in settings.
- Sensitive endpoints use `@csrf_exempt` (`launch_spider_api`, `retrain_model_api`).
- API keys stored in plaintext-style fields (`CreatorConfig.llm_api_key`) and `.env` includes a real key string.
- No explicit RBAC for destructive or expensive operations.

## 12. Thesis-to-Code Consistency Check
### 12.1 Supported by implementation
- End-to-end crawler -> data persistence -> AI analysis -> visualization -> prediction flow.
- Multimodal extraction with visual/audio/ASR signals.
- Model retraining with artifact versioning and runtime reload behavior.
- Theme baseline statistics and Bayesian smoothing logic.

### 12.2 Partially supported
- "Admin model management" is present, but mostly at prototype governance level.
- "MLOps" is partially realized through artifact manifest/version rotation, but lacks full CI/CD and strict model registry governance.

### 12.3 Inconsistent or overstated claims in docs
- MySQL-centered architecture claims conflict with active SQLite runtime (`settings.py`).
- Celery/Redis active queue claims conflict with thread-based runtime in `views.py`.
- Some architecture docs mention Django-Q; no active Django-Q integration was found.
- `whisper_asr_architecture.md` describes `moviepy` extraction path, but current code uses `ffmpeg` subprocess extraction.
- Some reports describe production-ready posture more strongly than current security/testing state supports.

### 12.4 Latest-document handling note
- No thesis filenames with `最新/最终/final/latest` cues were found.
- Latest docs by modified time (e.g., `implementation_verification_report.md`, `training_logic_audit.md`, `whisper_asr_architecture.md`) were prioritized for comparison.
- Even among latest docs, selective drift from code exists; code was treated as final authority.

## 13. Completion Assessment
### 13.1 Evidence-based completion by dimension
- Core system workflow completion: **high (about 80-85%)**.
  - Reason: major pipeline modules and UI are connected and operational.
- Supporting analytics/prediction completion: **medium-high (about 75-80%)**.
  - Reason: chart modules, predictor, baseline, and retraining exist.
- Admin/governance completion: **medium (about 55-65%)**.
  - Reason: admin functions exist but role control and audit rigor are limited.
- Testing/reliability completion: **medium-low to medium (about 45-60%)**.
  - Reason: robustness code is substantial, but test automation/dependency consistency is weak.
- Thesis/document alignment completion: **medium (about 60-70%)**.
  - Reason: several key claims are supported, but multiple stack/architecture claims are mismatched.

### 13.2 Overall practical completion
A defensible overall estimate for mid-term is roughly **70-80% practical completion**, with strong visible workload and core functionality, but clear polishing needs in reliability governance and thesis consistency.

## 14. Mid-Term Defense Presentation Value
### 14.1 What already proves sufficient workload
- Full-stack integration: crawler + DB + AI worker + model pipeline + frontend dashboards.
- Real dataset scale (thousands of videos/comments, many themes).
- Non-trivial model engineering (feature topology, Bayesian smoothing, versioned artifact deployment).
- Multimodal processing (visual/audio/ASR + text sentiment).
- Operational mechanics (progress polling, queue states, retraining trigger).

### 14.2 Strongest progress-proof modules
- `AIAnalysisWorker` orchestration with status lifecycle and fault handling.
- `VideoContentAnalyzer` multimodal extraction.
- `DiggPredictionService` with manifest-driven hot reload/fallback.
- `train_master_arena.py` champion-challenger training/deployment logic.

### 14.3 Safe claims for defense
Safe to claim:
- Implemented end-to-end data-to-insight system for Douyin short-video analysis.
- Implemented multimodal feature extraction and theme-aware popularity prediction.
- Implemented retraining and versioned model deployment mechanics.

Avoid or qualify:
- "Production-ready" security/governance level.
- "Fully Celery-based distributed async runtime" (not reflected in current code path).
- "MySQL deployment architecture already in active use" (current runtime is SQLite).

### 14.4 Likely teacher challenge points
- Why docs mention MySQL/Celery while code uses SQLite/threading.
- Why failed analysis count is high (`analysis_status=-1` proportion).
- Whether LLM model selection is truly configurable.
- What test evidence exists for stability under load and edge conditions.

### 14.5 High-impact polishing before defense
- Align written architecture claims with current runtime reality.
- Prepare one slide with measured module completion and known limitations (shows engineering honesty).
- Reduce failed-analysis backlog for cleaner progress metrics.
- Prepare a short reproducible demo script covering crawl/import -> analyze -> chart -> predict.

## 15. Risks, Weaknesses, and Open Issues
High-priority risks:
- **Doc-code mismatch risk:** architecture claims in several docs do not match runtime implementation.
- **Data quality/processing risk:** high failure-state share in `analysis_status` can be questioned.
- **Security risk:** plaintext secrets and CSRF exemptions on sensitive task endpoints.
- **Permission risk:** destructive or expensive operations available to any logged-in user.

Medium-priority risks:
- **Model management dual path:** `AIModelConfig` and artifact-manifest pathways may diverge.
- **Legacy/stale scripts:** some files use outdated import paths or assumptions.
- **Duplicate source file risk:** duplicate analyzer path increases confusion during maintenance.
- **Environment reproducibility risk:** tests and model inference can fail with missing/incompatible dependencies.

Lower-priority but notable:
- Mixed logging style (`print` + logger) and limited centralized observability.
- Non-normalized theme schema can complicate long-term analytics consistency.

## 16. Priority Recommendations
### Highest priority
1. **Synchronize thesis/docs with actual runtime architecture immediately.**
2. **Create a defense-safe "truth table" of implemented vs planned modules, with evidence paths.**
3. **Fix security baseline for demo environment:** remove exposed key material, tighten CSRF and sensitive operation controls.
4. **Address AI failure backlog:** inspect top failure causes and recover a meaningful subset before defense.

### Medium priority
1. **Unify model management source of truth** (artifact manifest vs DB config model).
2. **Add minimal role checks** for destructive/admin operations.
3. **Stabilize reproducible test command** and dependency lockstep for demonstration machines.
4. **Retire or clearly mark stale scripts/docs** to prevent reviewer confusion.

### Nice-to-have
1. Add one concise architecture diagram generated directly from current code modules.
2. Add small benchmark table (processing throughput, prediction latency).
3. Add structured operation log model for important actions.
## 17. Suggested PPT Material Extraction
Recommended **6-slide** core set:

1. **Slide: End-to-End System Pipeline**
- Focus: Crawl/import -> DB -> AI worker -> analytics/prediction -> user output.
- Show: architecture flow diagram + endpoint list from `urls.py` + key worker class names.
- Proof points: demonstrates integrated workload, not isolated scripts.

2. **Slide: Data Asset Scale and Theme Coverage**
- Focus: dataset size and theme breadth.
- Show: DB counts (videos/comments/status), top themes, sample CSV inventory from `data/`.
- Proof points: demonstrates real data volume and operational progress.

3. **Slide: Multimodal Feature Extraction Implementation**
- Focus: visual/audio/ASR extraction pipeline.
- Show: `VideoContentAnalyzer` module highlights and dashboard indicators using extracted metrics.
- Proof points: demonstrates non-trivial technical depth beyond CRUD.

4. **Slide: Prediction and Model Lifecycle**
- Focus: inference path and retraining/versioning.
- Show: prediction UI result card, `version_manifest.json` fields, retrain trigger logic.
- Proof points: demonstrates ML engineering workload and iterative model governance.

5. **Slide: User-Facing Analytics Results**
- Focus: user/content/sentiment chart pages and business insight value.
- Show: screenshots of three chart pages + short interpretation examples.
- Proof points: demonstrates outcome-oriented analytics, not only backend work.

6. **Slide: Completion Assessment, Risks, and Next Milestones**
- Focus: honest status and defense-ready roadmap.
- Show: module completion matrix, top 5 risks, prioritized fixes before final defense.
- Proof points: demonstrates engineering maturity and credible planning.

## 18. Appendix: Evidence Index
- `renhangxi_tiktok_bysj/settings.py` - Confirms active DB engine (SQLite), debug flags, installed apps.
- `renhangxi_tiktok_bysj/urls.py` - Confirms routing entry to app URLs and admin endpoint.
- `renhangxi_tiktok_bysj/douyin_hangxi/urls.py` - Confirms implemented page/API surface.
- `renhangxi_tiktok_bysj/douyin_hangxi/models.py` - Confirms core data schema and relationships.
- `renhangxi_tiktok_bysj/douyin_hangxi/migrations/0001_initial.py` - Confirms base Video/Comment model creation.
- `renhangxi_tiktok_bysj/douyin_hangxi/migrations/0006_video_analysis_status_video_local_temp_path.py` - Confirms async analysis status and temp path fields added.
- `renhangxi_tiktok_bysj/douyin_hangxi/migrations/0008_creatorconfig.py` - Confirms user config model exists.
- `renhangxi_tiktok_bysj/douyin_hangxi/views.py` - Confirms workflow logic, APIs, worker orchestration, prediction, retraining.
- `renhangxi_tiktok_bysj/douyin_hangxi/apps.py` - Confirms auto-start of AI worker in app lifecycle.
- `renhangxi_tiktok_bysj/db_signals.py` - Confirms SQLite WAL activation behavior.
- `renhangxi_tiktok_bysj/douyin_hangxi/utils/video_analyzer.py` - Confirms multimodal extraction implementation details.
- `renhangxi_tiktok_bysj/douyin_hangxi/utils/llm_service.py` - Confirms DeepSeek/OpenAI integration and model selection behavior.
- `services/predict_service.py` - Confirms manifest-based model load, hot reload, fallback, feature construction.
- `services/data_manager.py` - Confirms persistence manager, idempotent writes, comment batch persistence, retry logic.
- `crawler/spyder_unified.py` - Confirms crawler core, queueing/progress, video download pipeline.
- `crawler/crawl_comments_only.py` - Confirms comment-only supplement crawler exists.
- `crawler/run_comment_crawler.py` - Shows stale import path risk (`data.spyder_unified`).
- `crawler/config.py` - Shows MySQL-style legacy config inconsistent with active runtime.
- `ml_pipeline/train_master_arena.py` - Confirms model training, feature engineering, champion-challenger deployment.
- `ml_pipeline/theme_baseline_engine.py` - Confirms Bayesian smoothing and optimal publish-time computation.
- `artifacts/version_manifest.json` - Confirms deployed model version metadata and feature topology.
- `artifacts/model_v20260317_091925.pkl` - Confirms non-trivial trained model artifact exists.
- `templates/base.html` - Confirms navigation, theme operations, global task modal wiring.
- `templates/dashboard.html` - Confirms queue/progress/retraining monitoring UI.
- `templates/data/warehouse.html` - Confirms import/crawl/comment-supplement/theme management UI.
- `templates/prediction/dashboard.html` - Confirms upload-predict-result-advice-PDF user flow.
- `templates/charts/user_charts.html` - Confirms fan portrait/word-cloud visualization.
- `templates/charts/content_charts.html` - Confirms multimodal content-feature visualizations.
- `templates/charts/sentiment_charts.html` - Confirms sentiment analysis visualization and theme operations.
- `renhangxi_tiktok_bysj/douyin_hangxi/admin.py` - Confirms admin-side data/model interfaces.
- `tests/test_topic_aware.py` - Confirms test intent for Bayesian smoothing and prediction integration.
- `tests/test_django_db.py` - Confirms DB test intent for ORM CRUD.
- `TEST0531/locustfile.py` - Confirms performance testing scenario design exists.
- `requirements.txt` - Confirms declared dependency stack.
- `docker-compose.yml` - Confirms declared deployment intent (with potential mismatch to actual runtime wiring).
- `README.md` - Provides declared architecture/stack claims for consistency comparison.
- `docs/implementation_verification_report.md` - Latest verification doc; partially aligned with model pipeline.
- `docs/training_logic_audit.md` - Latest training audit context.
- `docs/whisper_asr_architecture.md` - Latest ASR architecture doc; partially drifted from current code.
- `docs/4Layer_System_Architecture_Design.md` - Shows doc claims (MySQL/Django-Q) used in consistency check.
- `docs/Project_Research_Report.md` - Shows broader thesis claims used for support/inconsistency analysis.
