# Creator Business Process Mapping Report

## Workflow 1: 全局概览与状态追踪 (Global Overview & Status Tracking)
**Creator Intent:** The creator wants to get a high-level view of the current data assets, active themes, and monitor the progress of background AI/spider tasks.

**Step 1: 情况一眼看**
* **Presentation Layer:** Creator clicks/opens "情况一眼看".
* **Logic Layer:** Triggers the `dashboard` view function, which calculates average sentiment and interactive metrics (digg, comment counts) and checks the `AIAnalysisWorker` status.
* **Data Layer:** Reads from `Video` and `Comment` models via Django ORM (aggregating data based on the active theme).
* **Infra Layer:** Utilizes the web server (Django/Python) and potentially memory caching (e.g., Redis or local cache) for global pipeline status.

## Workflow 2: 数据采集与资产精炼 (Data Acquisition & Refining)
**Creator Intent:** The creator initiates cloud-based spider tasks to gather new video/comment data based on keywords, and then refines/cleans the raw CSV data into structured database entries with multimodal features.

**Step 1: 配置在线抓取任务**
* **Presentation Layer:** Creator clicks the "Start New Task" button to open the "配置在线抓取任务" modal.
* **Logic Layer:** Triggers the `launch_spider_api` backend task to initiate data scraping for target keywords and limits.
* **Data Layer:** Prepares temporal CSV files in the `data/` directory.
* **Infra Layer:** Utilizes DrissionPage for web scraping and network bandwidth for HTTP requests to TikTok servers.

**Step 2: 资产精炼工厂**
* **Presentation Layer:** Creator clicks/opens "资产精炼工厂" and submits data for refinement.
* **Logic Layer:** Triggers the `run_clean_data_api` to execute text cleaning (`clean_text_nuclear`) and triggers the `VideoContentAnalyzer` for multimodal feature extraction.
* **Data Layer:** Writes raw and processed features into the `Video` and `Comment` models via the `UnifiedPersistenceManager`.
* **Infra Layer:** Utilizes local PyTorch/CUDA environments (for `VideoContentAnalyzer`) and CPU for TF-IDF/SnowNLP text processing.

## Workflow 3: 数据明细管理 (Data Details Management)
**Creator Intent:** The creator checks the specific details of scraped videos and corresponding audience comments to curate content.

**Step 1: 视频信息保存处**
* **Presentation Layer:** Creator clicks/opens "视频信息保存处".
* **Logic Layer:** Triggers the `video_list` pagination and filtering logic.
* **Data Layer:** Reads from the `Video` database model.
* **Infra Layer:** Utilizes relational database query engine (SQLite/MySQL).

**Step 2: 观众评论保存处**
* **Presentation Layer:** Creator clicks/opens "观众评论保存处".
* **Logic Layer:** Triggers the `comment_list` view for paginated comment browsing.
* **Data Layer:** Reads from the `Comment` database model.
* **Infra Layer:** Utilizes relational database query engine.

## Workflow 4: 多维可视洞察分析 (Multidimensional Visual Insight Analysis)
**Creator Intent:** The creator visually analyzes the audience persona, video content features, and audience sentiment trends to guide future content creation.

**Step 1: 粉丝长什么样**
* **Presentation Layer:** Creator clicks/opens "粉丝长什么样".
* **Logic Layer:** Triggers the `chart_user` view to aggregate user interaction and demographic traits.
* **Data Layer:** Reads aggregated data from `Video` and `Comment` models.
* **Infra Layer:** Utilizes the ECharts parsing on the frontend and Django aggregate functions on the backend.

**Step 2: 视频深度可视化分析**
* **Presentation Layer:** Creator clicks/opens "视频深度可视化分析".
* **Logic Layer:** Triggers the `chart_content` view to map video multimodal features (brightness, saturation, BPM) against performance metrics.
* **Data Layer:** Reads from the `Video` model.
* **Infra Layer:** Utilizes backend analytical processing and frontend visualization.

**Step 3: 观众心情与想法**
* **Presentation Layer:** Creator clicks/opens "观众心情与想法".
* **Logic Layer:** Triggers the `chart_sentiment` view to group NLP-processed comments by sentiment labels.
* **Data Layer:** Reads from the `Comment` database model (specifically the sentiment_score and sentiment_label fields).
* **Infra Layer:** Utilizes NLP processing outputs powered by CPU.

## Workflow 5: AI 爆款生成决断 (AI Viral Content Prediction)
**Creator Intent:** The creator uses historical feature data and trained models to predict the performance of future video concepts before production.

**Step 1: 预测视频火不火**
* **Presentation Layer:** Creator clicks/opens "预测视频火不火".
* **Logic Layer:** Triggers the `predict_page` view and `predict_api` to execute the champion model inference via `predict_service`.
* **Data Layer:** May read the serialized machine learning model files (e.g., joblib/pkl) and `CreatorConfig` or baseline stats.
* **Infra Layer:** Utilizes CPU/GPU computing resources for Machine Learning inference (e.g., Random Forest/XGBoost execution).

## Workflow 6: 账号及系统配置 (Account & System Configuration)
**Creator Intent:** The creator manages personal account credentials and switches operational theme spaces.

**Step 1: 我的账号**
* **Presentation Layer:** Creator clicks/opens "我的账号".
* **Logic Layer:** Triggers `profile_view` to handle user authentication state and preferences.
* **Data Layer:** Reads/Writes to the default Django `User` model.
* **Infra Layer:** Utilizes Django's Session/Authentication Middleware.
