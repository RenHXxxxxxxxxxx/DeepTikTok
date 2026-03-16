# Black-Box Functional Audit Report
**Project:** Multimodal TikTok Short Video Intelligent Analysis System
**Role:** Senior System Architect and Business Analyst

## 1. Actors Definition
The system interactions are fundamentally driven by two primary roles. From a black-box perspective, these are the entities that interact with the system's external interfaces.

*   **Content Creator / Marketer (Primary Actor):** The end-user who utilizes the system to monitor market trends, analyze competitors' video content, extract multimodal features (visual/audio), gauge audience sentiment, and leverage AI predictions to optimize their own content strategy.
*   **System Administrator (Secondary Actor):** The user responsible for global system configuration, such as setting up global API keys (DeepSeek/ERNIE), managing crawler quotas, physical data deletion, and monitoring the overall pipeline health (AI Worker).

---

## 2. Use Case Inventory
This section breaks down the system into functional clusters. Each item below represents a granular action that maps to a single "Ellipse" in a UML Use Case Diagram.

### 2.1 Data Acquisition & Asset Management (Acquisition Engine)
*   **Initiate Crawler Task:** User submits a scraping task by providing keywords, quantity limits, and a specific "Theme" label.
*   **Monitor Task Status:** User views real-time progress bars (ETA, throughput) of the active crawler task.
*   **Import External Data:** User uploads/imports legacy CSV files (video and comment data) into the data warehouse.
*   **Browse Video Archive:** User views a paginated list of collected video metadata.
*   **Browse Comment Archive:** User views a paginated list of raw comment strings and their details.
*   **Switch Active Workspace (Theme):** User toggles between datasets belonging to different thematic queries.
*   **Clear Theme Data:** User securely wipes all video and comment records associated with a specific theme from the physical database.

### 2.2 Multimodal Feature Quantification
*   **Trigger Multimodal Extraction:** System automatically (or user manually) processes video files to quantify Brightness, Saturation, BPM, and Scene-cut frequency.
*   **View Visual DNA Distribution:** User visualizes histograms or pie charts of low/mid/high thresholds for screen brightness and saturation.
*   **View Audio DNA Distribution:** User monitors rhythm pacing through BPM and scene-cut frequency distributions.
*   **Analyze Engagement Correlation:** User views a scatter plot mapping visual traits (e.g., Brightness) against actual engagement (e.g., Digg counts).

### 2.3 Sentiment & Popularity Analysis (Intelligence & Decision Support)
*   **View NLP Word Cloud:** User visualizes high-frequency, cleaned keywords extracted from audience comments.
*   **Analyze Regional Sentiment:** User views a bubble chart mapping the volume and average sentiment of comments based on IP locations.
*   **Monitor Sentiment Temperature:** User views a meter displaying the proportion of "Positive," "Neutral," and "Negative" audience reactions.
*   **Trigger Retroactive Sentiment Repair:** User forces the system to recalculate previously computed sentiment scores against updated noise-filtration rules.

### 2.4 AI-Driven Diagnosis (Personalization & Prediction)
*   **Upload Draft Video:** User uploads an unpublished MP4 video file to the prediction laboratory.
*   **Configure Predictive Parameters:** User inputs expected follower count and targeted publish hour for the simulation.
*   **View Predicted Engagement:** User receives a quantitative forecast (e.g., predicted Digg count) and a percentile ranking compared to historical theme benchmarks.
*   **Receive AI Operational Output:** User reads a generated, highly specialized structural diagnosis and publishing advice produced by an LLM (typically DeepSeek or ERNIE).

### 2.5 System Configuration (Personalization)
*   **Manage Authentication:** User signs up, logs in, or logs out of the platform.
*   **Configure LLM API Keys:** User inputs or updates personal credentials for third-party AI interfaces.
*   **Select Processing Model:** User selects the preferred AI engine (e.g., ERNIE 4.0, DeepSeek logic).
*   **View Account Footprints:** User observes personal system usage metrics (quotas consumed) and recent activity logs.

---

## 3. Primary Scenario Flow (The "Happy Path")
This describes the standard, end-to-end journey of a Content Creator utilizing the system to derive actionable intelligence.

1.  **Authentication & Configuration:** The User logs into the dashboard and navigates to the Profile module to securely input their DeepSeek API key and model preferences.
2.  **Market Data Acquisition:** The User enters the Data Warehouse module and launches a Spider Task by inputting a competitor keyword (e.g., "Tech Reviews") and assigning a "Theme" name. The User monitors the real-time progress bar until completion.
3.  **Automated Multimodal Profiling:** In the background, the system performs a sequence of feature extractions on downloaded videos (calculating visual/audio DNA) and calculating sentiment scores for scraped comments. The User views the live percentage completion rate via the Dashboard.
4.  **Insight Visualizations:** The User navigates through the Chart views (User, Content, Sentiment) to observe where audience IPs concentrate, what words resonate most, and the exact screen brightness ranges that correlate with high Digg counts within the specified Theme.
5.  **Draft Video Upload:** The User creates a draft video based on these insights and uploads it to the "AI Prediction Laboratory."
6.  **AI Diagnosis Output:** The User inputs their current follower count. The system processes the Draft Video, matches its features against the Theme's statistical baseline, predicts the potential Digg count, and outputs an expert diagnostic report with specific operational tweaks before the video goes live.

---

## 4. System Boundaries
Defining the locus of control clarifies what is manually orchestrated versus automatically driven.

*   **User Responsibilities (External to System):**
    *   Defining the exact business query (Keywords, Quotas, Theme Label).
    *   Creating and uploading the raw draft video file.
    *   Providing external API credentials (LLM Keys).
    *   Interpreting the final strategic advice and adjusting the actual video on the TikTok platform.
*   **System Automation (Internal Handling):**
    *   *Black-box crawling:* Navigating TikTok's DOM, handling pagination, and downloading binary media/text.
    *   *Signal Processing:* Extracting structural numbers (Brightness, BPM) from raw MP4 files via background worker threads.
    *   *NLP Pipeline:* Tokenizing Chinese strings, filtering stop words, and mapping sentiment lexicons (via Jieba and SnowNLP).
    *   *Statistical Aggregation:* Grouping benchmarks, calculating Bayesian smoothings, and percentile ranks.
    *   *API Orchestration:* Formatting the multimodal feature array alongside statistical baselines into a prompt and handling network requests to the LLM backend.
