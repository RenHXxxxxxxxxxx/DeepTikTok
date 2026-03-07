# Project Status & Architecture Report

> **Last Updated:** 2026-02-10
> **Version:** 2.0 (Post-"Great Refactor")
> **Status:** Stable / Production-Ready

This document serves as the **Source of Truth** for the `renhangxi_tiktok_bysj` project. It documents the architectural decisions, fault-tolerance mechanisms, and data flows established during the recent refactoring.

---

## 1. 🏗️ System Architecture Overview

The system has evolved from a monolithic script into a robust **Producer-Consumer** architecture, decoupled via the Database and File System.

### 1.1 High-Level Data Flow

```mermaid
graph LR
    subgraph Producer [Crawler Layer]
        A[spyder_unified.py] -->|Downloads| B(media/pending_videos)
        A -->|Metadata| C(MySQL: Video Table)
    end

    subgraph Storage [Persistence Layer]
        B -->|Temporary Storage| D{File System}
        C -->|Status=0 (Pending)| E[(MySQL Database)]
    end

    subgraph Consumer [AI Analysis Layer]
        F[AIAnalysisWorker] -->|Polls| E
        F -->|Reads| D
        F -->|Updates Features| E
        F -->|Deletes File| D
    end
```

### 1.2 Component Roles

*   **Producer (`spyder_unified.py`)**:
    *   **Responsibility**: Scrapes video data, downloads video files, and saves metadata to the database.
    *   **Output**: Creates `.mp4` headers in `media/pending_videos` and inserts records into MySQL with `analysis_status=0`.
    *   **Independence**: Completely unaware of the AI analysis process. It focuses solely on data acquisition and integrity.

*   **Consumer (`AIAnalysisWorker` in `views.py`)**:
    *   **Responsibility**: A background daemon thread running within the Django process.
    *   **Operation**: Continuously polls the database for "Pending" videos (`analysis_status=0`).
    *   **Action**: Performs multi-modal analysis (Visual/Audio) and updates the database with extracted features.
    *   **Cleanup**: **Immediately deletes** the local video file after successful analysis to manage disk space for the RTX 3060 environment.

---

## 2. 🕷️ Crawler Core Evolution (`spyder_unified.py`)

The crawler has been hardened against network instability and anti-scraping measures.

### 2.1 Robustness Mechanisms

*   **The "Double-Lock" Wait Mechanism**:
    *   **Concept**: Prevents "False Empty" results where the scraper checks for data before the AJAX request completes.
    *   **Implementation**:
        1.  **Hard Wait**: 3-second unconditional sleep to allow network initiation.
        2.  **Smart Wait**: 10-second explicit wait for `ul li a[href*="/video/"]` elements.
        3.  **Visual Confirmation**: Only proceeds if actual video elements are detected in the DOM.

*   **Session Isolation**:
    *   **Problem**: Previous versions used global history to stop crawling, causing immediate exits on restarts.
    *   **Fix**: Introduced `session_collected` counter. The crawler now guarantees fetching `N` new videos *in the current session*, regardless of historical data.

*   **Atomic Writes**:
    *   **Mechanism**: Videos are first downloaded as `.mp4.part`.
    *   **Validation**: OpenCV verifies the `.part` file (checks for corruption/truncation).
    *   **Commit**: Only valid files are renamed to `.mp4`. This ensures the Consumer never encounters a half-written or corrupted file.

*   **Circuit Breakers**:
    *   **Duration**: Videos longer than **15 minutes** are deleted immediately (preventing AI worker timeouts).
    *   **Size**: Downloads larger than **300MB** are truncated or discarded.

*   **Fault Tolerance Strategies**:
    *   **Network Recovery**: On consecutive timeouts, the browser automatically refreshes or navigates to `about:blank` to reset the connection.
    *   **DOM Fallback (Strategy C)**: If network packet interception fails, the crawler switches to "Eyes Open" mode, scraping video IDs directly from the HTML DOM to salvage data.

---

## 3. 💾 Data Persistence & Management

### 3.1 Async Strategy
*   **Video Data**: Written synchronously to ensuring strong consistency (File + DB Record must exist together).
*   **Comment Data**: Written **asynchronously** via `_db_queue` and `_async_db_worker`. This prevents the high-volume comment IO from blocking the video scraping loop.

### 3.2 Consistency & State Management
*   **Zombie Record Prevention**: The `fix_anything.py` tool provides a `handle_orphans` module to import `.mp4` files that missed database entry.
*   **Status Guard**:
    *   `save_video_record` ensures that existing "Completed" videos are not accidentally reset to "Pending".
    *   `AIAnalysisWorker` includes a **Stuck Detection** mechanism: if a video remains in "Processing" state for > 3 minutes, it is marked as failed (`-1`) to prevent queue blockage.

### 3.3 Storage Structure
*   **`media/pending_videos/`**: Transient storage for video files. Empty when the system is idle (as the Consumer deletes processed files).
*   **`data/*.csv`**: Legacy/Backup flat-file storage.

---

## 4. 🧠 AI Analysis Pipeline

### 4.1 Decoupling
The AI logic is fully isolated in `views.py` -> `AIAnalysisWorker`. The crawler does not import `snownlp` or `torch`, keeping the scraping process lightweight and stable.

### 4.2 Current Capabilities
The `VideoContentAnalyzer` extracts:
*   **Visual**: Brightness, Saturation.
*   **Audio**: BPM (Beats Per Minute).
*   **Editing**: Cut Frequency (Shot changes per second).
*   **Sentiment**: Comment sentiment analysis using `SnowNLP` (Cleaned via "Nuclear" text cleaning).

### 4.3 Hardware Constraints & Optimization
*   **Single Threaded**: The worker runs as a single thread to respect the RTX 3060's VRAM limits.
*   **Batch Size**: Processes videos in small batches (5) to avoid database locks.
*   **Aggressive Cleanup**: The "Process & Delete" specific strategy is critical for running long-term tasks on limited disk space.

---

## 5. 🚀 Future Roadmap (The "To-Do" List)

Based on the current architecture, these are the immediate next steps:

### 5.1 Comment Mining
*   **Status**: `run_comment_crawler.py` exists but needs full integration into the `spyder_unified.py` workflow or a separate scheduler.
*   **Goal**: Implement a "Rhythmic" crawler that targets specific high-value videos for deep comment extraction.

### 5.2 Batch Processing
*   **Current**: Continuous polling.
*   **Plan**: Move AI analysis to off-peak hours (e.g., 2 AM - 6 AM) using a scheduler, allowing the crawler to utilize full bandwidth during the day.

### 5.3 Performance
*   **Download Acceleration**: Move from sequential `requests.get` to a `ThreadPoolExecutor` for video downloads, saturating the network bandwidth.
