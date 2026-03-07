# MLOps Retraining Pipeline: Logical Architecture Audit Report

## 1. Executive Summary
This audit evaluated the current AI model retraining pipeline (`train_model_arena.py`, `views.py`, and `predict_service.py`) against production-grade MLOps standards. While the pipeline successfully automates data ingestion, feature engineering, and model persistence, it currently lacks structural protections against **Catastrophic Forgetting**, lacks a true **Champion-Challenger validation mechanism**, and employs an **unsafe artifact Hot-Swap** that exposes the production environment to zero-downtime failures and file-locking collisions.

---

## 2. Vulnerability Matrix

| Dimension | Risk Level | Vulnerability Description | Root Cause Location |
| :--- | :---: | :--- | :--- |
| **Data Layer** | <span style="color:red">**High**</span> | **Global Unbounded Training (Memory / Slowdown)**<br>The system continuously aggregates *all* historical data for every retraining cycle. While this ironically *prevents* catastrophic forgetting of old themes, it creates a scalable bottleneck. It does not employ a bounded "Replay Buffer" or proportionate sampling (e.g., 70% new / 30% global). | `train_model_arena.py` -> `load_data()`:<br>`Video.objects.filter(analysis_status=2)` |
| **Evaluation Layer** | <span style="color:red">**High**</span> | **No Champion-Challenger Showdown**<br>The pipeline uses 5-Fold CV to select the best *new* algorithm (e.g., XGB vs RF) via lowest CV RMSE. However, it **never compares** the winning Challenger against the currently deployed Champion model. It indiscriminately overwrites the deployed model even if the new model performs worse globally. | `train_model_arena.py` -> `train_and_evaluate()` |
| **Evaluation Layer** | <span style="color:orange">**Medium**</span> | **Missing Statistical Significance Threshold**<br>The selection is based on `min(mean_rmse)`. A Challenger might "win" by an insignificant margin (e.g., $\Delta < 0.001$), leading to unnecessary production turbulence due to statistical noise. | `train_model_arena.py` -> `save_artifacts()` |
| **Deployment Layer** | <span style="color:red">**High**</span> | **Non-Atomic Artifact Swap & Lock Collisions**<br>`joblib.dump` overwrites `.pkl` artifacts directly while `views.py` resets the memory Singleton. If an incoming inference request hits `predict_service.py` at the exact millisecond `joblib.dump` is writing the file, it will read a corrupted pickle, causing the service to crash. | `train_model_arena.py` -> `joblib.dump()`<br>`predict_service.py` -> `_load_assets()` |
| **Deployment Layer** | <span style="color:red">**High**</span> | **No Physical Fallback / Rollback**<br>If the newly exported `model_XGB_v2026xxxx.pkl` fails during inference (e.g., dimension mismatch), `predict_service.py` catches the error but permanently falls back to `predicted_digg: 0`. There is no logic to revert `version_manifest.json` back to the last known-good Champion version. | `predict_service.py` -> `predict_digg_count()` |

---

## 3. Architectural Blueprint: The Ideal Workflow

To resolve the above vulnerabilities, the following logical workflow should be implemented without fundamentally rewriting the entire infrastructure:

### Phase 1: Stratified Replay Buffer (Data Layer)
To balance memory constraints and Catastrophic Forgetting:
1. **Target Constraint:** Maximize training payload at $N = 10,000$ samples.
2. **Data Extraction Logic:**
   - **Challenger Data (New):** Query the $7,000$ most recent videos from the `active_theme`.
   - **Replay Buffer (Global Historical):** Uniformly sample $3,000$ videos across all *other* historical `theme_label`s to maintain global baseline metrics.
   - **Merge & Shuffle:** Combine DataFrames before passing to the feature engineering pipeline.

### Phase 2: The Champion-Challenger Arena (Evaluation Layer)
To prevent model degradation:
1. **Global Hold-out Validation Set:** Set aside 15% of the Replay Buffer strictly for the final showdown. Do *not* use this in the 5-Fold CV.
2. **Algorithm Selection (Challenger vs Challenger):** Evaluate XGB vs RF via 5-Fold CV on the 85% training split. The winner becomes the **Challenger**.
3. **The Showdown (Champion vs Challenger):**
   - Load the *Current Champion* from the production `artifacts_dir`.
   - Evaluate both the Champion and the Challenger strictly on the 15% Global Hold-out Set using **$R^2$** and **RMSE**.
   - **Deployment Gate:** The Challenger is ONLY deployed if $(RMSE_{Champion} - RMSE_{Challenger}) > 0.02$ (Statistical Significance). 
   - If the Challenger fails the gate, the script exits gracefully without overwriting the manifest.

### Phase 3: Atomic Swaps & Zero-Downtime Rollback (Deployment Layer)
To ensure system stability during hot-swapping:
1. **Atomic Write:**
   - Write the Challenger to a temporary file: `model_temp.pkl`.
   - Write the new manifest to `manifest_temp.json`.
   - Use `os.replace('model_temp.pkl', 'model_[uuid].pkl')` to guarantee an OS-level atomic swap, preventing file-locking corruption.
2. **Singleton Thread-Safety:**
   - `views.py` sets `DiggPredictionService._instance = None`. 
   - The double-checked locking mechanism in `predict_service.py` safely handles the re-initialization.
3. **Silent Fallback:**
   - In `predict_service.py`, if `joblib.load` or `model.predict()` crashes on the new UUID, catch the exception, automatically read the *previous* UUID from `version_manifest.json` (requires saving a `previous_version` key), and reload the Champion model from disk.
