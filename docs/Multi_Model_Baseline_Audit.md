# Multi-Model Baseline Audit Report

**Date:** 2026-03-06
**Scope:** Architectural Audit of `build_LGBM.py`, `build_RF.py`, and `build_XGBOOST.py`
**Objective:** Risk assessment prior to unifying into a "6-Model Arena Leaderboard" training framework.

---

## 1. Artifact Collision & Overwrite Risk (CRITICAL)

### Context
All three scripts currently persist their model weights, scalers, and a version manifest at the end of their execution via the `save_artifacts()` equivalent block.

### Findings
*   **Manifest Overwriting (Absolute Data Loss):** All three scripts write their metadata to the **exact same target path**: `manifest_path = os.path.join(assets_dir, 'version_manifest.json')`. If trailing scripts are executed concurrently or sequentially, the last script to finish will indiscriminately overwrite the `version_manifest.json`. This completely erases the tracking metadata for the other two models (e.g., their scaling parameters, features, and model file pointers), destroying the ability to track multiple models simultaneously.
*   **Naming Inconsistencies:** The artifact naming convention is visibly fragmented. The Random Forest and LightGBM scripts prefix their `.pkl` files with their algorithm name (e.g., `model_RF_{version_id}.pkl`, `model_LGBM_{version_id}.pkl`), whereas the XGBoost script generically names its artifact `model_{version_id}.pkl`. This makes artifact discovery brittle and ambiguous when scanning a directory containing multiple models.

---

## 2. Scaler & Feature Space Fragmentation

### Context
Each of the three scripts independently defines and fits its own `StandardScaler` to its implicitly loaded data partition.

### Findings
*   **Redundant Computation & Parameter Drift:** Although all scripts use `random_state=42`, initializing `StandardScaler().fit_transform(X_train)` in three separate scripts produces three identical sets of mean and variance parameters if the random states and distributions remain perfectly synchronized. However, if data logic ever drifts between files, the scalers will diverge.
*   **Evaluation Impracticality:** During a unified "6-Model Leaderboard" phase, evaluating test samples through three disparate, uncoupled scalers creates massive inference overhead. A fair leaderboard fundamentally demands that the validation set is transformed by exactly **one truth-source scaler**. Having three independent scalers breaks the theoretical symmetry required to ensure all models are evaluated on the identical mathematical feature space.

---

## 3. Feasibility of the "6-Model Evaluation" & Proposed Design

### Context
The goal is to load 3 frozen baseline models alongside training 3 new algorithms concurrently on fresh data, evaluating all 6.

### Architectural Complexity Assessment
Currently, attempting a 6-model evaluation is impossible due to the `version_manifest.json` overwrite bug. The system possesses no mechanism to chronologically distinguish "current baselines" from "historical baselines." Loading 3 old `.pkl` files requires scanning directory hierarchies blindly since there is no unified registry mapping multiple active champions to their respective algorithms. The memory footprint of 6 concurrent tree-based regressors is manageable, but the pipeline orchestration to align historical scalers with historical models, and new scalers with new models, is totally absent.

### Proposed Unified `train_arena.py` Architecture

To achieve a stringent, mathematically fair 6-Model Arena, the architecture must centralize scaling and metadata tracking.

1.  **Unified Data Ingestion & Splitting:** Single query ingestion. Split the dataset once into `X_train` and a unified **Global Hold-out Validation Set** (`X_test`).
2.  **Global Feature Space Initialization:** Instantiate and fit exactly **ONE** `StandardScaler` strictly on `X_train`. Apply this single scaler to both train and hold-out sets.
3.  **Concurrent Model Fitting (New Candidates):** Fit the 3 new algorithms (XGBoost, LightGBM, Random Forest) against this centrally scaled `X_train`.
4.  **Baseline Retrieval:** Fetch the 3 historical baseline `.pkl` files and their 1 associated historical global scaler from a newly structured `arena_master_manifest.json` (acting as a registry, not a single-file overwrite).
5.  **Unified Evaluation Engine:**
    *   Transform the global validation set using the *new* scaler for the 3 *new* models.
    *   Transform the global validation set using the *historical* scaler for the 3 *baseline* models.
    *   Compute metrics (R2, RMSE) for all 6 models independently across these matched spaces.
6.  **Leaderboard & Deployment Gate:** Rank all 6 implementations. If a new candidate outperforms the reigning Champions, select the single best Champion.
7.  **Atomic Artifact Commits:** Save the new model, save the *one unified scaler*, and append the winner to `arena_master_manifest.json`, ensuring historical states are vaulted, not overwritten.
