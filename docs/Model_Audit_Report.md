# 1. Overview & Meta-data

- **Latest Model Filename:** `model_v20260223_190915.pkl`
- **Latest Scaler Filename:** `scaler_v20260223_190915.pkl`
- **Date Trained:** 2026-02-23T19:13:44.613085
- **Model Version:** v20260223_190915
- **Performance Metrics (MSE, RMSE, R2, etc.):** Not recorded in `version_manifest.json` or logs. Output only printed to console during `build_model/build.py`.
- **Dataset Size:** Not explicitly recorded in manifest/logs.
- **Feature Count:** 28

## 2. Model Architecture & Feature Space

**Model Architecture:** XGBRegressor

**Feature Columns (28 total):**
- `duration_sec`
- `follower_count`
- `publish_hour`
- `avg_sentiment`
- `visual_brightness`
- `visual_saturation`
- `cut_frequency`
- `audio_bpm`
- `visual_impact`
- `sensory_pace`
- `sentiment_intensity`
- `audio_visual_energy`
- `content_density`
- **`theme_PDD`**
- **`theme_刀马舞`**
- **`theme_北大`**
- **`theme_卢本伟`**
- **`theme_大先生`**
- **`theme_户外运动`**
- **`theme_新疆大学`**
- **`theme_旅行`**
- **`theme_法环`**
- **`theme_硬核科技`**
- **`theme_美食探店`**
- **`theme_职场技能`**
- **`theme_酷跑`**
- **`theme_青春没有售价旅行`**
- **`theme_非遗手工艺`**

## 3. Integration Status & Gap Analysis

### Overview
This analysis compares the explicit feature space required by the pre-trained `v20260223_190915` XGBoost model to the current live prediction logic found within `predict_service.py`.

### 🚨 Discrepancies & Findings

There are some core discrepancies that expose risk or currently affect predictions:

| Issue | Description | Impact Level |
| :--- | :--- | :--- |
| **Missing Core Feature** | `avg_sentiment` is a trained feature in the target `.pkl` file but is **omitted** from the static `self.feature_cols` list in `predict_service.py`. | **HIGH** - `avg_sentiment` is defaulted to `0.0` in the prediction array rather than reading actual request data. |
| **Theme Baseline Safety** | The models require 15 specific `theme_` One-Hot encoded columns (e.g. `theme_PDD`, `theme_北大`). `predict_service.py` handles this dynamically by checking `dynamic_theme_col in expected_features`. | **LOW** - Logic safely defaults to `0.0` for all themes, activating only the matching one, successfully averting shape mismatch errors. |
| **Previous Crash Origin** | The previous Numpy crash (`ValueError: at least one array or dtype is required`) was triggered because fallback mechanisms for `expected_features` did not account for the 15 dynamic `theme_` columns nor the missing `avg_sentiment` feature when `feature_names_in_` failed to resolve, passing a heavily truncated array (size 12 instead of 28) into the scaler. | **RESOLVED / HIGH RISK** - While fortified, `predict_service.py`'s fallback array `self.feature_cols + self.theme_cols` would still technically fail since `avg_sentiment` is completely missing. |

### Recommendations for Remediation
- **Immediate Fix required:** Add `'avg_sentiment'` to the `self.feature_cols` array in `predict_service.py:70-79` to ensure incoming data routes successfully to the prediction DataFrame.
