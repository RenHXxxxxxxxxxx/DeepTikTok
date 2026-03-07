# Stage 0: Model Baseline & Code Logic Audit

## 1. Model Identification & Typology
The machine learning pipeline currently utilizes three core models:
1. **Model A:** `XGBRegressor` (Ensemble Tree, Gradient Boosting, from the `xgboost` library).
2. **Model B:** `RandomForestRegressor` (Ensemble Tree, Bagging, from the `scikit-learn` library).
3. **Model C:** `XGBRegressor` ("Universal Model" for point estimation mapped natively over log-transformed target variables, similarly an Ensemble Tree regression model from the `xgboost` library).

All three are tree-based ensemble algorithms focusing strictly on continuous regression tasks (`digg_count` point estimation).

## 2. Feature Composition (The "Diet")
The input vector utilizes structured metadata alongside multimodal feature signals extracted through an automated factory process (`feature_factory_v2.py`), scaling up dynamically to a **29-D feature space**:
- **Baseline Physics:** Raw values such as `follower_count`, `duration_sec`, `publish_hour`.
- **Visual Traits:** Extracted via pixel-level GPU aggregation (`visual_brightness`, `visual_saturation`).
- **Audio Traits:** Rythmic characteristics measured in beats-per-minute (`audio_bpm`).
- **Audio/Visual Edits:** Scene shifts represented by `cut_frequency`.
- **Text/NLP Traits:** Aggregated string sentiment scoring encoded as `avg_sentiment`.
- **Categorical Mapping:** The NLP `theme_label` string is mapped directly to distinct flags via One-Hot Encoding (`pd.get_dummies()`).
- **Synthesized Interaction Indices (Dynamic):** A synthetic layer combines physical traits into non-linear expressions calculated during inference:
  - `visual_impact` = `(brightness * saturation) / 1000.0`
  - `sensory_pace` = `bpm * cut_frequency`
  - `sentiment_intensity` = `abs(sentiment - 0.5) * 2`
  - `audio_visual_energy` = `brightness * bpm / 1000.0`
  - `content_density` = `cut_frequency / (duration + 1)`

*Transformation/Scaling:* 
Continuous variables missing values are imputed to global medians using `SimpleImputer(strategy='median')`. Missing unseen inference traits default to the `scaler.mean_` metric to prevent feature decay. All final input structures are Z-Score standardized via the `StandardScaler()` estimator.

## 3. Current Training Methodology
**Training Logic Analysis:**
- **Cross-Validation Implementation:** 
  - `train_model_arena.py` implements a 5-Fold Cross-Validation cycle (`n_splits=5`, `KFold(shuffle=True)`) targeting `scikit-learn`'s `cross_validate()` function.
  - `train_universal_model.py` utilizes a flat Hold-Out set via `train_test_split()` bounded at an 80/20 proportion ratio (`test_size=0.2`).
- **Hyperparameter Tuning:** Currently, **zero** formalized continuous hyperparameter bounding (e.g., `GridSearchCV`, `RandomizedSearchCV`, `Optuna`) exists. The `fit()` function triggers on statically hardcoded estimator thresholds (`n_estimators=100`, `learning_rate=0.1`, `max_depth=5`) across both ensemble architectures.
- **Log Transformation:** 
  - The singular XGBRegressor target variable evaluates using Numpy's `log1p(y)` and reconstructs during scoring via `expm1()`, mitigating exponential target distribution skew.

## 4. Evaluation Metrics & Current Effects
**Metrics Currently Captured:**
1. **RMSE:** Recorded via `-cv_results['test_neg_rmse'].mean()`.
2. **MAE (Mean Absolute Error):** Computed over real, non-bounded point space after logarithmic predictions revert.
3. **R-Squared ($R^2$):** Analyzes distribution coverage performance within log boundaries.

**Artifact Selection and Deployment Flow:**
Within the model arena function (`train_and_evaluate()` in `train_model_arena.py`), the environment aggregates total negative RMSE scores. The script scans the array keys, filtering `min(results, key=lambda k: results[k]['mean_rmse'])`. 
Upon confirming the champion model (lowest `mean_rmse`), the architecture re-evaluates the optimal parameters against the *full* data distribution (Global `fit(X, y)` operation), locking the serialized model structure, `StandardScaler()` topology, and an architectural manifest (`version_manifest.json`) down to the immutable memory disk `d:\renhangxi_tiktok_bysj\ml_pipeline\artifacts` directories.
