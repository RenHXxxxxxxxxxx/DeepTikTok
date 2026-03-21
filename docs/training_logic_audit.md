# Training Logic Audit Report

## 1. Dynamic Dimensionality (CRITICAL)
**Status:** Partial Compliance / Needs Alignment
**Analysis:** 
The primary `train_master_arena.py` accurately implements dynamic feature topologies by dynamically computing dummy variables (`pd.get_dummies`) and serializing the exact feature order via the `feature_importances` keys into the manifest. Online prediction (`predict_service.py`) successfully reconstructs this space.
**Identified Risks:**
- **Implicit Topology Dependence:** `predict_service.py` relies on `feature_importances` dictionary keys to guarantee column order (`list(feat_dict.keys())`). While Python 3.7+ maintains insertion order, this is fragile. A change in serialization logic could scramble the feature matrix.
- `train_universal_model.py` completely lacks the `theme_encoded` and `theme_*` dummy parameters inside its `GLOBAL_CONFIG["feature_columns"]`.

**Code Snippet Recommendation:**
Explicitly save and rely on the `feature_names_in_` array instead of deriving the topology from `feature_importances`.

*In `train_master_arena.py`:*
```python
# *明确将列顺序保存到 manifest 中，不要依赖 feature_importances 的字典键序*
challenger_metadata['feature_names_in'] = feature_cols
```

*In `predict_service.py`:*
```python
# *优先直接读取 feature_names_in 保证拓扑排序 100% 对齐*
if 'feature_names_in' in self.manifest:
    self.ordered_feature_names = self.manifest['feature_names_in']
elif feat_dict:
    self.ordered_feature_names = list(feat_dict.keys())
```

## 2. Bayesian Target Encoding & Cold Start
**Status:** Robust Implementation, Minor Hardcoding Flaws
**Analysis:**
Both `train_master_arena.py` and `predict_service.py` implement Bayesian smoothing `(count * local_mean + weight * global_mean) / (count + weight)`. This provides excellent protection against cold-start overfitting on rare, newly crawled themes.
**Identified Risks:**
- **Asymmetric Global Means:** In `train_master_arena.py`, `global_mean` is dynamically derived (`df_biased['temp_digg_log'].mean()`). However, within `predict_service.py`, the `global_mean_log` is hardcoded as `np.log1p(10000.0)`. This drift between training and inference can cause distribution skew for rare themes.

**Code Snippet Recommendation:**
Pass the `global_mean` via `manifest.json` during model export.

*In `train_master_arena.py`:*
```python
# *将基于训练动态分布的全局均值写入字典，避免两端不一致*
challenger_metadata['bayesian_global_mean'] = float(global_mean)
```

*In `predict_service.py`:*
```python
# *优先使用模型训练时导出的基准线，否则回退兜底配置*
global_mean_log = float(self.manifest.get('bayesian_global_mean', np.log1p(10000.0)))
```

## 3. Outlier & Null Handling
**Status:** Moderately Robust
**Analysis:**
Missing features are imputed using the median fitted strictly on the training partition (`X_train[col].median()`). Targets are securely target-capped (`y_train_orig_pre.quantile(0.95)`). Multi-modal faults are caught by `_safe_float` online.
**Identified Risks:**
- Heavy-tail features like `follower_count` are log-transformed, which stabilizes gradients, but they are *not* explicitly capped before the `StandardScaler`. Extreme spikes in new follower data can still distort `X_scaled_ndarray`.

**Code Snippet Recommendation:**

*In `train_master_arena.py`:*
```python
# *对自变量也执行盖帽操作，抵御极端长尾样本导致的新维度爆炸*
p99_follower = X_train['follower_count_log'].quantile(0.99)
X_train['follower_count_log'] = X_train['follower_count_log'].clip(upper=p99_follower)
X_test['follower_count_log'] = X_test['follower_count_log'].clip(upper=p99_follower)
```

## 4. Serialization Compliance
**Status:** Compliant
**Analysis:**
Model weights (`model_*.pkl`) and the parameters (`scaler_*.pkl`) are atomically stored alongside a JSON manifest. The `predict_service.py` gracefully fills unknown fields with global means and drops unrecognized features for backwards compatibility.
**Identified Risks:**
None detected. The Atomic Deployment with `.tmp` and `os.replace` / `shutil.move` guarantees no dirty reads during concurrent prediction traffic.

## Checklist for Safely Adding New Topics
- [ ] Evaluate the new topic's initial frequency. Ensure it surpasses `GLOBAL_CONFIG['min_theme_samples']` (currently set to 5) before training triggers; otherwise, it resolves safely to `Unknown`.
- [ ] Monitor the Bayesian smoothed value for the new topic. If metrics differ significantly from the hardcoded inference global mean, implement the passing of `bayesian_global_mean` via manifest.
- [ ] Upon pipeline execution, confirm the `version_manifest.json` natively expanded its feature tracking to include `theme_{New_Topic}`.
- [ ] Run online verification to ensure the `DiggPredictionService` (Singleton) correctly detected the `manifest_path` timestamp shift and reloaded the Scaler + array dimension without downtime.
