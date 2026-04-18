# ML Data Strategy Contract

## Status
This document is the repository-local source of truth for theme-data organization and utilization.

It codifies the target contract for later remediation steps without changing runtime behavior in this step.

## Core Rules

### 1. Training uses all valid themes
- Training continues to learn from all valid themes after normal data-quality filtering.
- Non-current themes are part of the shared training distribution and provide global priors, topology coverage, and transfer signal.
- The current or active theme may guide sampling emphasis or evaluation focus, but it does not redefine the repository as a single-theme training system.

### 2. Online prediction uses the current theme only
- Request-time inference should only care about the requested/current theme plus version-owned global priors.
- The online path must not dynamically recompute statistics for unrelated themes during a prediction request.

### 3. Display compares against the current theme only
- Percentile explanations, benchmark ranges, and publishing-time suggestions are display-layer outputs for the current theme only.
- These presentation statistics help interpret a prediction; they do not define model-input preprocessing truth.

### 4. Model-input stats and display stats stay separate
- Model-input preprocessing metadata belongs to the trained model version and will later live in version-owned artifacts such as `prep_{version}.json` plus the manifest/scaler/model bundle.
- Display-only theme baselines are UI/explanation statistics for the current theme.
- These two data families are intentionally separate and must not be mixed.

### 5. Request-time inference avoids wasteful global recomputation
- The long-term contract does not allow full-database recomputation for every prediction request.
- Runtime prediction should move toward cached/precomputed current-theme baselines and version-owned preprocessing metadata instead of ad hoc global scans.

### 6. Full bundle versions own preprocessing truth
- Future preprocessing truth belongs to the versioned model bundle.
- The authoritative bundle concept is:
  - `model_{version}.pkl`
  - `scaler_{version}.pkl`
  - `prep_{version}.json`
  - `version_manifest.json`
- Request-time estimation is not the long-term source of truth for model-input preprocessing.

## Source Of Truth
- Training-time preprocessing truth belongs to versioned artifacts emitted by the training pipeline in later steps.
- Display-time theme baselines are not the source of truth for model input.
- Runtime prediction should not depend on recomputing unrelated-theme statistics.
- When there is tension between a display baseline and a version bundle, the version bundle wins for model-input preprocessing.

## Responsibility Map

### `ml_pipeline/train_master_arena.py`
- Owns training over all valid themes.
- Owns generation of training-derived preprocessing truth and global priors.
- Will later emit version-owned preprocessing metadata for inference consumption.

### `services/predict_service.py`
- Owns version loading, bundle-awareness, and online inference execution.
- Will later consume version-owned preprocessing metadata from the loaded bundle.
- Should treat current-theme display baselines as presentation context, not as long-term preprocessing authority.

### `ml_pipeline/theme_baseline_engine.py`
- Owns display/explanation statistics for the current theme.
- Produces theme baselines, percentile ranges, and publishing-time hints for UI interpretation.
- Does not own authoritative model-input preprocessing truth.

### `renhangxi_tiktok_bysj/douyin_hangxi/views.py`
- Owns request orchestration and presentation wiring.
- Passes current-theme context through to prediction, UI, and LLM explanation layers.
- Should move away from request-time full-database recomputation toward cached/precomputed display baselines in later steps.

## Current Transitional Notes
- Some current code paths still mix request-time theme baseline data into inference-time feature generation for backward compatibility.
- That legacy coupling is tolerated only until later steps move model-input statistics fully into version-owned preprocessing metadata.
- Step 3 defines the contract now so later work can converge on one direction without strategy drift.

## Deferred To Later Steps
- Rewriting training feature generation.
- Emitting `prep_{version}.json` from training.
- Switching `predict_service.py` feature building to consume `self.prep`.
- Reworking request-time baseline retrieval to avoid full-database recomputation.
- Full bundle rollback and bundle-consistent fallback.
- Manifest schema changes and artifact producer changes.
