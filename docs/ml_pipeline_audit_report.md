# DeepTikTok ML Pipeline: High-Granularity Technical Audit Report

## 1. Data Alignment & Synchronization Logic
**Cross-Modal Data Merging:**  
The integration of heterogeneous sources resides within `data_fusion_and_cleaning.py`. The visual/audio feature tensor from the local hardware and the textual metadata from the web crawler are aligned using `视频ID` (Video ID) as the standard join key. A left outer join (`pd.merge(v_df, comment_agg, on='视频ID', how='left')`) anchors the video record as the primary entity context.

**Handling Null Values and Mismatched Dimensions:**  
- **Missing Web Comments:** Samples lacking crawled comments have their `avg_comment_likes` implicitly populated with `0` via standard zero-imputation (`fillna(0)`), retaining the principal video vectors.  
- **Missing Physical Files:** `feature_factory_v2.py` triggers an exception-handling fallback for missing local files, generating a baseline zero-vector `[0.0, 0.0, 0.0, 110.0]` preserving the dimensional congruity of the matrix format.  
- **Training Pipeline Protections:** Within `train_master_arena.py`, continuous feature variables utilize dynamic subset median imputation decoupled geometrically across train/test splits, anchored globally by a resilient `default_fallback` dictionary against total nan-degradation.

## 2. Feature Quantization & Normalization
**Visual Quantization (PyTorch):**  
Frame metrics are constrained linearly ($n \le 500$ frames, stride modulo sequence of 5).
- **Luminance & Saturation:** Computed natively upon GPU architecture using standard mathematical reduction. Brightness projects via mean calculation (`torch.mean`). Saturation is extracted contextually through max-min pooling across the RGB spatial channels.  
- **Shot-Cut Frequency:** Identified spatially out-of-core on the CPU via frame-sequential Chi-Square histogram profiling (`cv2.compareHist`). Peaks transcending standard thresholds (`> 10000`) count as cut permutations, which are subsequently aggregated and normalized over the video length scalar.  

**Audio Quantization (Librosa):**  
Temporally bounded sequential waveforms (30-second slice) stream into a static local memory space. Using target temporal beat extraction (`librosa.beat.beat_track`), the continuous spectrum is abstracted directly down into a standalone mathematical metric (BPM). 

**Text Quantization (SnowNLP):**  
Discrete NLP polarity strings convert upstream; the core arena script processes the downstream interval constraint variables through spatial aggregation (`Avg('comments__sentiment_score')`), consolidating user discourse into a $[0, 1]$ numerical density factor.

**Normalization Pipeline:**  
- **Target Normalization:** The prediction axis undergoes right-tail truncation down to the `cap_quantile` (95%), followed strictly by a logarithmic decay transformation `np.log1p(y)`, isolating target variables against viral extreme sparsity logic.  
- **Independent Feature Scaling:** Predictors distribute computationally via `StandardScaler`, localized stringently isolated inside the training structure to definitively eliminate metric leakage.

## 3. "Model Arena" Training Workflow
**Train-Test Split Strategy:**  
`train_master_arena.py` structures a strict partitioning topology bounded by `GLOBAL_CONFIG['test_size'] = 0.15` (equivalent approximately logic to an 85/15 algorithmic split, representing a variant of the classical 80/20 standard). It also instantiates a 'biased sampling' paradigm for specific themes (Target/Foil), mathematically increasing the representation of specific local target parameters while separating testing verification logically to retain global accuracy constraints.

**Algorithmic Tree-Ensemble Rationales:**  
Non-linear viral propagation maps poorly via planar geometric regressions.  
- **Random Forest:** Limits high-variance viral overfitting directly across orthogonal subset trees through bootstrap aggregation techniques.  
- **LightGBM:** Inherently incorporates Exclusive Feature Bundling, operating swiftly across categorical binary tree variables with exceptionally minimized VRAM allocations via gradient-based one-side sampling.  
- **XGBoost:** Projects higher operational capabilities over sparse multimodal bounds (such as binary encoded NLP indicators matched against variable sequential frame metrics), guided firmly by a distinct regularization vector parameter ($\Omega$) preventing terminal node over-branching.

**Key Metric Academic Substrates:**  
- **RMSE (Root Mean Squared Error):** Serves fundamentally to amplify sensitivity to extreme algorithmic failures across high-viral-yield samples, penalizing non-linear overestimations quadratically.
- **$R^2$:** Standard deviation normalization benchmark defining precisely the ratio of operational target volatility properly abstracted by the model against a naïve dataset mean.
- **MAE:** Although absent in explicit pipeline execution, it represents absolute baseline expected metric margins isolated away from exponential magnitude skew mathematically, serving distinct business evaluation boundaries.

## 4. VRAM & Resource Management
**Process-and-Destroy Operational Lifecycle:**  
`feature_factory_v2.py` isolates memory structurally through batch synchronization logic. Hardware resources decode local assets linearly. Tensors instantiated statically via `torch.from_numpy` reside in memory solely for singular mathematical projection constraints, and loop iterations invoke internal object garbage collection rapidly.

**12GB VRAM Architecture Integrity:**  
OOM events on the target hardware (RTX 3060, 12GB VRAM) are mathematically prevented by avoiding loading an entire multidimensional sequential stream block entirely into RAM/VRAM matrices.
Calculations are isolated structurally to bounded temporal single-frame calculations (`max_frames=500`). 
Simultaneously, downstream boosting architectures enforce VRAM conservation using local geometric abstractions via the `tree_method='hist'` construct inherently mapped for exact minimal block memory deployment (`device='cuda'`).
