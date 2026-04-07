# Medical Diagnosis Pipeline — 4-Phase Implementation Plan

## Overview

The project is a **two-stage hierarchical disease classifier** with explainability, served via FastAPI. The skeleton exists — all source files are empty placeholders. The dataset (`dataset.csv`) contains 4,920 records across 41 diseases (120 per disease), with symptoms as comma-separated text per row (up to 17 symptom columns).

---

## Phase 1 — Data Processing Layer (25%)

> **Goal**: Load raw CSV → canonicalize symptoms → build binary feature matrix → 3-state encode → save processed data

### [MODIFY] [canonicalize.py](file:///Users/adarsh.vijay/Desktop/Projects/Classifier%20AX/src/pipeline/canonicalize.py)
- `SYNONYM_MAP` dictionary mapping alternate symptom names to canonical forms
- `SymptomCanonicalizer` class with `fit()` (learns unique symptoms from dataset) and `transform()` (applies synonym map, strips whitespace, deduplicates)
- Handles edge cases: leading spaces in CSV, inconsistent casing (`hepatitis A` vs `Hepatitis A`), trailing whitespace on symptom values

### [MODIFY] [encoder.py](file:///Users/adarsh.vijay/Desktop/Projects/Classifier%20AX/src/pipeline/encoder.py)
- `ThreeStateEncoder` class:
  - `fit(df)` — discovers all unique symptom tokens across all rows
  - `transform(df)` → binary matrix (N × S) where:
    - `+1` = symptom present in that patient record
    - `0` = symptom not mentioned (unknown / not queried)
  - Note: dataset has no explicit "absent" markers, so the 3-state encoding will use `+1` (present) and `0` (not recorded). The `-1` (confirmed absent) state will be used at **inference time** when a doctor explicitly marks a symptom as absent.
  - Returns `pd.DataFrame` with symptom column names and disease label column

### [MODIFY] [feature_eng.py](file:///Users/adarsh.vijay/Desktop/Projects/Classifier%20AX/src/pipeline/feature_eng.py)
- `FeatureEngineer` class:
  - `symptom_count` — number of symptoms present per patient (count of +1s)
  - `category_counts` — count of symptoms per disease category group (respiratory symptoms, GI symptoms, etc.)
  - These become additional continuous features alongside the binary symptom matrix

### [MODIFY] [config.yaml](file:///Users/adarsh.vijay/Desktop/Projects/Classifier%20AX/config.yaml)
- Add `disease_categories` mapping: each of the 41 diseases assigned to one of 4 categories (Respiratory, Metabolic/Endocrine, Infectious, Musculoskeletal/Neurological)
- Add `stage1` and `stage2` hyperparameter sections
- Add `augmentation` config section

---

## Phase 2 — Augmentation + Two-Stage Classifier (25%)

> **Goal**: Implement SMOTE/CTGAN augmentation, Stage-1 pruned DT router, Stage-2 ensemble, and the main `train.py` that orchestrates everything

### [MODIFY] [smote_handler.py](file:///Users/adarsh.vijay/Desktop/Projects/Classifier%20AX/src/augmentation/smote_handler.py)
- `SMOTEHandler` class wrapping `imblearn.over_sampling.SMOTENC`
- `fit_resample(X, y)` method with configurable `k_neighbors` and `sampling_strategy`

### [MODIFY] [ctgan_handler.py](file:///Users/adarsh.vijay/Desktop/Projects/Classifier%20AX/src/augmentation/ctgan_handler.py)
- `CTGANHandler` class wrapping `ctgan.CTGAN`
- `fit(X_majority)` and `sample(n)` methods
- Configurable `epochs`, `batch_size`

### [MODIFY] [hybrid.py](file:///Users/adarsh.vijay/Desktop/Projects/Classifier%20AX/src/augmentation/hybrid.py)
- `HybridAugmentor` class:
  1. SMOTE-NC on minority classes
  2. CTGAN on majority classes (generates 25% more synthetic records)
  3. Tomek link cleaning to sharpen decision boundaries
- `fit_resample(X, y)` → returns (X_augmented, y_augmented)

### [MODIFY] [stage1_router.py](file:///Users/adarsh.vijay/Desktop/Projects/Classifier%20AX/src/models/stage1_router.py)
- `Stage1Router` class:
  - `fit(X, y_category)` — trains a `DecisionTreeClassifier` with CCP pruning
  - Sweeps `ccp_alpha` logarithmically, selects optimal via 3-fold CV on Macro-F1
  - `max_depth` constrained to 4–8
  - `predict(X)` → category labels
  - `get_decision_path(X_single)` → human-readable IF-THEN rule string

### [MODIFY] [stage2_ensemble.py](file:///Users/adarsh.vijay/Desktop/Projects/Classifier%20AX/src/models/stage2_ensemble.py)
- `Stage2Ensemble` class:
  - Trains one `VotingClassifier(DT, RF, LightGBM, voting='soft')` per disease category
  - Weights: `[1, 2, 3]` (LightGBM weighted highest)
  - `fit(X, y, categories)` — trains per-category ensembles
  - `predict(X, category)` and `predict_proba(X, category)` methods

### [MODIFY] [train.py](file:///Users/adarsh.vijay/Desktop/Projects/Classifier%20AX/src/train.py)
- End-to-end orchestration:
  1. Load CSV → canonicalize → encode → feature engineer
  2. Apply augmentation (optional, controlled via config)
  3. Stratified train/test split (80/20)
  4. Train Stage-1 router on category labels
  5. Train Stage-2 ensembles per category
  6. Evaluate metrics (Macro-F1, per-class recall)
  7. Save all models to `models/` as `.pkl` files
  8. Print classification reports

---

## Phase 3 — Calibration, Explainability & Monitoring (25%)

> **Goal**: Add Platt scaling calibration, SHAP + decision path explainability, PSI drift monitor

### [MODIFY] [calibrator.py](file:///Users/adarsh.vijay/Desktop/Projects/Classifier%20AX/src/models/calibrator.py)
- `ProbabilityCalibrator` class:
  - `fit(model, X_cal, y_cal)` — wraps model with `CalibratedClassifierCV(method='sigmoid', cv='prefit')`
  - `calibrate(raw_proba)` → calibrated probabilities
  - `compute_ece(y_true, y_proba, n_bins=10)` — Expected Calibration Error metric

### [MODIFY] [explainer.py](file:///Users/adarsh.vijay/Desktop/Projects/Classifier%20AX/src/models/explainer.py)
- `PredictionExplainer` class:
  - `fit_shap(model, X_background)` — creates `shap.TreeExplainer`
  - `explain(X_single)` → dict of `{feature_name: shap_value}` for top-15 features
  - `extract_decision_path(tree_model, X_single, feature_names)` → IF-THEN rule string from a decision tree

### [MODIFY] [psi_monitor.py](file:///Users/adarsh.vijay/Desktop/Projects/Classifier%20AX/src/monitoring/psi_monitor.py)
- `PSIMonitor` class:
  - `fit(X_train)` — stores training distribution per feature (binned)
  - `compute_psi(X_new)` → dict of `{feature: psi_value}`
  - `check_drift(X_new, threshold=0.2)` → list of drifted features
  - `report()` → summary dict with status per feature

### Update `train.py`:
  - After training, fit calibrator on held-out calibration set
  - Compute and print ECE before/after calibration
  - Fit SHAP explainer on training background data
  - Fit PSI monitor on training data
  - Save calibrator, explainer, PSI monitor as `.pkl` artifacts

---

## Phase 4 — FastAPI Deployment, Tests & Docker (25%)

> **Goal**: Connect everything via REST API, write tests, finalize Dockerfile

### [MODIFY] [schemas.py](file:///Users/adarsh.vijay/Desktop/Projects/Classifier%20AX/src/api/schemas.py)
- `PredictRequest` — `symptoms: Dict[str, float]` (e.g. `{"fever": 1, "cough": -1}`)
- `PredictResponse` — `disease`, `probability`, `decision_path`, `shap_values`, `latency_ms`

### [MODIFY] [predict.py](file:///Users/adarsh.vijay/Desktop/Projects/Classifier%20AX/src/api/predict.py)
- `ModelService` singleton:
  - Loads all `.pkl` models at startup
  - `run_prediction(symptoms: dict)` → executes full pipeline:
    1. Build feature vector from symptom dict (using stored symptom list)
    2. Stage-1 → get category
    3. Stage-2 → get disease + raw probability
    4. Calibrate probability
    5. Generate decision path + SHAP explanations
    6. Return `PredictResponse` dict with latency

### [MODIFY] [main.py](file:///Users/adarsh.vijay/Desktop/Projects/Classifier%20AX/src/api/main.py)
- Add `POST /predict` endpoint using schemas
- Add `GET /symptoms` endpoint returning list of valid symptom names
- Add `GET /diseases` endpoint returning list of diseases by category
- CORS middleware for frontend integration

### [MODIFY] [test_encoder.py](file:///Users/adarsh.vijay/Desktop/Projects/Classifier%20AX/tests/test_encoder.py)
- Test 3-state encoding: present, absent, unknown
- Test canonicalization: synonym mapping, whitespace handling
- Test feature matrix shape and values

### [MODIFY] [test_augmentation.py](file:///Users/adarsh.vijay/Desktop/Projects/Classifier%20AX/tests/test_augmentation.py)
- Test SMOTE handler: output shape, class balance
- Test hybrid augmentor end-to-end

### [MODIFY] [test_api.py](file:///Users/adarsh.vijay/Desktop/Projects/Classifier%20AX/tests/test_api.py)
- Test `/health` endpoint
- Test `/predict` with sample symptoms
- Test `/symptoms` and `/diseases` endpoints
- Test error handling (invalid symptoms, empty request)

### [MODIFY] [Dockerfile](file:///Users/adarsh.vijay/Desktop/Projects/Classifier%20AX/Dockerfile)
- Ensure models are copied
- Health check instruction
- Non-root user for security

---

## Execution Order

| Phase | What Gets Built | Deliverable |
|-------|----------------|-------------|
| **1** | Data pipeline: load CSV → canonicalize → encode → feature matrix | `data/processed/` output files, runnable encoder |
| **2** | Augmentation + classifiers → trained models | `models/*.pkl` files, training script that runs end-to-end |
| **3** | Calibration + SHAP + PSI monitoring | Enhanced models with calibrated probabilities, explanations |
| **4** | REST API + tests + Docker | Working `POST /predict` endpoint, passing tests |

---

## Open Questions

> [!IMPORTANT]
> **Augmentation Strategy**: Since this dataset is **already perfectly balanced** (120 records per disease), SMOTE/CTGAN augmentation won't change class balance. Should I:
> 1. Still implement the full augmentation pipeline for demonstration/portfolio purposes (it will be a configurable flag that defaults to OFF for this dataset)?
> 2. Skip augmentation entirely?
> 
> **Recommendation**: Option 1 — implement it but default to OFF, so the code is complete for your project presentation.

> [!NOTE]
> **CTGAN dependency**: The `ctgan` package pulls in heavy PyTorch dependencies. For this clean dataset it won't improve results. I'll implement the handler but keep it optional. Training will work with or without CTGAN installed.

## Verification Plan

### Automated Tests
```bash
# After Phase 1
python -c "from src.pipeline.encoder import ThreeStateEncoder; print('Encoder OK')"

# After Phase 2
python src/train.py  # Full training pipeline — should print F1 ~1.0

# After Phase 3
python src/train.py  # Should also print ECE metrics

# After Phase 4
pytest tests/ -v  # All tests pass
uvicorn src.api.main:app --port 8000  # API starts, test via Swagger
```

### Manual Verification
- curl `POST /predict` with sample symptoms → verify disease + confidence + explanation
- Check model files exist in `models/`
- Verify decision path output is human-readable
