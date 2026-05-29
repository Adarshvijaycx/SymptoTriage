# SymptoTriage — Interpretable Hierarchical Medical Symptom Classifier

A two-stage machine-learning pipeline that maps a patient's symptom set to one of **100 diseases**, with calibrated confidence, per-prediction explainability (SHAP + decision-path rules), drift monitoring, and a glassmorphism web dashboard.

> **Disclaimer:** This is a research and educational project. It is **not** a medical device and must not be used for real diagnosis or treatment decisions. Predictions, descriptions, and precautions (including auto-generated metadata) require review by a qualified clinician before any real-world use.

---

## What it does

Given a set of symptoms (each marked **present**, **confirmed absent**, or **unknown**), the system:

1. **Routes** the case into 1 of 14 organ-system categories (Stage-1).
2. **Classifies** the specific disease within that category using a calibrated soft-voting ensemble (Stage-2).
3. **Explains** the result — a human-readable decision path from the router plus a SHAP feature-impact chart from the ensemble.
4. **Reports** calibrated confidence, a sparsity penalty when few symptoms are given, plus disease description, precautions, and a symptom-severity breakdown.
5. **Monitors** input drift over time via PSI, flagging when retraining may be warranted.

---

## Architecture

```
 symptoms (JSON)
      │
      ▼
┌─────────────────┐   binary 0/1 matrix (230 symptom columns)
│ Encoder         │   +1 present · -1 confirmed-absent · 0 unknown
└─────────────────┘
      │  + derived features (symptom_count, 13 category counts, 12 interactions)
      ▼
┌─────────────────┐
│ Stage-1 Router  │   Pruned Decision Tree (cost-complexity pruning)
│                 │   → 1 of 14 organ-system categories
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ Stage-2 Ensemble│   one soft-voting ensemble per category
│  (per category) │   DecisionTree + RandomForest + LightGBM, weights [1,3,5]
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ Platt Calibrator│   sigmoid calibration on a held-out set (low ECE)
└─────────────────┘
      │
      ▼
  disease + calibrated confidence + SHAP + decision path
```

**Why two stages?** Splitting "which body system" from "which specific disease" keeps each model small and interpretable, and lets each category specialize. The router is a shallow, prunable tree whose path is readable; the per-category ensembles only ever discriminate between clinically-related diseases.

---

## Current performance (honest, held-out)

Trained on the `Diseases_and_Symptoms_dataset.csv` (96,088 rows, 100 diseases, 230 symptoms), evaluated on a ~19k-row stratified test split:

| Metric | Value |
|---|---|
| Stage-1 routing macro-F1 (14 categories) | **0.909** |
| Disease-level macro-F1 (100 classes) | **0.835** |
| Disease-level accuracy | **0.835** |
| Post-calibration ECE (target < 0.05) | **0.029** |

macro-F1 ≈ accuracy means performance is even across classes, not propped up by frequent ones. See [Dataset notes](#dataset-notes) for why the older dataset's "perfect" scores were misleading.

---

## Project structure

```
SymptoTriage/
├── config.yaml                  # single source of truth: dataset, category map, model hyperparams
├── Diseases_and_Symptoms_dataset.csv   # active dataset (100-disease binary matrix)
├── dataset.csv                  # legacy dataset (41-disease, name columns) — still supported
├── symptom_Description.csv      # disease → description    (100 diseases covered)
├── symptom_precaution.csv       # disease → precautions    (100 diseases covered)
├── Symptom-severity.csv         # symptom → severity weight (230 symptoms covered)
├── src/
│   ├── train.py                 # training entry point (dataset-format aware)
│   ├── pipeline/
│   │   ├── canonicalize.py      # data loaders (binary-matrix + legacy) & canonicalizers
│   │   ├── encoder.py           # 3-state encoder (+1/-1/0)
│   │   └── feature_eng.py       # derived features: counts, category counts, interactions
│   ├── models/
│   │   ├── stage1_router.py     # pruned decision-tree router
│   │   ├── stage2_ensemble.py   # per-category soft-voting ensembles
│   │   ├── calibrator.py        # Platt scaling + ECE metric
│   │   └── explainer.py         # SHAP + decision-path extraction
│   ├── monitoring/psi_monitor.py# population stability index drift monitor
│   └── api/
│       ├── main.py              # FastAPI app (CORS, endpoints)
│       ├── predict.py           # ModelService singleton + inference orchestration
│       └── schemas.py           # Pydantic request/response models
├── frontend/                    # vanilla HTML/CSS/JS dashboard (Chart.js for SHAP)
│   ├── index.html
│   ├── style.css
│   └── app.js
├── models/                      # serialized .pkl artifacts (produced by training)
├── notebooks/                   # EDA, augmentation, training, evaluation
├── tests/                       # pytest: encoder, augmentation, API
└── Dockerfile
```

---

## Setup

Requires Python 3.10+.

```bash
# from the project directory
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Note on versions:** `requirements.txt` lists pinned versions (e.g. scikit-learn 1.3.0), but the working environment was validated against newer releases (scikit-learn 1.8.0, lightgbm 4.6, shap 0.52, fastapi 0.136, pandas 3.0, numpy 2.4). The calibrator uses `sklearn.frozen.FrozenEstimator` (available in sklearn ≥ 1.6) with a fallback to the older `cv="prefit"` API, so both work. If you hit dependency conflicts, prefer the newer versions.

---

## Running it

### 1. Start the API (backend)

```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
# models load on startup; GET /health reports {"models_loaded": true} when ready
```

Pre-trained artifacts ship in `models/`, so you can serve immediately without retraining.

### 2. Serve the frontend

```bash
cd frontend
python -m http.server 5500
# open http://127.0.0.1:5500/index.html
```

Port 5500 (and 8000) are already in the API's CORS allowlist. Search symptoms, toggle each **present (+)** or **confirmed absent (−)**, and click **Analyze Case**.

### Docker

```bash
docker build -t symptotriage .
docker run -p 8000:8000 symptotriage    # serves the API only
```

---

## API reference

Base URL: `http://127.0.0.1:8000`

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness + whether models are loaded |
| GET | `/symptoms` | List of the 230 valid symptom names |
| GET | `/diseases` | List of the 100 disease names |
| GET | `/drift` | PSI drift status over buffered inference requests |
| POST | `/predict` | Run a diagnosis |

**Predict request:**

```json
{
  "symptoms": {
    "shortness of breath": 1,
    "wheezing": 1,
    "cough": 1,
    "chest tightness": -1
  }
}
```

`1` = present, `-1` = confirmed absent, omit a symptom for "unknown".

**Predict response (abridged):**

```json
{
  "disease": "asthma",
  "category": "Respiratory_ENT",
  "probability": 0.55,
  "decision_path": "IF ... THEN Routes To Category",
  "shap_values": {"wheezing": 0.21, "shortness of breath": 0.18, "...": 0.0},
  "disease_description": "A chronic condition in which the airways narrow ...",
  "disease_precautions": ["consult a healthcare professional", "avoid smoke ..."],
  "symptom_severities": {"shortness of breath": 6, "wheezing": 4},
  "latency_ms": 42.0
}
```

---

## Training

```bash
python -m src.train
```

Training reads everything from `config.yaml`, then writes 9 artifacts to `models/`: `encoder`, `feat_eng`, `cat_encoder`, `stage1`, `stage2`, `calibrator`, `explainer`, `psi`, and `canonicalizer`.

### Switching datasets (reversible)

The pipeline supports both dataset formats via one config switch:

```yaml
paths:
  dataset: Diseases_and_Symptoms_dataset.csv   # or dataset.csv
data:
  format: binary_matrix     # binary_matrix (new) | legacy_names (old)
  label_col: diseases       # label column for binary_matrix format
```

- `binary_matrix` — wide 0/1 matrix, label in column 0 (the 100-disease dataset).
- `legacy_names` — disease + symptom-name columns (the original 41-disease `dataset.csv`).

To revert to the old dataset: set `dataset: dataset.csv` + `format: legacy_names`, restore the old `disease_categories` map, and retrain. Backups of the pre-migration source files live in `.migration_backup/`, and the old 41-disease model artifacts in `models_old_41/`.

### Key config sections

- `disease_categories` — the 100-disease → 14-category map (Stage-1 routing targets).
- `stage1` / `stage2` — model hyperparameters.
- `augmentation` — symptom-dropout tiers and the `negative_evidence_rate` that synthesizes the `-1` signal (see caveats).
- `evaluation` / `monitoring` — target thresholds and PSI bins.

---

## Tests

```bash
pytest -q       # covers encoder, augmentation, and API
```

---

## Dataset notes

The project was migrated from a 41-disease dataset to a 100-disease one. A like-for-like comparison (identical model, 3-fold stratified CV) showed why:

| Dataset | Rows | Diseases | Macro-F1 (OOF) | Interpretation |
|---|---|---|---|---|
| `dataset.csv` (old) | 4,920 | 41 | **1.000** | Trivially separable — near-deterministic symptom templates; the model memorizes it. Not a sign of real skill. |
| `Diseases_and_Symptoms_dataset.csv` (new) | 96,088 | 100 | **0.876** | Genuinely learned — overlapping symptoms, realistic sparsity, ~20× the data. |

The old dataset's perfect score is the same triviality that produced the legacy "always predicts AIDS" overconfidence bug. The new dataset is a real learning problem the architecture handles well.

---

## Caveats & known limitations

- **Not medical advice.** Research/education only.
- **Auto-generated metadata.** Descriptions and precautions for the 92 diseases unique to the new dataset are concise, general-knowledge placeholders with conservative, generic precautions. They need clinician review before any real use. No drug/dosage advice is generated.
- **Synthetic negative evidence.** The new dataset has no true "confirmed absent" (`-1`) signal, so it is synthesized during training via dropout augmentation (`negative_evidence_rate`). The UI's present/absent toggle therefore reflects a learned-but-synthetic signal, not one observed in the source data.
- **No authentication.** The API is unauthenticated and intended for local use. Do not expose it publicly without adding auth and rate limiting.
- **Within-category confusion.** Clinically overlapping diseases (e.g. depression vs. schizophrenia sharing psychotic features) can be confused; calibrated confidence is correspondingly lower in those cases, which is the honest behavior.

---

## License

Add a license file before distributing. Source datasets retain their original licenses.
