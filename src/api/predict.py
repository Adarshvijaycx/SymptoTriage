"""Prediction endpoint business logic.

Manages the Model singleton, orchestrates feature extraction,
runs 2-stage inference, and packages explainability results.
"""

import os
import time
import joblib
import yaml
import numpy as np
import pandas as pd
from collections import deque
from typing import Dict, Any, List

from src.pipeline import SymptomCanonicalizer, ThreeStateEncoder, FeatureEngineer
from src.models.stage1_router import Stage1Router
from src.models.stage2_ensemble import Stage2Ensemble
from src.models.calibrator import ProbabilityCalibrator
from src.models.explainer import PredictionExplainer
from src.monitoring.psi_monitor import PSIMonitor
from src.api.schemas import PredictRequest, PredictResponse


class ModelService:
    """Singleton service wrapping the loaded pipeline attributes."""
    
    _instance = None
    
    def __new__(cls, models_dir: str = "models"):
        if cls._instance is None:
            cls._instance = super(ModelService, cls).__new__(cls)
            cls._instance._load_models(models_dir)
        return cls._instance
        
    def _load_models(self, models_dir: str):
        """Load all saved PKL artifacts from training."""
        try:
            self.encoder: ThreeStateEncoder = joblib.load(os.path.join(models_dir, "encoder.pkl"))
            self.feat_eng: FeatureEngineer = joblib.load(os.path.join(models_dir, "feat_eng.pkl"))
            self.cat_encoder = joblib.load(os.path.join(models_dir, "cat_encoder.pkl"))
            self.router: Stage1Router = joblib.load(os.path.join(models_dir, "stage1.pkl"))
            self.calibrator: ProbabilityCalibrator = joblib.load(os.path.join(models_dir, "calibrator.pkl"))
            self.explainer: PredictionExplainer = joblib.load(os.path.join(models_dir, "explainer.pkl"))
            # Keep reference to the uncalibrated ensemble for fallback class extraction
            self.ensemble: Stage2Ensemble = joblib.load(os.path.join(models_dir, "stage2.pkl"))

            # Load Metadata CSVs
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

            # PSI drift monitor (optional — absence must not break inference)
            self.psi_monitor: PSIMonitor = None
            try:
                self.psi_monitor = joblib.load(os.path.join(models_dir, "psi.pkl"))
            except Exception as e:
                print(f"PSI monitor not loaded ({e}); drift checks disabled.")

            # Rolling buffer of enriched inference vectors for batch PSI computation.
            # PSI is a batch statistic (needs >=50 samples), so we accumulate live
            # requests here and compute drift on demand via check_drift().
            self._psi_buffer: deque = deque(maxlen=2000)

            # Drift thresholds from config.yaml (fall back to PSI conventions)
            self.psi_warn_threshold = 0.10
            self.psi_retrain_threshold = 0.20
            try:
                cfg_path = os.path.join(root_dir, "config.yaml")
                with open(cfg_path, "r") as f:
                    mon_cfg = (yaml.safe_load(f) or {}).get("monitoring", {})
                self.psi_warn_threshold = mon_cfg.get("psi_threshold_warning", 0.10)
                self.psi_retrain_threshold = mon_cfg.get("psi_threshold_retrain", 0.20)
            except Exception:
                pass
            
            try:
                desc_df = pd.read_csv(os.path.join(root_dir, "symptom_Description.csv"))
                self.descriptions = dict(zip(desc_df.iloc[:, 0].str.strip(), desc_df.iloc[:, 1].str.strip()))
            except Exception:
                self.descriptions = {}

            try:
                prec_df = pd.read_csv(os.path.join(root_dir, "symptom_precaution.csv"))
                self.precautions = {}
                for _, row in prec_df.iterrows():
                    disease = str(row.iloc[0]).strip()
                    precs = [str(x).strip() for x in row.iloc[1:] if pd.notna(x) and str(x).strip()]
                    self.precautions[disease] = precs
            except Exception:
                self.precautions = {}

            try:
                sev_df = pd.read_csv(os.path.join(root_dir, "Symptom-severity.csv"))
                self.severities = dict(zip(sev_df.iloc[:, 0].str.strip(), sev_df.iloc[:, 1]))
            except Exception:
                self.severities = {}

        except FileNotFoundError as e:
            print(f"Failed to load models. Run training first. Error: {e}")
            raise e

    def get_valid_symptoms(self) -> List[str]:
        return self.encoder.get_feature_names()

    def get_valid_diseases(self) -> List[str]:
        return self.encoder.get_disease_names()

    # Minimum number of symptoms expected for full confidence
    MIN_SYMPTOMS_FOR_FULL_CONFIDENCE = 3

    def run_prediction(self, req: PredictRequest) -> PredictResponse:
        """Run the end-to-end inference pipeline for a request."""
        start_time = time.perf_counter()
        
        # 1. 3-State encode raw string dict -> 1D binary vector
        X_base = self.encoder.encode_inference_input(req.symptoms)
        
        # Count how many symptoms the patient actually provided
        active_symptom_count = sum(1 for v in req.symptoms.values() if v == 1.0)
        
        # 2. Extract derived counts -> continuous
        X_enriched = self.feat_eng.transform(X_base)

        # Buffer this enriched vector for batch drift (PSI) monitoring.
        if self.psi_monitor is not None:
            self._psi_buffer.append(X_enriched[0].astype(np.float32))

        # 3. Stage 1: Route Category
        category_id = self.router.predict(X_enriched)[0]
        category_name = self.cat_encoder.inverse_transform([category_id])[0]
        
        # 4. Stage 2: Fine-grained prediction with Probability Calibration
        try:
            calibrated_probas = self.calibrator.predict_proba(X_enriched, category_id)[0]
            # Derive the label from the SAME distribution we report confidence over.
            # The calibrated classifier's column order is given by its classes_, so
            # argmax(calibrated_probas) and that label always describe one disease.
            cal_clf = self.calibrator.calibrated_ensembles_[category_id]
            pred_idx = cal_clf.classes_[int(np.argmax(calibrated_probas))]
        except Exception as e:
             import traceback
             traceback.print_exc()
             # Use the most common class in the category rather than index 0
             ens = self.ensemble.ensembles_.get(category_id)
             if ens is not None and hasattr(ens, 'classes_'):
                 pred_idx = ens.classes_[0]
             else:
                 pred_idx = 0
             calibrated_probas = np.array([1.0 / max(len(self.encoder.get_disease_names()), 1)])
        
        raw_confidence = float(np.max(calibrated_probas))
        
        # 5. Sparsity-aware confidence scaling
        # With <3 symptoms, scale down confidence proportionally
        sparsity_penalty = min(1.0, active_symptom_count / self.MIN_SYMPTOMS_FOR_FULL_CONFIDENCE)
        confidence = raw_confidence * sparsity_penalty
        
        disease_name = self.encoder.decode_label(pred_idx)
        
        # 5. Explainer block
        explanations = self.explainer.explain_prediction(X_enriched, category_id, pred_idx)
        
        latency = (time.perf_counter() - start_time) * 1000  # ms
        
        # Determine metadata mappings
        # Use canonicalizer logic or exact matches to find description and precautions
        fallback_desc = "No detailed description available for this disease."
        actual_disease_str = disease_name
        
        # Some simple heuristics to match names loosely
        desc = self.descriptions.get(actual_disease_str)
        if not desc:
            # try case-insensitive
            desc_map_lower = {k.lower(): v for k, v in self.descriptions.items()}
            desc = desc_map_lower.get(actual_disease_str.lower(), fallback_desc)

        # A NaN value is truthy, so it slips past `if not desc`; coerce any
        # non-string / NaN result to the fallback so we never return NaN.
        if not isinstance(desc, str) or not desc.strip():
            desc = fallback_desc
            
        precs = self.precautions.get(actual_disease_str)
        if not precs:
            prec_map_lower = {k.lower(): v for k, v in self.precautions.items()}
            precs = prec_map_lower.get(actual_disease_str.lower(), [])
            
        active_symps = {k: v for k, v in req.symptoms.items() if v == 1.0}
        sev_dict = {}
        sev_map_lower = {k.lower(): v for k, v in self.severities.items()}
        for symp in active_symps:
            # e.g., symp might be "high_fever" but severity dataset has "high_fever" or "high fever"
            val = sev_map_lower.get(symp.lower(), 0)
            sev_dict[symp] = val

        return PredictResponse(
            disease=disease_name,
            category=category_name,
            probability=confidence,
            decision_path=explanations["decision_path"],
            shap_values=explanations["shap_values"],
            disease_description=desc,
            disease_precautions=precs,
            symptom_severities=sev_dict,
            latency_ms=round(latency, 2)
        )

    def check_drift(self) -> Dict[str, Any]:
        """Compute PSI drift over buffered inference requests vs. training.

        PSI is a batch statistic, so this aggregates the rolling buffer of
        enriched inference vectors and compares their distribution against the
        training distribution captured by the fitted PSIMonitor.

        Returns
        -------
        dict with keys:
            status: 'disabled' | 'insufficient_data' | 'ok' | 'warning' | 'retrain'
            n_samples: number of buffered requests
            max_psi / mean_psi: summary statistics over per-feature PSI
            top_drift: list of {feature, psi} for the highest-drift features
        """
        if self.psi_monitor is None:
            return {"status": "disabled", "n_samples": 0}

        n = len(self._psi_buffer)
        # PSIMonitor.compute_psi requires a minimal batch (returns {} below 50)
        if n < 50:
            return {"status": "insufficient_data", "n_samples": n,
                    "min_required": 50}

        X_batch = np.vstack(list(self._psi_buffer))
        psi_scores = self.psi_monitor.compute_psi(X_batch)
        if not psi_scores:
            return {"status": "insufficient_data", "n_samples": n}

        feature_names = self.feat_eng.get_all_feature_names()

        def _fname(i: int) -> str:
            return feature_names[i] if i < len(feature_names) else f"feature_{i}"

        max_psi = max(psi_scores.values())
        mean_psi = float(np.mean(list(psi_scores.values())))

        if max_psi >= self.psi_retrain_threshold:
            status = "retrain"
        elif max_psi >= self.psi_warn_threshold:
            status = "warning"
        else:
            status = "ok"

        top = sorted(psi_scores.items(), key=lambda kv: kv[1], reverse=True)[:10]
        return {
            "status": status,
            "n_samples": n,
            "max_psi": round(float(max_psi), 4),
            "mean_psi": round(mean_psi, 4),
            "warn_threshold": self.psi_warn_threshold,
            "retrain_threshold": self.psi_retrain_threshold,
            "top_drift": [{"feature": _fname(i), "psi": round(float(p), 4)}
                          for i, p in top],
        }
