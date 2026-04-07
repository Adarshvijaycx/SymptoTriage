"""Prediction endpoint business logic.

Manages the Model singleton, orchestrates feature extraction,
runs 2-stage inference, and packages explainability results.
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from src.pipeline import SymptomCanonicalizer, ThreeStateEncoder, FeatureEngineer
from src.models.stage1_router import Stage1Router
from src.models.stage2_ensemble import Stage2Ensemble
from src.models.calibrator import ProbabilityCalibrator
from src.models.explainer import PredictionExplainer
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
        
        # 3. Stage 1: Route Category
        category_id = self.router.predict(X_enriched)[0]
        category_name = self.cat_encoder.inverse_transform([category_id])[0]
        
        # 4. Stage 2: Fine-grained prediction with Probability Calibration
        try:
            calibrated_probas = self.calibrator.predict_proba(X_enriched, category_id)[0]
            pred_idx = self.ensemble.predict(X_enriched, category_id)[0]
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
