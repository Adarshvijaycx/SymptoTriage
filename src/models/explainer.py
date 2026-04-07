"""Explanation utilities for decision paths and SHAP values.

Combines rule extraction from Stage-1 Routing Trees with fast
TreeSHAP approximation over the LightGBM portion of the Stage-2 ensemble.
"""

from typing import Dict, List, Any
import numpy as np
import shap
from sklearn.tree import DecisionTreeClassifier

from src.models.stage1_router import Stage1Router
from src.models.stage2_ensemble import Stage2Ensemble


class PredictionExplainer:
    """Manages explainability artifacts for a prediction.

    Parameters
    ----------
    router : Stage1Router
        The Stage-1 coarse router.
    ensemble : Stage2Ensemble
        The Stage-2 fine-grained ensemble.
    feature_names : list of str
        The full feature list to match indices to human-readable strings.
    """

    def __init__(self, router: Stage1Router, ensemble: Stage2Ensemble, feature_names: List[str]):
        self.router = router
        self.ensemble = ensemble
        self.feature_names = feature_names
        
        # Maps Stage-1 category ID -> SHAP TreeExplainer for the LightGBM sub-model
        self.shap_explainers_: Dict[int, shap.TreeExplainer] = {}

    def fit_shap(self, X_background: np.ndarray, categories: np.ndarray):
        """Fit TreeSHAP explainers over the LightGBM base models.

        Note: We extract the LightGBM model from the fitted VotingClassifier
        because it dominates the soft voting [weight=3] and supports
        lightning-fast exact TreeSHAP.
        """
        for cat_id, voting_clf in self.ensemble.ensembles_.items():
            mask = categories == cat_id
            X_bg_cat = X_background[mask]

            if len(X_bg_cat) == 0:
                continue

            # Limit background size for speed
            if len(X_bg_cat) > 500:
                # Deterministic subsample
                np.random.seed(42)
                idx = np.random.choice(len(X_bg_cat), 500, replace=False)
                X_bg_cat = X_bg_cat[idx]

            # Attempt to extract LightGBM from the soft voting ensemble.
            lgb_model = None
            if hasattr(voting_clf, "named_estimators_") and "lgb" in voting_clf.named_estimators_:
                lgb_model = voting_clf.named_estimators_["lgb"]
                
            if lgb_model is not None:
                # Using feature_perturbation="tree_path_dependent" allows speedup
                # and bypasses the strict requirement for a background dataset 
                # strictly on LGBM, but passing Data builds a clearer marginal mapping.
                try:
                    explainer = shap.TreeExplainer(lgb_model)
                    self.shap_explainers_[cat_id] = explainer
                except Exception as e:
                    print(f"[SHAP] Explainer init failed for category {cat_id}: {e}")
                    
        return self

    def explain_prediction(self, X_single: np.ndarray, category_id: int, 
                           pred_idx: int) -> Dict[str, Any]:
        """Extract both types of explanations.

        Parameters
        ----------
        X_single : np.ndarray, shape (1, n_features)
            Patient feature vector.
        category_id : int
            The Stage-1 category ID.
        pred_idx : int
            The max-confidence integer class index within the category.

        Returns
        -------
        dict
            {
                "decision_path": str, 
                "shap_values": Dict[str, float]
            }
        """
        # 1. Decision path
        path = self.router.get_decision_path(X_single, self.feature_names)

        # 2. SHAP Values
        shap_dict = {}
        if category_id in self.shap_explainers_:
            explainer = self.shap_explainers_[category_id]
            try:
                # Map global pred_idx to local index within this category's ensemble
                voting_clf = self.ensemble.ensembles_.get(category_id)
                local_idx = 0
                if voting_clf is not None and hasattr(voting_clf, "classes_"):
                    idx_arr = np.where(voting_clf.classes_ == pred_idx)[0]
                    if len(idx_arr) > 0:
                        local_idx = idx_arr[0]
                
                # TreeSHAP outputs shape (1, n_features, n_classes) for multiclass LGBM
                # Or a list of (1, n_features) arrays
                shap_vals = explainer.shap_values(X_single)
                
                # Handle varying shap return types based on lightgbm/shap version
                if isinstance(shap_vals, list):
                    if local_idx < len(shap_vals):
                        vals = shap_vals[local_idx][0]
                    else:
                        vals = shap_vals[0][0]
                elif len(shap_vals.shape) == 3:
                    if local_idx < shap_vals.shape[2]:
                        vals = shap_vals[0, :, local_idx]
                    else:
                        vals = shap_vals[0, :, 0]
                else:
                    vals = shap_vals[0]
                
                # Filter down to top-15 most impactful symptoms
                ranked = sorted(
                    zip(self.feature_names, vals),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )
                
                # Exclude purely zero SHAP features
                shap_dict = {
                    feat: round(float(val), 4) 
                    for feat, val in ranked[:15] if abs(val) > 1e-5
                }
            except Exception as e:
                print(f"[SHAP Error] Could not compute SHAP: {e}")

        return {
            "decision_path": f"IF {path} THEN Routs To Category",
            "shap_values": shap_dict
        }
