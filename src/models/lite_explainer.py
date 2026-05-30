"""Lightweight explainer for resource-constrained runtimes (e.g. Appwrite
Functions). Keeps the Stage-1 router decision path but skips TreeSHAP, so the
heavy `shap`/`numba` import and the ~150MB SHAP explainer artifact are not
needed at inference time. `shap_values` comes back empty.

This trades the SHAP contribution chart for a cold start that fits within the
serverless readiness window; predictions, confidence, decision path, and all
metadata are unaffected.
"""

from typing import Dict, List, Any
import numpy as np

from src.models.stage1_router import Stage1Router


class LiteExplainer:
    """Decision-path-only explainer. Drop-in for PredictionExplainer's
    `explain_prediction` interface, minus SHAP values."""

    def __init__(self, router: Stage1Router, feature_names: List[str]):
        self.router = router
        self.feature_names = feature_names

    def explain_prediction(self, X_single: np.ndarray, category_id: int,
                           pred_idx: int) -> Dict[str, Any]:
        path = self.router.get_decision_path(X_single, self.feature_names)
        return {
            "decision_path": f"IF {path} THEN Routs To Category",
            "shap_values": {},
        }
