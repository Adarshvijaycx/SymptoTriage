"""Stage-2 fine-grained ensemble model (DT + RF + LightGBM).

Ensemble classifier executed *after* the Stage-1 router.
Trains one ensemble subset per disease category to provide
granular disease probability differentiation.
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from lightgbm import LGBMClassifier


class Stage2Ensemble:
    """Soft-voting ensemble per Stage-1 category.

    Parameters
    ----------
    dt_kwargs : dict
        Arguments for DecisionTreeClassifier base learner.
    rf_kwargs : dict
        Arguments for RandomForestClassifier base learner.
    lgb_kwargs : dict
        Arguments for LGBMClassifier base learner.
    weights : list of float, default=[1, 2, 3]
        Relative weights for soft voting [DT, RF, LGB].
    """

    def __init__(self, dt_kwargs: Dict[str, Any] = None,
                 rf_kwargs: Dict[str, Any] = None,
                 lgb_kwargs: Dict[str, Any] = None,
                 weights: List[float] = None):
        self.dt_kwargs = dt_kwargs or {"max_depth": 8, "random_state": 42}
        self.rf_kwargs = rf_kwargs or {"n_estimators": 200, "random_state": 42, "n_jobs": -1}
        self.lgb_kwargs = lgb_kwargs or {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "random_state": 42,
            "verbose": -1
        }
        self.weights = weights or [1.0, 2.0, 3.0]

        # Dictionary of fitted VotingClassifier ensembles mapped by Category ID
        self.ensembles_: Dict[int, VotingClassifier] = {}

    def fit(self, X: np.ndarray, y: np.ndarray, categories: np.ndarray) -> "Stage2Ensemble":
        """Fit a separate ensemble for each Stage-1 category.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix for all patients.
        y : np.ndarray
            Fine-grained disease labels for all patients.
        categories : np.ndarray
            Associated broad category ID for all patients.
        """
        unique_categories = np.unique(categories)

        for cat_id in unique_categories:
            # Mask data to only this category
            mask = categories == cat_id
            X_cat = X[mask]
            y_cat = y[mask]

            # Even if there's only 1 target class, fit so index matching works
            if len(np.unique(y_cat)) == 1:
                # We need a single weak learner for degenerate cases
                dt = DecisionTreeClassifier(max_depth=1)
                dt.fit(X_cat, y_cat)
                
                # Mock a VotingClassifier for API consistency
                ensemble = VotingClassifier(
                    estimators=[("dt", dt)],
                    voting="soft",
                    weights=[1.0]
                )
                ensemble.fit(X_cat, y_cat)
            else:
                # Normal ensemble logic
                dt = DecisionTreeClassifier(**self.dt_kwargs)
                rf = RandomForestClassifier(**self.rf_kwargs)
                lgb = LGBMClassifier(**self.lgb_kwargs)

                ensemble = VotingClassifier(
                    estimators=[("dt", dt), ("rf", rf), ("lgb", lgb)],
                    voting="soft",
                    weights=self.weights
                )
                ensemble.fit(X_cat, y_cat)

            self.ensembles_[cat_id] = ensemble
            print(f"[Stage-2 Ensemble] Fitted category {cat_id} on {len(X_cat)} samples.")

        return self

    def predict_proba(self, X: np.ndarray, category_id: int) -> np.ndarray:
        """Predict disease probabilities restricted to the specific category.

        Parameters
        ----------
        X : np.ndarray
            Patient feature vectors.
        category_id : int
            The scalar category ID evaluated from Stage-1 Router to
            process this batch of patients.

        Returns
        -------
        np.ndarray
            Probabilities over the set of fine-grained disease labels
            *within* the designated subset ensemble.
        """
        if category_id not in self.ensembles_:
            raise ValueError(f"Category ID {category_id} has no fitted ensemble.")
        return self.ensembles_[category_id].predict_proba(X)

    def predict(self, X: np.ndarray, category_id: int) -> np.ndarray:
        """Predict maximum-likelihood disease restricted to a category."""
        if category_id not in self.ensembles_:
            raise ValueError(f"Category ID {category_id} has no fitted ensemble.")
        return self.ensembles_[category_id].predict(X)
