"""Probability calibration utilities (Platt scaling).

Fixes systematic overconfidence in tree ensembles by fitting
a logistic regression layer over the base model's raw probabilities.
Includes the Expected Calibration Error (ECE) metric computation.
"""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
try:
    from sklearn.frozen import FrozenEstimator
except ImportError:
    # Fallback to older sklearn syntax if needed
    FrozenEstimator = None
from sklearn.ensemble import VotingClassifier

from src.models.stage2_ensemble import Stage2Ensemble


class ProbabilityCalibrator:
    """Wraps the Stage-2 Ensemble with Platt scaling calibration.

    We use CalibratedClassifierCV(method='sigmoid', cv='prefit')
    to scale the probability outputs of each category ensemble individually.
    """

    def __init__(self):
        # Maps Stage-1 category ID -> CalibratedClassifierCV wrapper around VotingClassifier
        self.calibrated_ensembles_: Dict[int, CalibratedClassifierCV] = {}

    def fit(self, ensemble_model: Stage2Ensemble, X_cal: np.ndarray,
            y_cal: np.ndarray, categories_cal: np.ndarray) -> "ProbabilityCalibrator":
        """Fit Platt scaling models on a held-out calibration set.

        Parameters
        ----------
        ensemble_model : Stage2Ensemble
            Pre-trained Stage-2 ensemble model.
        X_cal : np.ndarray
            Features from held-out calibration set.
        y_cal : np.ndarray
            Disease labels from held-out calibration set.
        categories_cal : np.ndarray
            Category IDs from held-out calibration set.
        """
        for cat_id, sub_ensemble in ensemble_model.ensembles_.items():
            mask = categories_cal == cat_id
            if not np.any(mask):
                # No calibration data for this category, retain uncalibrated ensemble
                self.calibrated_ensembles_[cat_id] = sub_ensemble
                continue

            X_c = X_cal[mask]
            y_c = y_cal[mask]

            # If the calibration subset only has 1 class, Platt scaling fails.
            if len(np.unique(y_c)) <= 1:
                self.calibrated_ensembles_[cat_id] = sub_ensemble
                continue

            if FrozenEstimator is not None:
                calibrated = CalibratedClassifierCV(FrozenEstimator(sub_ensemble), method="sigmoid")
            else:
                calibrated = CalibratedClassifierCV(sub_ensemble, method="sigmoid", cv="prefit")
            
            calibrated.fit(X_c, y_c)
            self.calibrated_ensembles_[cat_id] = calibrated

        return self

    def predict_proba(self, X: np.ndarray, category_id: int) -> np.ndarray:
        """Get calibrated class probabilities for a specific category."""
        return self.calibrated_ensembles_[category_id].predict_proba(X)


def compute_ece(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    """Compute the Expected Calibration Error (ECE).

    ECE = sum(|B_m| / N * |acc(B_m) - conf(B_m)|) over all bins.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        True integer labels.
    y_proba : np.ndarray, shape (n_samples, n_classes)
        Probability array for all classes.
    n_bins : int, default=10
        Number of confidence bins.

    Returns
    -------
    ece : float
        The ECE score (lower is better, < 0.05 is well-calibrated).
    """
    confidences = np.max(y_proba, axis=1)
    predictions = np.argmax(y_proba, axis=1)
    accuracies = (predictions == y_true)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    N = len(y_true)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Use right-inclusive boundaries properly
        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        pop_size = np.sum(in_bin)
        if pop_size > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (pop_size / N) * np.abs(bin_acc - bin_conf)

    return ece
