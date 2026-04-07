"""Population Stability Index (PSI) drift monitoring module.

Compares the distribution of incoming inference data 
against the distribution of the training data.
Features showing a PSI > 0.20 indicate substantial drift 
warranting pipeline retraining.
"""

from typing import Dict, List, Tuple
import numpy as np


class PSIMonitor:
    """Calculates PSI for tracking dataset distribution shift.

    Parameters
    ----------
    n_bins : int, default=10
        Number of quantiles to use for continuous features.
        For binary features (0/1), this is ignored.
    epsilon : float, default=1e-4
        Small constant to prevent zero-division in KL divergence math.
    """

    def __init__(self, n_bins: int = 10, epsilon: float = 1e-4):
        self.n_bins = n_bins
        self.epsilon = epsilon
        
        # Stores expected distribution (proportions) per feature
        self.expected_distributions_: Dict[int, np.ndarray] = {}
        # Stores cutoffs/bin edges per continuous feature
        self.cutoffs_: Dict[int, np.ndarray] = {}
        # Marks which features are binary
        self.is_binary_: Dict[int, bool] = {}

    def fit(self, X_train: np.ndarray) -> "PSIMonitor":
        """Compute the expected distributions from training data."""
        n_samples, n_features = X_train.shape
        
        for i in range(n_features):
            col = X_train[:, i]
            unique_vals = np.unique(col)
            
            # Treat features with 2 or 3 distinct values as discrete/binary
            if len(unique_vals) <= 3:
                self.is_binary_[i] = True
                
                # Standardize to 3 states: 0, 1, -1 for index tracking
                counts = {"-1": 0, "0": 0, "1": 0}
                for val in col:
                    if int(val) == -1: counts["-1"] += 1
                    elif int(val) == 1: counts["1"] += 1
                    else: counts["0"] += 1
                
                dist = np.array([counts["-1"], counts["0"], counts["1"]], dtype=np.float32)
                dist = np.clip(dist / n_samples, self.epsilon, None)
                dist = dist / dist.sum()
                self.expected_distributions_[i] = dist
                
            # Continuous features (like age, continuous lab values)
            else:
                self.is_binary_[i] = False
                
                # Extract percentiles using the expected bin count
                try:
                    percentiles = np.linspace(0, 100, self.n_bins + 1)
                    cutoffs = np.percentile(col, percentiles)
                    
                    # Ensure boundaries safely wrap all future potential values
                    cutoffs[0] = -np.inf
                    cutoffs[-1] = np.inf
                    self.cutoffs_[i] = cutoffs
                except Exception:
                    # Fallback to simple min-max bin boundaries if percentiles fail trivially
                    cutoffs = np.linspace(-np.inf, np.inf, self.n_bins + 1)
                    self.cutoffs_[i] = cutoffs
                
                # Digitizing training data gives 1-indexed bins
                binned = np.digitize(col, self.cutoffs_[i])
                counts = np.bincount(binned, minlength=self.n_bins + 2)[1:self.n_bins + 1]
                
                dist = np.clip(counts.astype(np.float32) / n_samples, self.epsilon, None)
                dist = dist / dist.sum()
                self.expected_distributions_[i] = dist
                
        return self

    def compute_psi(self, X_new: np.ndarray) -> Dict[int, float]:
        """Compute PSI for all features on an incoming batch.

        Parameters
        ----------
        X_new : np.ndarray
            Incoming feature matrix.

        Returns
        -------
        psi_scores : Dict[int, float]
            Mapping from feature index to PSI float.
        """
        n_samples = X_new.shape[0]
        # Need at least a minimal batch size to compute meaningful distributions
        if n_samples < 50:
            return {}
            
        psi_scores = {}
        for i in range(X_new.shape[1]):
            col = X_new[:, i]
            
            if self.is_binary_[i]:
                counts = {"-1": 0, "0": 0, "1": 0}
                for val in col:
                    if int(val) == -1: counts["-1"] += 1
                    elif int(val) == 1: counts["1"] += 1
                    else: counts["0"] += 1
                actual_dist = np.array([counts["-1"], counts["0"], counts["1"]], dtype=np.float32)
                
            else:
                binned = np.digitize(col, self.cutoffs_[i])
                counts = np.bincount(binned, minlength=self.n_bins + 2)[1:self.n_bins + 1]
                actual_dist = counts.astype(np.float32)
            
            actual_dist = np.clip(actual_dist / n_samples, self.epsilon, None)
            actual_dist = actual_dist / actual_dist.sum()
            expected_dist = self.expected_distributions_[i]
            
            # PSI = sum((Actual - Expected) * ln(Actual / Expected))
            psi = np.sum((actual_dist - expected_dist) * np.log(actual_dist / expected_dist))
            psi_scores[i] = float(psi)
            
        return psi_scores
