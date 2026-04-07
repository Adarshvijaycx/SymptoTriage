"""Hybrid augmentation pipeline (SMOTE-NC + CTGAN + Tomek links).

1. SMOTE-NC handles minority class oversampling.
2. CTGAN (optional) generates varied majority class samples to improve robustness.
3. TomekLinks removes borderline noisy samples overlapping the classes.
"""

from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from imblearn.under_sampling import TomekLinks

from src.augmentation.smote_handler import SMOTEHandler
from src.augmentation.ctgan_handler import CTGANHandler, CTGAN_AVAILABLE


class HybridAugmentor:
    """Performs full end-to-end curriculum augmentation.

    Parameters
    ----------
    cat_indices : list of int
        Column indices for categorical (discrete) features.
    use_smote : bool, default=True
        Whether to oversample minority classes.
    use_ctgan : bool, default=False
        Whether to generate additional majority class samples via GAN.
    use_tomek : bool, default=True
        Whether to apply Tomek Links to clean class boundaries.
    smote_kwargs : dict
        Arguments for SMOTEHandler.
    ctgan_kwargs : dict
        Arguments for CTGANHandler.
    """

    def __init__(self, cat_indices: List[int],
                 use_smote: bool = True,
                 use_ctgan: bool = False,
                 use_tomek: bool = True,
                 smote_kwargs: Optional[dict] = None,
                 ctgan_kwargs: Optional[dict] = None):
        self.cat_indices = cat_indices
        self.use_smote = use_smote
        self.use_ctgan = use_ctgan
        self.use_tomek = use_tomek

        _smote_kwargs = smote_kwargs or {"k_neighbors": 5}
        self.smote = SMOTEHandler(categorical_features=self.cat_indices, **_smote_kwargs)

        if self.use_ctgan and not CTGAN_AVAILABLE:
            print("Warning: CTGAN requested but package missing. Disabling CTGAN augmentor.")
            self.use_ctgan = False

        if self.use_ctgan:
            _ctgan_kwargs = ctgan_kwargs or {"epochs": 100}
            self.ctgan = CTGANHandler(discrete_columns=self.cat_indices, **_ctgan_kwargs)

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply the hybrid augmentation pipeline sequentially.

        Returns
        -------
        X_res, y_res derived augmented matrices.
        """
        X_res, y_res = X.copy(), y.copy()

        # Step 1: SMOTE-NC
        if self.use_smote:
            # Note: This will naturally ignore perfectly balanced datasets
            # when strategy is 'auto' (default)
            X_res, y_res = self.smote.fit_resample(X_res, y_res)

        # Step 2: CTGAN for the majority class
        if self.use_ctgan:
            # We sample an extra 10% from the largest class using CTGAN
            unique, counts = np.unique(y_res, return_counts=True)
            majority_class = unique[np.argmax(counts)]
            majority_mask = y_res == majority_class

            # Convert to DataFrame for CTGAN logic
            X_maj_df = pd.DataFrame(X_res[majority_mask])
            self.ctgan.fit(X_maj_df)

            n_samples = max(1, len(X_maj_df) // 10)
            X_syn_df = self.ctgan.sample(n_samples)

            X_syn = X_syn_df.values
            y_syn = np.full(n_samples, majority_class)

            X_res = np.vstack([X_res, X_syn])
            y_res = np.concatenate([y_res, y_syn])

        # Step 3: Tomek Links
        if self.use_tomek:
            tomek = TomekLinks()
            try:
                X_res, y_res = tomek.fit_resample(X_res, y_res)
            except Exception as e:
                # Tomek can sometimes fail if dataset variance rules are violated
                print(f"Tomek cleanup skipped due to exception: {e}")

        return X_res, y_res
